import os
import sys
import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from tqdm import tqdm

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import SymbolFields, MmtvDailyBar, YfBar1d
from scripts.common_function import get_database_engine, add_environment_args, get_session_maker

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"market_breadth_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_us_stocks_with_sector(session):
    """Get US stocks with sector and industry information"""
    query = sa.select(
        SymbolFields.symbol, SymbolFields.sector, SymbolFields.industry
    ).where(
        SymbolFields.country == "United States",
        SymbolFields.is_etf == 0,
    )
    result = session.execute(query)

    us_stocks_sector = [
        {"symbol": row.symbol, "sector": row.sector, "industry": row.industry}
        for row in result
    ]

    logger.info(f"Found {len(us_stocks_sector)} US stocks with valid sector and industry data (excluding ETFs)")
    
    sector_counts = defaultdict(int)
    industry_counts = defaultdict(int)
    for stock in us_stocks_sector:
        sector_counts[stock["sector"]] += 1
        industry_counts[stock["industry"]] += 1

    for sector, count in sector_counts.items():
        logger.info(f"Sector '{sector}' has {count} symbols")
    for industry, count in industry_counts.items():
        logger.info(f"Industry '{industry}' has {count} symbols")

    return us_stocks_sector, dict(sector_counts), dict(industry_counts)


def get_stock_data(engine, symbols, start_date, end_date):
    """Get stock price data for the specified symbols and date range"""
    placeholders = ",".join(f":symbol_{i}" for i in range(len(symbols)))
    # Use DATE() to compare only the date portion of timestamp
    query = sa.text(f"""
        SELECT symbol, timestamp, open, high, low, close
        FROM yf_daily_bar
        WHERE symbol IN ({placeholders})
        AND DATE(timestamp) BETWEEN DATE(:start_date) AND DATE(:end_date)
        ORDER BY symbol, timestamp
    """)

    try:
        with engine.connect() as conn:
            logger.info(f"Retrieving stock data from {start_date} to {end_date}")
            params = {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            }
            params.update({f"symbol_{i}": symbol for i, symbol in enumerate(symbols)})

            result = conn.execute(query, params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        logger.info(f"Retrieved {len(df)} rows of stock data")
        return df
    except Exception as e:
        logger.error(f"Error in get_stock_data: {str(e)}")
        raise


def calculate_market_breadth(
    stock_data, sector_info, calculation_date, days_average=20, total_counts=None
):
    """Calculate market breadth metrics based on stock data"""
    # Convert timestamp to date only and then filter
    stock_data['date_only'] = stock_data['timestamp'].dt.date
    calculation_date_only = calculation_date.date() if isinstance(calculation_date, datetime) else calculation_date
    
    # Use date_only for filtering instead of timestamp
    stock_data = stock_data[stock_data['date_only'] <= calculation_date_only]
    
    # Set index to timestamp for historical operations
    stock_data = stock_data.set_index("timestamp")

    logger.info(f"Calculating breadth metrics for {calculation_date}")
    
    # CHANGE 1: Initialize DataFrames with float zeros (0.0) instead of integer zeros (0)
    sector_breadth = defaultdict(
        lambda: pd.DataFrame(
            0.0, index=[calculation_date], columns=["open", "high", "low", "close"]
        )
    )
    industry_breadth = defaultdict(
        lambda: pd.DataFrame(
             0.0, index=[calculation_date], columns=["open", "high", "low", "close"]
        )
    )
    total_market_breadth = pd.DataFrame(
         0.0, index=[calculation_date], columns=["open", "high", "low", "close"]
    )

    sector_symbol_counts = defaultdict(int)
    industry_symbol_counts = defaultdict(int)
    total_stocks = 0
    total_above_sma = defaultdict(int)

    for symbol, sector, industry in sector_info:
        # Skip if sector or industry is None (this is a safety check, earlier filter should handle this)
        if not sector or not industry:
            continue

        df = stock_data[stock_data["symbol"] == symbol].copy()
        if df.empty:
            continue

        # Skip if we don't have enough data for SMA calculation
        if len(df) < days_average:
            continue

        sector_symbol_counts[sector] += 1
        industry_symbol_counts[industry] += 1
        total_stocks += 1

        for col in ["open", "high", "low", "close"]:
            df[f"{col}_SMA"] = df[col].rolling(window=days_average).mean()
            
            # Add debug logging to understand SMA calculations
            if len(df) > 0 and calculation_date in df.index:
                price = df.loc[calculation_date, col]
                sma = df.loc[calculation_date, f"{col}_SMA"]
                if not pd.isna(sma):
                    logger.debug(f"Symbol {symbol}: {col} price={price}, SMA={sma}, above_SMA={price > sma}")

            above_sma = (df[col] > df[f"{col}_SMA"]).astype(int)

            if calculation_date in df.index:
                is_above = above_sma.loc[calculation_date]
                sector_breadth[sector].loc[calculation_date, col] += is_above
                industry_breadth[industry].loc[calculation_date, col] += is_above
                total_above_sma[col] += is_above
            else:
                # Try to find any entry with the same date
                same_date_indices = df.index[df.index.date == calculation_date_only]
                if len(same_date_indices) > 0:
                    calculation_ts = same_date_indices[0]  # Use the first timestamp with matching date
                    is_above = above_sma.loc[calculation_ts]
                    sector_breadth[sector].loc[calculation_date, col] += is_above
                    industry_breadth[industry].loc[calculation_date, col] += is_above
                    total_above_sma[col] += is_above

    # Calculate Stock Percentages
    for sector, data in sector_breadth.items():
        sector_count = sector_symbol_counts[sector]
        if sector_count > 0:
            sector_breadth[sector] = (data / sector_count) * 100
        logger.info(
            f"Sector '{sector}' breadth: {sector_breadth[sector].iloc[0].to_dict()}"
        )

    for industry, data in industry_breadth.items():
        industry_count = industry_symbol_counts[industry]
        if industry_count > 0:
            industry_breadth[industry] = (data / industry_count) * 100
        logger.info(
            f"Industry '{industry}' breadth: {industry_breadth[industry].iloc[0].to_dict()}"
        )

    # Optional: When setting values, round to 4 decimal places for cleaner storage
    if total_stocks > 0:
        for col in ["open", "high", "low", "close"]:
            total_market_breadth.loc[calculation_date, col] = round(
                (total_above_sma[col] / total_stocks) * 100, 4
            )
    logger.info(f"Total market breadth: {total_market_breadth.iloc[0].to_dict()}")

    logger.info(f"Calculated market breadth for date {calculation_date}")
    logger.info(f"Number of sectors processed: {len(sector_breadth)}")
    logger.info(f"Number of industries processed: {len(industry_breadth)}")
    logger.info(f"Total stocks processed: {total_stocks}")

    return dict(sector_breadth), dict(industry_breadth), total_market_breadth


def update_market_breadth_table(session, breadth_data, field_type, calculation_date):
    """Update the market breadth data in the database"""
    if field_type == "total":
        breadth_df = breadth_data.reset_index()
        breadth_df["field_name"] = "TotalMarket"
        breadth_df["field_type"] = "total_market"
    else:
        dfs = []
        for field_name, data in breadth_data.items():
            df = data.reset_index()
            df["field_name"] = field_name
            dfs.append(df)

        if not dfs:
            logger.warning(f"No data to update for {field_type} on {calculation_date}")
            return

        breadth_df = pd.concat(dfs, ignore_index=True)
        breadth_df["field_type"] = field_type

    breadth_df = breadth_df.rename(columns={"index": "date"})
    breadth_df["date"] = pd.to_datetime(breadth_df["date"])

    processed_count = 0
    skipped_count = 0
    
    # Process each row to insert/update
    for _, row in breadth_df.iterrows():
        try:
            # Check if all OHLC values are 0 or very close to 0
            open_value = float(row['open'])
            high_value = float(row['high'])
            low_value = float(row['low'])
            close_value = float(row['close'])
            
            # Skip if all values are 0 or very close to 0 (using a small epsilon value)
            epsilon = 0.0001
            if (abs(open_value) < epsilon and 
                abs(high_value) < epsilon and 
                abs(low_value) < epsilon and 
                abs(close_value) < epsilon):
                logger.debug(
                    f"Skipping {field_type} data for {row['field_name']} on {row['date'].date()} - all OHLC values are zero"
                )
                skipped_count += 1
                continue
                
            # Check if record already exists
            existing = session.query(MmtvDailyBar).filter_by(
                date=row['date'].date(),
                field_name=row['field_name'],
                field_type=row['field_type']
            ).first()
            
            if existing:
                # Update existing record
                existing.open = open_value
                existing.high = high_value
                existing.low = low_value
                existing.close = close_value
            else:
                # Create new record
                new_record = MmtvDailyBar(
                    date=row['date'].date(),
                    field_name=row['field_name'],
                    field_type=row['field_type'],
                    open=open_value,
                    high=high_value,
                    low=low_value,
                    close=close_value
                )
                session.add(new_record)
            
            processed_count += 1
            logger.info(
                f"Inserted/Updated {field_type} data for {row['field_name']} on {row['date'].date()}"
            )
        except Exception as e:
            logger.error(f"Error updating record: {e}")
            continue
            
    # Commit all changes
    session.commit()
    logger.info(f"Updated {field_type} market breadth data for {calculation_date}")
    logger.info(f"Processed: {processed_count}, Skipped (all zeros): {skipped_count}")
    if field_type != "total":
        logger.info(f"Updated {len(breadth_data)} {field_type}s")


def process_date(args):
    """Process market breadth for a single date"""
    symbols, sector_info, stock_data, current_date, days_average, total_counts, env = args
    
    if current_date.weekday() >= 5:  # Skip weekends
        return
        
    logger.info(f"Processing date: {current_date} (Process: {os.getpid()})")

    if stock_data.empty:
        logger.warning(f"No stock data available for {current_date}")
        return

    try:
        # Setup database connection for this process
        engine = get_database_engine(env)
        Session = get_session_maker(env)
        session = Session()
        
        # Calculate market breadth
        sector_breadth, industry_breadth, total_market_breadth = calculate_market_breadth(
            stock_data, sector_info, current_date, days_average, total_counts
        )

        # Update database
        if sector_breadth:
            update_market_breadth_table(session, sector_breadth, "sector", current_date)
        if industry_breadth:
            update_market_breadth_table(session, industry_breadth, "industry", current_date)
        if not total_market_breadth.empty:
            update_market_breadth_table(session, total_market_breadth, "total", current_date)

        logger.info(f"Completed market breadth calculation and update for {current_date}")
        session.close()
    except Exception as e:
        logger.error(f"Error processing date {current_date}: {str(e)}")


def is_trading_day(date):
    """Check if a date is a trading day (M-F)"""
    return date.weekday() < 5


def get_trading_day_lookback(end_date, days, additional_buffer=30):
    """Get the date that is 'days' trading days before end_date with additional buffer days"""
    current_date = end_date
    trading_days = 0
    # Add buffer to ensure enough data for proper SMA calculation
    total_days_needed = days + additional_buffer
    
    while trading_days < total_days_needed:
        current_date -= timedelta(days=1)
        if is_trading_day(current_date):
            trading_days += 1
    return current_date


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Calculate market breadth")
    parser = add_environment_args(parser)  # Add standard environment selection
    
    parser.add_argument(
        "--last_day", action="store_true", help="Calculate for the last day"
    )
    parser.add_argument(
        "--last_week", action="store_true", help="Calculate for the last week"
    )
    parser.add_argument(
        "--last_month", action="store_true", help="Calculate for the last month"
    )
    parser.add_argument(
        "--start_date", type=str, help="Start date for calculation (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date", type=str, help="End date for calculation (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size for processing"
    )
    parser.add_argument(
        "--num_processes", type=int, default=4, help="Number of processes to use"
    )
    parser.add_argument(
        "--days_average",
        type=int,
        default=20,
        help="Number of trading days for SMA calculation",
    )
    return parser.parse_args()


def get_date_range(args):
    """Determine the date range based on command line arguments"""
    end_date = datetime.now().date()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    if args.last_day:
        start_date = end_date - timedelta(days=1)
        while not is_trading_day(start_date):
            start_date -= timedelta(days=1)
    elif args.last_week:
        start_date = end_date - timedelta(days=7)
        while not is_trading_day(start_date):
            start_date -= timedelta(days=1)
    elif args.last_month:
        start_date = end_date - timedelta(days=30)
        while not is_trading_day(start_date):
            start_date -= timedelta(days=1)
    elif args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        start_date = datetime(2025, 4, 1).date()

    # Get lookback date with additional buffer days to ensure enough historical data
    lookback_start = get_trading_day_lookback(start_date, args.days_average, additional_buffer=30)
    return lookback_start, start_date, end_date


def main():
    """Main function to run the market breadth calculation"""
    try:
        # Parse arguments
        args = parse_arguments()
        lookback_start, start_date, end_date = get_date_range(args)
        
        # Setup database connections
        engine = get_database_engine(args.env)
        Session = get_session_maker(args.env)
        session = Session()

        env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
        logger.info(f"Starting market breadth calculation using {env_name} database")
        logger.info(f"Calculating market breadth from {start_date} to {end_date}")
        logger.info(f"Using data from {lookback_start} for SMA calculation ({args.days_average} trading days)")

        # Get US stocks with sector data
        us_stocks, sector_counts, industry_counts = get_us_stocks_with_sector(session)
        symbols = [str(stock["symbol"]) for stock in us_stocks]
        sector_info = [
            (str(stock["symbol"]), stock["sector"], stock["industry"])
            for stock in us_stocks
        ]

        logger.info(f"Retrieved {len(symbols)} symbols in total")
        total_counts = {"sectors": sector_counts, "industries": industry_counts}

        # Fetch stock data for all symbols
        stock_data = get_stock_data(engine, symbols, lookback_start, end_date)
        logger.info(f"Loaded {len(stock_data)} stock data points")

        # Prepare arguments for multiprocessing
        date_args = [
            (
                symbols,
                sector_info,
                stock_data,
                current_date,
                args.days_average,
                total_counts,
                args.env,
            )
            for current_date in pd.date_range(start_date, end_date)
            if current_date.weekday() < 5
        ]

        # Process dates in parallel
        with Pool(processes=args.num_processes) as pool:
            list(
                tqdm(
                    pool.imap(process_date, date_args),
                    total=len(date_args),
                    desc="Processing dates",
                )
            )

        session.close()
        logger.info("Market breadth calculation and update completed.")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    main()

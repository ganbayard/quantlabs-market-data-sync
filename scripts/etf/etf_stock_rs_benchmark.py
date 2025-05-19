import os
import sys
import logging
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import datetime, timedelta
import concurrent.futures
import multiprocessing

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import IShareETF, IShareETFHolding, YfBar1d, EtfStockRsBenchmark
from scripts.common_function import get_database_engine, add_environment_args, get_session_maker

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"etf_stock_rs_benchmark_{pd.Timestamp.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Relative Strength Parameters
RELATIVE_STRENGTH_BENCHMARK = 'SPY'
RELATIVE_STRENGTH_WINDOW = 63
TOTAL_LOOKBACK_DAYS = 100

def read_stock_symbols(file_path=None):
    """Read stock symbols from the specified file"""
    default_path = project_root / "scripts/symbols/stock_symbols.txt"
    path_to_use = Path(file_path) if file_path else default_path
    
    try:
        with open(path_to_use, 'r') as f:
            content = f.read()
            symbols = [symbol.strip() for symbol in content.split(',') if symbol.strip()]
            logger.info(f"Read {len(symbols)} stock symbols from {path_to_use}")
            return symbols
    except FileNotFoundError:
        logger.error(f"Stock symbols file not found: {path_to_use}")
        return []

def get_etf_holdings_for_stocks(session, stock_symbols, full_output=False):
    """
    Query ETF holdings for the given stock symbols
    If full_output is False, only return the ETF with highest market value for each stock
    """
    if not stock_symbols:
        logger.warning("No stock symbols provided to query")
        return pd.DataFrame()
    
    # Create a query to find ETF holdings for the given stock symbols
    query = """
    SELECT 
        h.ticker AS stock_symbol, 
        e.ticker AS etf_symbol, 
        h.ishare_etf_id, 
        h.market_value,
        h.weight
    FROM 
        ishare_etf_holding h
    JOIN 
        ishare_etf e ON h.ishare_etf_id = e.id
    WHERE 
        h.ticker IN :symbols
    ORDER BY 
        h.ticker, h.market_value DESC
    """
    
    try:
        # Execute the query
        result = pd.read_sql(text(query), session.bind, params={"symbols": tuple(stock_symbols)})
        
        if result.empty:
            logger.warning("No ETF holdings found for the provided stock symbols")
            return pd.DataFrame()
        
        logger.info(f"Found {len(result)} ETF holdings for {len(result['stock_symbol'].unique())} unique stocks")
        
        # If not full output, keep only the ETF with highest market value for each stock
        if not full_output:
            result = result.sort_values(['stock_symbol', 'market_value'], ascending=[True, False])
            result = result.drop_duplicates(subset='stock_symbol', keep='first')
            logger.info(f"Filtered to top ETF for each stock: {len(result)} entries")
        
        return result
    
    except Exception as e:
        logger.error(f"Error querying ETF holdings: {str(e)}")
        return pd.DataFrame()

def get_price_data(session, symbols, lookback_days=None):
    """Retrieve price data for the given symbols from YfBar1d table"""
    # Calculate required lookback period for RS calculations
    required_days = RELATIVE_STRENGTH_WINDOW + TOTAL_LOOKBACK_DAYS
    
    logger.info(f"Retrieving price data for {len(symbols)} symbols")
    logger.info(f"Required historical data: {required_days} days")
    
    # Query the database for the total available historical data for the benchmark
    benchmark_query = """
    SELECT COUNT(*) as count
    FROM yf_daily_bar
    WHERE symbol = :benchmark
    """
    
    try:
        benchmark_count_result = pd.read_sql(text(benchmark_query), session.bind, params={"benchmark": RELATIVE_STRENGTH_BENCHMARK})
        available_days = benchmark_count_result['count'].iloc[0]
        logger.info(f"Total available days for {RELATIVE_STRENGTH_BENCHMARK}: {available_days}")
        
        # Adjust required_days if necessary based on available data
        if available_days < required_days:
            logger.warning(f"Adjusting required days from {required_days} to {available_days} based on available data")
            adjusted_days = available_days
            # If we have fewer than RELATIVE_STRENGTH_WINDOW days, we can't calculate RS at all
            if adjusted_days <= RELATIVE_STRENGTH_WINDOW:
                logger.error(f"Not enough data to calculate RS (need at least {RELATIVE_STRENGTH_WINDOW + 1} days)")
                return {}
        else:
            adjusted_days = required_days
        
    except Exception as e:
        logger.error(f"Error checking benchmark data: {str(e)}")
        # Use a conservative approach and continue with original days
        adjusted_days = required_days
    
    end_date = datetime.now()
    
    # Add benchmark symbol if not already in the list
    if RELATIVE_STRENGTH_BENCHMARK not in symbols:
        symbols.append(RELATIVE_STRENGTH_BENCHMARK)
    
    # Create a query to get price data for the last adjusted_days days
    query = """
    WITH ranked_bars AS (
        SELECT 
            symbol, 
            timestamp, 
            open, 
            high, 
            low, 
            close, 
            volume,
            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as row_num
        FROM 
            yf_daily_bar
        WHERE 
            symbol IN :symbols
    )
    SELECT 
        symbol, 
        timestamp, 
        open, 
        high, 
        low, 
        close, 
        volume
    FROM 
        ranked_bars
    WHERE 
        row_num <= :days
    ORDER BY 
        symbol, timestamp
    """
    
    try:
        # Execute the query
        logger.info(f"Executing database query for {len(symbols)} symbols, fetching last {adjusted_days} days...")
        result = pd.read_sql(text(query), session.bind, params={
            "symbols": tuple(symbols), 
            "days": adjusted_days
        })
        
        if result.empty:
            logger.warning("No price data found in the database for the requested symbols")
            return {}
        
        logger.info(f"Retrieved {len(result)} total price records from database")
        
        # Convert to dictionary of dataframes by symbol
        assets = {}
        symbols_with_insufficient_data = []
        
        for symbol in symbols:
            symbol_data = result[result['symbol'] == symbol].copy()
            if not symbol_data.empty:
                symbol_data.set_index('timestamp', inplace=True)
                symbol_data.sort_index(inplace=True)
                assets[symbol] = symbol_data
                
                # Check if we have enough data for RS calculation
                if len(symbol_data) < RELATIVE_STRENGTH_WINDOW + 1:  # Need at least window + 1 days
                    symbols_with_insufficient_data.append((symbol, len(symbol_data)))
        
        # Report on symbols with insufficient data
        if symbols_with_insufficient_data:
            logger.warning(f"{len(symbols_with_insufficient_data)} symbols have insufficient data for RS calculations (need at least {RELATIVE_STRENGTH_WINDOW + 1} days):")
            for symbol, count in symbols_with_insufficient_data[:10]:  # Show first 10
                logger.warning(f"  {symbol}: {count} records")
            if len(symbols_with_insufficient_data) > 10:
                logger.warning(f"  ...and {len(symbols_with_insufficient_data) - 10} more")
        
        # Check if benchmark data exists and is sufficient
        if RELATIVE_STRENGTH_BENCHMARK not in assets:
            logger.error(f"Benchmark {RELATIVE_STRENGTH_BENCHMARK} has no price data in database! Cannot calculate relative strength.")
            return {}
        
        benchmark_data_count = len(assets[RELATIVE_STRENGTH_BENCHMARK])
        if benchmark_data_count <= RELATIVE_STRENGTH_WINDOW:
            logger.error(f"Benchmark {RELATIVE_STRENGTH_BENCHMARK} has insufficient data ({benchmark_data_count} records). Need at least {RELATIVE_STRENGTH_WINDOW + 1}.")
            return {}
            
        logger.info(f"Retrieved price data for {len(assets)} symbols out of {len(symbols)} requested")
        logger.info(f"Benchmark {RELATIVE_STRENGTH_BENCHMARK} has {benchmark_data_count} records, sufficient for calculations")
        return assets
        
    except Exception as e:
        logger.error(f"Error retrieving price data: {str(e)}")
        return {}

def calculate_relative_strength(assets, num_workers=4):
    """Calculate relative strength metrics following the original methodology"""
    logger.info(f"Calculating relative strength metrics using {num_workers} workers...")
    
    # Check data sufficiency
    logger.info(f"Number of symbols with price data: {len(assets)}")
    
    # Get benchmark data
    if RELATIVE_STRENGTH_BENCHMARK not in assets:
        logger.error(f"Benchmark {RELATIVE_STRENGTH_BENCHMARK} not found in price data")
        return {}
        
    benchmark_df = assets[RELATIVE_STRENGTH_BENCHMARK]
    logger.info(f"Benchmark date range: {benchmark_df.index.min()} to {benchmark_df.index.max()}")
    
    try:
        # Calculate benchmark percentage changes (exactly as in the reference code)
        benchmark_df['RSPct'] = 100 * (benchmark_df['close'] - benchmark_df['close'].shift(RELATIVE_STRENGTH_WINDOW)) / benchmark_df['close'].shift(RELATIVE_STRENGTH_WINDOW)
        logger.info(f"Successfully calculated benchmark RS percentages")
    except Exception as e:
        logger.error(f"Error calculating benchmark percentages: {str(e)}")
        return {}
    
    # Dictionary to store full RS value lists for each symbol
    rs_data = {}
    benchmark_latest_date = benchmark_df.index[-1]
    logger.info(f"Benchmark latest date: {benchmark_latest_date}")
    
    # Prepare list of symbols to process (exclude benchmark)
    symbols_to_process = [symbol for symbol in assets.keys() if symbol != RELATIVE_STRENGTH_BENCHMARK]
    
    # Define worker function for parallel processing
    def process_symbol(symbol):
        try:
            df = assets[symbol]
            
            # Check if data is fresh enough (within 4 days of benchmark)
            if df.empty:
                return None
                
            symbol_latest_date = df.index[-1]
            days_difference = (benchmark_latest_date - symbol_latest_date).days
            
            if days_difference > 4:
                return None
            
            # Calculate original RS metrics for each day
            df['RSPct'] = 100 * (df['close'] - df['close'].shift(RELATIVE_STRENGTH_WINDOW)) / df['close'].shift(RELATIVE_STRENGTH_WINDOW)
            df[f"RS_{RELATIVE_STRENGTH_BENCHMARK}"] = df['RSPct'] / benchmark_df['RSPct']
            
            # Check if we have enough data - at least the window size plus lookback periods
            min_required_points = RELATIVE_STRENGTH_WINDOW + 1  # Minimum to calculate RSPct
            
            # Get the non-NaN RS values
            rs_values = df[f"RS_{RELATIVE_STRENGTH_BENCHMARK}"].dropna()
            
            # Only proceed if we have some valid RS values
            if len(rs_values) > 0:
                # Adjust lookback periods based on available data
                lookback_100 = min(len(rs_values), TOTAL_LOOKBACK_DAYS)
                lookback_50 = min(len(rs_values), 50)
                lookback_20 = min(len(rs_values), 20)
                lookback_5 = min(len(rs_values), 5)
                
                # Store detailed RS data for this symbol
                symbol_data = {
                    'current': rs_values.iloc[-1],
                    'rs_100': rs_values.iloc[-lookback_100:].tolist(),
                    'rs_50': rs_values.iloc[-lookback_50:].tolist(),
                    'rs_20': rs_values.iloc[-lookback_20:].tolist(),
                    'rs_5': rs_values.iloc[-lookback_5:].tolist(),
                    'mean_100': rs_values.iloc[-lookback_100:].mean(),
                    'mean_50': rs_values.iloc[-lookback_50:].mean(),
                    'mean_20': rs_values.iloc[-lookback_20:].mean(),
                    'mean_5': rs_values.iloc[-lookback_5:].mean()
                }
                
                # Calculate positive percentage values
                symbol_data['pos_pct_100'] = round(sum(1 for x in symbol_data['rs_100'] if x > 0) / len(symbol_data['rs_100']) * 100, 2)
                symbol_data['pos_pct_50'] = round(sum(1 for x in symbol_data['rs_50'] if x > 0) / len(symbol_data['rs_50']) * 100, 2)
                symbol_data['pos_pct_20'] = round(sum(1 for x in symbol_data['rs_20'] if x > 0) / len(symbol_data['rs_20']) * 100, 2)
                symbol_data['pos_pct_5'] = round(sum(1 for x in symbol_data['rs_5'] if x > 0) / len(symbol_data['rs_5']) * 100, 2)
                
                return (symbol, symbol_data)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error calculating RS for {symbol}: {str(e)}")
            return None
    
    # Process symbols in parallel
    successful = 0
    skipped = 0
    
    # Use either ProcessPoolExecutor or ThreadPoolExecutor based on your needs
    # ProcessPoolExecutor is better for CPU-bound tasks, ThreadPoolExecutor for I/O-bound
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_symbol = {executor.submit(process_symbol, symbol): symbol for symbol in symbols_to_process}
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol)):
            if i > 0 and i % 1000 == 0:
                logger.info(f"Processed {i} symbols...")
                
            result = future.result()
            if result:
                symbol, data = result
                rs_data[symbol] = data
                successful += 1
            else:
                skipped += 1
    
    processed = successful + skipped
    logger.info(f"Completed RS calculations. Processed {processed} symbols, successfully calculated {successful}, skipped {skipped}")
    
    if successful == 0:
        logger.warning("No symbols had sufficient data for RS calculations!")
        logger.warning(f"Please ensure you have loaded enough historical price data (at least {RELATIVE_STRENGTH_WINDOW + 1} days)")
        
    return rs_data

def check_already_processed_today(session):
    """
    Check if data has already been processed today to avoid duplicate entries
    Returns True if data for today already exists, False otherwise
    """
    try:
        today = datetime.now().date()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today, datetime.max.time())
        
        # Query to check if there are any records created today
        query = """
        SELECT COUNT(*) as count
        FROM etf_stock_rs_benchmark
        WHERE created_at BETWEEN :start_date AND :end_date
        """
        
        result = pd.read_sql(text(query), session.bind, params={
            "start_date": today_start, 
            "end_date": today_end
        })
        
        count = result['count'].iloc[0]
        if count > 0:
            logger.info(f"Found {count} records already processed today ({today})")
            return True
        else:
            logger.info(f"No records found for today ({today}), proceeding with processing")
            return False
            
    except Exception as e:
        logger.error(f"Error checking if data already processed today: {str(e)}")
        # If there's an error, assume not processed to be safe
        return False

def save_rs_data_to_db(session, holdings_df, rs_data, force_save=False):
    """Save the calculated RS data to the etf_stock_rs_benchmark table"""
    if not rs_data or holdings_df.empty:
        logger.warning("No data to save to the database")
        return 0
    
    # Check if data has already been processed today
    if not force_save and check_already_processed_today(session):
        logger.warning("Data has already been processed today. Use --force flag to override.")
        return 0
    
    logger.info("Saving RS data to etf_stock_rs_benchmark table...")
    
    # Create a list to hold the records to be inserted
    records_to_insert = []
    current_time = datetime.now()
    
    # Process each row in the holdings dataframe
    for index, row in holdings_df.iterrows():
        stock_symbol = row['stock_symbol']
        etf_symbol = row['etf_symbol']
        ishare_etf_id = row['ishare_etf_id']
        market_value = row['market_value']
        weight = row['weight']
        
        # Skip if no RS data for this symbol
        if stock_symbol not in rs_data:
            continue
        
        # Get RS data for this symbol
        symbol_rs = rs_data[stock_symbol]
        
        # Create a new record
        record = EtfStockRsBenchmark(
            stock_symbol=stock_symbol,
            etf_symbol=etf_symbol,
            ishare_etf_id=ishare_etf_id,
            market_value=market_value,
            weight=weight,
            rs_current=symbol_rs['current'],
            rs_mean_100d=symbol_rs['mean_100'],
            rs_mean_50d=symbol_rs['mean_50'],
            rs_mean_20d=symbol_rs['mean_20'],
            rs_mean_5d=symbol_rs['mean_5'],
            rs_pos_pct_100d=symbol_rs['pos_pct_100'],
            rs_pos_pct_50d=symbol_rs['pos_pct_50'],
            rs_pos_pct_20d=symbol_rs['pos_pct_20'],
            rs_pos_pct_5d=symbol_rs['pos_pct_5'],
            rs_values_100d=json.dumps(symbol_rs['rs_100']),
            rs_values_50d=json.dumps(symbol_rs['rs_50']),
            rs_values_20d=json.dumps(symbol_rs['rs_20']),
            rs_values_5d=json.dumps(symbol_rs['rs_5']),
            benchmark_symbol=RELATIVE_STRENGTH_BENCHMARK,
            created_at=current_time,
            updated_at=current_time
        )
        
        records_to_insert.append(record)
    
    # Insert the records in batches
    try:
        batch_size = 1000
        total_records = len(records_to_insert)
        
        for i in range(0, total_records, batch_size):
            batch = records_to_insert[i:i+batch_size]
            session.add_all(batch)
            session.commit()
            logger.info(f"Inserted batch {i//batch_size + 1}/{(total_records+batch_size-1)//batch_size}: {len(batch)} records")
        
        logger.info(f"Successfully saved {total_records} records to the database")
        return total_records
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving data to database: {str(e)}")
        return 0

def main():
    """Main function to analyze ETF holdings for stocks and save to database"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETF Stock RS Benchmark Analysis')
    parser = add_environment_args(parser)
    parser.add_argument('--symbols', type=str, help='Path to file with comma-separated stock symbols')
    parser.add_argument('--full', action='store_true', help='Show all ETFs for each stock, not just the one with highest market value')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads for calculations (default: 4)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to database')
    parser.add_argument('--force', action='store_true', help='Force save to database even if already processed today')
    args = parser.parse_args()
    
    # Setup database connection based on environment
    engine = get_database_engine(args.env)
    Session = get_session_maker(args.env)
    session = Session()
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting ETF Stock RS Benchmark Analysis using {env_name} database with {args.workers} worker threads")

    try:
        # Read stock symbols
        stock_symbols = read_stock_symbols(args.symbols)
        if not stock_symbols:
            logger.error("No stock symbols to process. Exiting.")
            return
            
        # Get ETF holdings for these stocks
        holdings_df = get_etf_holdings_for_stocks(session, stock_symbols, args.full)
        if holdings_df.empty:
            logger.warning("No ETF holdings found for the provided symbols. Exiting.")
            return
        
        # Get price data for relative strength calculations
        price_data = get_price_data(session, stock_symbols) 
        if price_data:
            logger.info(f"Retrieved price data for {len(price_data)} symbols")
            
            # Output details about a few symbols as a sample
            sample_symbols = [s for s in price_data.keys()][:5]  # First 5 symbols
            for symbol in sample_symbols:
                df = price_data[symbol]
                logger.info(f"Symbol {symbol}: {len(df)} records, date range: {df.index.min()} to {df.index.max()}")
            
            # Show benchmark details
            if RELATIVE_STRENGTH_BENCHMARK in price_data:
                bm = price_data[RELATIVE_STRENGTH_BENCHMARK]
                logger.info(f"Benchmark {RELATIVE_STRENGTH_BENCHMARK}: {len(bm)} records, date range: {bm.index.min()} to {bm.index.max()}")
        
            # Calculate relative strength metrics
            rs_data = calculate_relative_strength(price_data, args.workers)
            
            # Add RS metrics to holdings dataframe
            if rs_data:
                logger.info("Adding relative strength metrics to holdings data")
                
                # Create columns for RS metrics
                rs_columns = [
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}",
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100", 
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_50", 
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_20", 
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_5",
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100_POS_PCT",
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_50_POS_PCT", 
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_20_POS_PCT", 
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_5_POS_PCT"
                ]
                
                for col in rs_columns:
                    holdings_df[col] = None
                
                # Add list columns
                list_columns = [
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100_LIST",
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_50_LIST", 
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_20_LIST", 
                    f"RS_{RELATIVE_STRENGTH_BENCHMARK}_5_LIST"
                ]
                
                for col in list_columns:
                    holdings_df[col] = None
                
                # Fill values from RS data
                for index, row in holdings_df.iterrows():
                    symbol = row['stock_symbol']
                    if symbol in rs_data:
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}"] = rs_data[symbol]['current']
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100"] = rs_data[symbol]['mean_100']
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_50"] = rs_data[symbol]['mean_50']
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_20"] = rs_data[symbol]['mean_20']
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_5"] = rs_data[symbol]['mean_5']
                        
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100_POS_PCT"] = rs_data[symbol]['pos_pct_100']
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_50_POS_PCT"] = rs_data[symbol]['pos_pct_50']
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_20_POS_PCT"] = rs_data[symbol]['pos_pct_20']
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_5_POS_PCT"] = rs_data[symbol]['pos_pct_5']
                        
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100_LIST"] = json.dumps(rs_data[symbol]['rs_100'])
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_50_LIST"] = json.dumps(rs_data[symbol]['rs_50'])
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_20_LIST"] = json.dumps(rs_data[symbol]['rs_20'])
                        holdings_df.at[index, f"RS_{RELATIVE_STRENGTH_BENCHMARK}_5_LIST"] = json.dumps(rs_data[symbol]['rs_5'])
                
                # Save data to database unless --no-save flag is used
                if not args.no_save:
                    total_saved = save_rs_data_to_db(session, holdings_df, rs_data, force_save=args.force)
                    if total_saved > 0:
                        logger.info(f"Saved {total_saved} records to etf_stock_rs_benchmark table")
                    else:
                        logger.info("No new records were saved to the database")
        
        # Format output columns
        base_columns = ['stock_symbol', 'etf_symbol', 'ishare_etf_id', 'market_value', 'weight']
        rs_display_columns = [
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}", 
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100", 
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_50", 
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_20", 
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_5",
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100_POS_PCT",
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_50_POS_PCT", 
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_20_POS_PCT", 
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_5_POS_PCT",
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100_LIST",
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_50_LIST", 
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_20_LIST", 
            f"RS_{RELATIVE_STRENGTH_BENCHMARK}_5_LIST"
        ]
        
        # Set pandas display options to show all columns and more rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        # Combine base columns and RS display columns
        display_columns = base_columns + rs_display_columns
        available_columns = [col for col in display_columns if col in holdings_df.columns]
            
        # Display results
        print("\nETF Holdings for Stocks with Relative Strength:")
        
        # Check if any RS data was calculated
        if 'rs_data' in locals() and rs_data:
            # Sort by RS_SPY_100 for display
            if f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100" in holdings_df.columns:
                holdings_df_display = holdings_df.sort_values(
                    by=f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100", 
                    ascending=False, 
                    na_position='last'
                ).copy()
            else:
                holdings_df_display = holdings_df.copy()
                
            # Only show the first few columns for display (omit the LIST columns which are too large)
            display_cols = [col for col in available_columns if not col.endswith('_LIST')]
            print(holdings_df_display[display_cols].head(10))
            print(f"\nTotal records: {len(holdings_df)}")
            
            # Print a message about the data
            print("\nRS columns explanation:")
            print(f"RS_{RELATIVE_STRENGTH_BENCHMARK}: Current relative strength value")
            print(f"RS_{RELATIVE_STRENGTH_BENCHMARK}_100: 100-day mean relative strength")
            print(f"RS_{RELATIVE_STRENGTH_BENCHMARK}_50: 50-day mean relative strength")
            print(f"RS_{RELATIVE_STRENGTH_BENCHMARK}_20: 20-day mean relative strength") 
            print(f"RS_{RELATIVE_STRENGTH_BENCHMARK}_5: 5-day mean relative strength")
            print(f"RS_{RELATIVE_STRENGTH_BENCHMARK}_XX_POS_PCT: Percentage of positive RS values in the period")
            print(f"RS_{RELATIVE_STRENGTH_BENCHMARK}_XX_LIST: JSON-encoded list of all RS values for the period")
            
            # Print database saving status
            if not args.no_save:
                if 'total_saved' in locals() and total_saved > 0:
                    print(f"\nSuccessfully saved {total_saved} records to etf_stock_rs_benchmark table")
                else:
                    print("\nNo records were saved to the database")
            else:
                print("\nDatabase saving was skipped (--no-save flag used)")
        else:
            print("Note: No relative strength data was calculated. Only showing basic ETF holdings.")
            print(holdings_df[base_columns].head(10))
            print(f"\nTotal records: {len(holdings_df)}")
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            holdings_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error in ETF Stock RS Benchmark analysis: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
from sqlalchemy import text
from sklearn.linear_model import LinearRegression
import concurrent.futures

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import YfBar1d, EquityRiskProfile
from scripts.common_function import get_database_engine, add_environment_args, get_session_maker

# Constants and Configuration
BENCHMARK_SYMBOL = 'SPY'              # Benchmark for beta calculation
LOOKBACK_DAYS = 756                   # Approximately 3 years of trading days (252 * 3)
MIN_LOOKBACK_DAYS = 252               # Minimum lookback period (1 year)
MIN_PRICE = 2.0                       # Minimum price threshold for stocks

# Risk classification thresholds
CONSERVATIVE_VOL_THRESHOLD = 0.15     # Volatility thresholds
MODERATE_VOL_THRESHOLD = 0.21
CONSERVATIVE_BETA_THRESHOLD = 0.8     # Beta thresholds
MODERATE_BETA_THRESHOLD = 1.20
CONSERVATIVE_MDD_THRESHOLD = -0.15    # Max drawdown thresholds
MODERATE_MDD_THRESHOLD = -0.21
CONSERVATIVE_ADR_THRESHOLD = 1.5      # ADR thresholds (percentage)
MODERATE_ADR_THRESHOLD = 3.0
CONSERVATIVE_SCORE_THRESHOLD = 1.75   # Final classification score thresholds
MODERATE_SCORE_THRESHOLD = 2.3

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"equity_risk_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_symbols(session, include_stocks=True, include_etfs=False):
    """
    Get symbols from the SymbolFields table with their ETF/stock classification
    
    Args:
        session: Database session
        include_stocks: Whether to include stocks
        include_etfs: Whether to include ETFs
        
    Returns:
        dict: Dictionary with symbols as keys and is_etf flag as values
    """
    conditions = []
    if include_stocks:
        conditions.append("is_etf = 0")
    if include_etfs:
        conditions.append("is_etf = 1")
        
    if not conditions:
        logger.warning("No symbol types selected (stocks or ETFs)")
        return {}
        
    where_clause = " OR ".join(conditions)
    
    query = f"""
    SELECT DISTINCT symbol, is_etf FROM symbol_fields 
    WHERE {where_clause}
    ORDER BY symbol
    """
    
    try:
        result = pd.read_sql(text(query), session.bind)
        symbols_dict = dict(zip(result['symbol'], result['is_etf'] == 1))
        
        stock_count = sum(1 for is_etf in symbols_dict.values() if not is_etf)
        etf_count = sum(1 for is_etf in symbols_dict.values() if is_etf)
        
        logger.info(f"Found {len(symbols_dict)} symbols in database ({stock_count} stocks, {etf_count} ETFs)")
        return symbols_dict
    except Exception as e:
        logger.error(f"Error fetching symbols from database: {e}")
        return {}

def get_price_data(session, symbols, benchmark_symbol, lookback_days):
    """
    Retrieve price data for the given symbols from YfBar1d table
    
    Returns:
        dict: Dictionary with symbols as keys and DataFrames with price data as values
    """
    logger.info(f"Retrieving price data for {len(symbols) + 1} symbols (including benchmark {benchmark_symbol})")
    
    # Make sure benchmark is included
    if benchmark_symbol not in symbols:
        symbols_with_benchmark = symbols + [benchmark_symbol]
    else:
        symbols_with_benchmark = symbols.copy()
    
    # Get current date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 30)  # Extra days to account for missing data
    
    price_data = {}
    
    # Process symbols in batches to avoid excessive memory usage
    batch_size = 500
    total_batches = (len(symbols_with_benchmark) + batch_size - 1) // batch_size
    
    for i in range(0, len(symbols_with_benchmark), batch_size):
        batch = symbols_with_benchmark[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch)} symbols)")
        
        # Query to get price data
        query = """
        SELECT symbol, timestamp, `open`, high, low, `close`, volume 
        FROM yf_daily_bar
        WHERE symbol IN :symbols
          AND timestamp BETWEEN :start_date AND :end_date
        ORDER BY symbol, timestamp
        """
        
        try:
            df = pd.read_sql(
                text(query), 
                session.bind, 
                params={
                    'symbols': tuple(batch),
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d')
                }
            )
            
            if df.empty:
                logger.warning(f"No price data found for batch")
                continue
                
            # Convert columns
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
                
            # Rename for consistency with original script
            df = df.rename(columns={
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Group by symbol and create a DataFrame for each
            grouped = df.groupby('symbol')
            for symbol, group in grouped:
                # Set Date as index
                symbol_df = group.set_index('Date')
                symbol_df = symbol_df.sort_index()
                price_data[symbol] = symbol_df
                
            logger.info(f"Retrieved data for {len(grouped)} symbols in batch")
            
        except Exception as e:
            logger.error(f"Error retrieving price data for batch: {e}")
    
    # Check if we have the benchmark data
    if benchmark_symbol not in price_data:
        logger.error(f"No data found for benchmark {benchmark_symbol}. Cannot proceed with risk calculations.")
        return {}
    
    logger.info(f"Successfully retrieved price data for {len(price_data)} symbols")
    return price_data

def process_symbol(symbol, df, benchmark_df, benchmark_returns, is_etf=False):
    """
    Calculate risk metrics for a single symbol
    
    Args:
        symbol: The ticker symbol
        df: DataFrame with price data
        benchmark_df: DataFrame with benchmark price data
        benchmark_returns: Series with benchmark returns
        is_etf: Boolean indicating if the symbol is an ETF
        
    Returns:
        dict: Dictionary of risk metrics or None if processing fails
    """
    try:
        # Check if data is fresh enough (within 7 days of benchmark)
        if df.empty:
            return None
            
        symbol_latest_date = df.index[-1]
        benchmark_latest_date = benchmark_df.index[-1]
        days_difference = (benchmark_latest_date - symbol_latest_date).days
        
        if days_difference > 7:
            logger.debug(f"{symbol}: Data is {days_difference} days old. Skipping.")
            return None
        
        # Check if price is above minimum threshold
        latest_price = df['Close'].iloc[-1]
        if latest_price < MIN_PRICE:
            logger.debug(f"{symbol}: Price (${latest_price}) below minimum threshold (${MIN_PRICE}). Skipping.")
            return None
            
        # Make sure we have enough data
        if len(df) < MIN_LOOKBACK_DAYS:
            logger.debug(f"{symbol}: Insufficient data ({len(df)} days). Skipping.")
            return None
            
        price_series = df['Close'].iloc[-LOOKBACK_DAYS:] if len(df) >= LOOKBACK_DAYS else df['Close']
        
        # Determine how to divide the data based on available history
        available_days = len(price_series)
        year_length = min(252, available_days // max(1, min(3, available_days // 126)))
        num_years = min(3, available_days // year_length)
        
        # Calculate annual metrics for each of the years we can calculate
        annual_volatilities = []
        annual_drawdowns = []
        annual_betas = []
        annual_adrs = []
        
        for i in range(num_years):
            start_idx = -available_days + (i * year_length)
            end_idx = -available_days + ((i + 1) * year_length)
            if i == num_years - 1:  # For the most recent period, use all remaining data
                end_idx = None
            
            # Get price data for this period
            annual_slice = df.iloc[start_idx:end_idx]
            annual_price = annual_slice['Close']
            annual_returns = annual_price.pct_change().dropna()
            
            # Calculate annual volatility
            annual_vol = np.std(annual_returns) * np.sqrt(252)  # Annualized
            annual_volatilities.append(annual_vol)
            
            # Calculate annual max drawdown
            annual_running_max = annual_price.cummax()
            annual_drawdown = (annual_price - annual_running_max) / annual_running_max
            annual_max_drawdown = annual_drawdown.min()
            annual_drawdowns.append(annual_max_drawdown)
            
            # Calculate ADR (Average Daily Range)
            if 'High' in annual_slice.columns and 'Low' in annual_slice.columns:
                daily_ranges = (annual_slice['High'] - annual_slice['Low']) / annual_slice['Close'] * 100
                annual_adr = daily_ranges.mean()
                annual_adrs.append(annual_adr)
            else:
                # If High/Low not available, estimate from OHLC approximation
                if 'Open' in annual_slice.columns:
                    # Estimate range using open and close prices
                    daily_ranges = abs(annual_slice['Open'] - annual_slice['Close']) / annual_slice['Close'] * 100
                    # Adjust by typical ratio between true range and open-close
                    annual_adr = daily_ranges.mean() * 1.5  # Empirical adjustment factor
                else:
                    # Last resort - estimate from daily returns
                    annual_adr = np.std(annual_returns) * 100  # Convert to percentage
                annual_adrs.append(annual_adr)
            
            # Calculate annual beta
            annual_stock_returns = annual_returns.dropna()
            
            # Get corresponding benchmark returns for this period
            benchmark_period_ratio = len(annual_stock_returns) / 252
            benchmark_period_length = int(len(benchmark_returns) * benchmark_period_ratio)
            
            # Make sure we don't go out of bounds
            benchmark_start = max(0, len(benchmark_returns) - benchmark_period_length)
            annual_benchmark_returns = benchmark_returns.iloc[benchmark_start:]
            
            min_len = min(len(annual_stock_returns), len(annual_benchmark_returns))
            if min_len < 20:
                continue
                
            X = annual_benchmark_returns[-min_len:].values.reshape(-1, 1)
            y = annual_stock_returns[-min_len:].values
            
            try:
                model = LinearRegression().fit(X, y)
                annual_beta = model.coef_[0]
                annual_betas.append(annual_beta)
            except Exception as e:
                logger.warning(f"Beta calculation failed for {symbol} year {i+1}: {e}")
                annual_betas.append(1.0)  # Default beta
        
        # Calculate 3-year average metrics
        if (len(annual_volatilities) < 1 or len(annual_drawdowns) < 1 or 
            len(annual_betas) < 1 or len(annual_adrs) < 1):
            logger.debug(f"{symbol}: Not enough data to calculate at least one period's metrics. Skipping.")
            return None
            
        volatility = np.mean(annual_volatilities)
        max_drawdown = np.mean(annual_drawdowns)
        beta = np.mean(annual_betas)
        adr = np.mean(annual_adrs)
        
        # Classify individual metrics
        # Volatility classification
        if volatility < CONSERVATIVE_VOL_THRESHOLD:
            vol_score = 1  # Conservative
        elif volatility <= MODERATE_VOL_THRESHOLD:
            vol_score = 2  # Moderate
        else:
            vol_score = 3  # Aggressive
        
        # Beta classification
        if beta < CONSERVATIVE_BETA_THRESHOLD:
            beta_score = 1
        elif beta <= MODERATE_BETA_THRESHOLD:
            beta_score = 2
        else:
            beta_score = 3
        
        # Max drawdown classification
        if max_drawdown > CONSERVATIVE_MDD_THRESHOLD:
            mdd_score = 1
        elif max_drawdown > MODERATE_MDD_THRESHOLD:
            mdd_score = 2
        else:
            mdd_score = 3
            
        # ADR classification
        if adr < CONSERVATIVE_ADR_THRESHOLD:
            adr_score = 1
        elif adr <= MODERATE_ADR_THRESHOLD:
            adr_score = 2
        else:
            adr_score = 3
        
        # Calculate average score and determine risk classification
        avg_score = (vol_score + beta_score + mdd_score + adr_score) / 4
        
        if avg_score <= CONSERVATIVE_SCORE_THRESHOLD:
            risk_type = "Conservative"
        elif avg_score <= MODERATE_SCORE_THRESHOLD:
            risk_type = "Moderate"
        else:
            risk_type = "Aggressive"
            
        # Prepare and return results
        result = {
            'symbol': symbol,
            'is_etf': is_etf,
            'price': latest_price,
            'risk_type': risk_type,
            'average_score': avg_score,
            'volatility_3yr_avg': volatility,
            'beta_3yr_avg': beta,
            'max_drawdown_3yr_avg': max_drawdown,
            'adr_3yr_avg': adr,
            'volatility_score': vol_score,
            'beta_score': beta_score,
            'drawdown_score': mdd_score,
            'adr_score': adr_score,
            'annual_volatilities': annual_volatilities,
            'annual_drawdowns': annual_drawdowns,
            'annual_betas': annual_betas,
            'annual_adrs': annual_adrs
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None

def save_to_database(session, risk_data, batch_size=100):
    """Save risk profile data to the database"""
    
    if not risk_data:
        logger.warning("No risk data to save.")
        return 0, 0
        
    logger.info(f"Saving risk profile data for {len(risk_data)} symbols")
    
    classified_at = datetime.now()
    insertion_count = 0
    update_count = 0
    
    # Process in batches
    total_batches = (len(risk_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(risk_data))
        
        batch = risk_data[start_idx:end_idx]
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} with {len(batch)} records")
        
        for record in batch:
            symbol = record['symbol']
            
            # Check if profile already exists for this symbol
            existing_profile = session.query(EquityRiskProfile).filter_by(symbol=symbol).first()
            
            # Collect the year data (Year3 is most recent, Year1 is oldest)
            volatility_years = record['annual_volatilities'][::-1]  # Reverse to get newest first
            beta_years = record['annual_betas'][::-1]
            drawdown_years = record['annual_drawdowns'][::-1]
            adr_years = record['annual_adrs'][::-1]
            
            # Create dictionary with column values
            risk_profile_data = {
                'symbol': symbol,
                'is_etf': record['is_etf'],
                'price': record['price'],
                'risk_type': record['risk_type'],
                'average_score': round(record['average_score'], 2),
                'volatility_3yr_avg': round(record['volatility_3yr_avg'], 3),
                'beta_3yr_avg': round(record['beta_3yr_avg'], 2),
                'max_drawdown_3yr_avg': round(record['max_drawdown_3yr_avg'], 3),
                'adr_3yr_avg': round(record['adr_3yr_avg'], 2),
                'volatility_score': record['volatility_score'],
                'beta_score': record['beta_score'],
                'drawdown_score': record['drawdown_score'],
                'adr_score': record['adr_score'],
                'classified_at': classified_at
            }
            
            # Add yearly data if available
            for i, year_label in enumerate(['year3', 'year2', 'year1']):
                if i < len(volatility_years):
                    risk_profile_data[f'volatility_{year_label}'] = round(volatility_years[i], 3)
                if i < len(beta_years):
                    risk_profile_data[f'beta_{year_label}'] = round(beta_years[i], 2)
                if i < len(drawdown_years):
                    risk_profile_data[f'max_drawdown_{year_label}'] = round(drawdown_years[i], 3)
                if i < len(adr_years):
                    risk_profile_data[f'adr_{year_label}'] = round(adr_years[i], 2)
            
            if existing_profile:
                # Update existing record
                for key, value in risk_profile_data.items():
                    setattr(existing_profile, key, value)
                update_count += 1
            else:
                # Create new record
                profile = EquityRiskProfile(**risk_profile_data)
                session.add(profile)
                insertion_count += 1
        
        # Commit batch
        try:
            session.commit()
            logger.info(f"Batch {batch_idx + 1}: Committed {len(batch)} records")
        except Exception as e:
            logger.error(f"Error committing batch {batch_idx + 1}: {e}")
            session.rollback()
    
    logger.info(f"Database save completed: {insertion_count} insertions, {update_count} updates")
    return insertion_count, update_count

def calculate_risk_profiles(session, symbols_dict=None, num_workers=4, limit=None, include_stocks=True, include_etfs=False):
    """Calculate risk profiles for symbols and save to database"""
    
    start_time = time.time()
    
    # Get symbols if not provided
    if symbols_dict is None:
        symbols_dict = read_symbols(session, include_stocks=include_stocks, include_etfs=include_etfs)
    
    symbols = list(symbols_dict.keys())
    
    # Apply limit if specified
    if limit and limit < len(symbols):
        logger.info(f"Limiting to {limit} symbols (from {len(symbols)} available)")
        symbols = symbols[:limit]
        # Update the symbols_dict to match the limited symbols list
        symbols_dict = {symbol: symbols_dict[symbol] for symbol in symbols}
    else:
        logger.info(f"Processing all {len(symbols)} symbols")
    
    # Get price data including benchmark
    price_data = get_price_data(session, symbols, BENCHMARK_SYMBOL, LOOKBACK_DAYS)
    
    if not price_data or BENCHMARK_SYMBOL not in price_data:
        logger.error(f"Missing benchmark data for {BENCHMARK_SYMBOL}. Cannot calculate risk profiles.")
        return
        
    benchmark_df = price_data[BENCHMARK_SYMBOL]
    benchmark_returns = benchmark_df['Close'].pct_change().dropna()
    benchmark_latest_date = benchmark_df.index[-1]
    
    logger.info(f"Benchmark {BENCHMARK_SYMBOL} latest date: {benchmark_latest_date}")
    logger.info(f"Starting risk profile calculations with {num_workers} workers")
    
    # Lists to track statistics
    filtered_symbols = []
    risk_profiles = []
    no_data_count = 0
    
    # Process symbols using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_symbol = {}
        
        # Submit jobs
        for symbol in symbols:
            if symbol == BENCHMARK_SYMBOL:
                continue
                
            if symbol not in price_data:
                no_data_count += 1
                continue
                
            future = executor.submit(
                process_symbol, 
                symbol, 
                price_data[symbol], 
                benchmark_df,
                benchmark_returns,
                symbols_dict.get(symbol, False)  # Pass is_etf flag to process_symbol
            )
            future_to_symbol[future] = symbol
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            
            if i > 0 and i % 100 == 0:
                logger.info(f"Processed {i}/{len(future_to_symbol)} symbols")
                
            try:
                result = future.result()
                if result:
                    risk_profiles.append(result)
                else:
                    filtered_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                filtered_symbols.append(symbol)
    
    # Log results
    elapsed_time = time.time() - start_time
    logger.info(f"Risk profile calculation completed in {elapsed_time:.1f} seconds")
    logger.info(f"Processed {len(symbols)} symbols")
    logger.info(f"No data available: {no_data_count}")
    logger.info(f"Filtered/skipped: {len(filtered_symbols)}")
    logger.info(f"Valid risk profiles: {len(risk_profiles)}")
    
    # Count risk types
    risk_counts = {'Conservative': 0, 'Moderate': 0, 'Aggressive': 0}
    etf_counts = {'ETF': 0, 'Stock': 0}
    
    for profile in risk_profiles:
        risk_counts[profile['risk_type']] = risk_counts.get(profile['risk_type'], 0) + 1
        etf_counts['ETF' if profile['is_etf'] else 'Stock'] += 1
    
    total_valid = len(risk_profiles)
    if total_valid > 0:
        logger.info("Risk classification summary:")
        for risk_type, count in risk_counts.items():
            percentage = (count / total_valid) * 100
            logger.info(f"  {risk_type}: {count} ({percentage:.1f}%)")
            
        logger.info("Symbol type summary:")
        for symbol_type, count in etf_counts.items():
            percentage = (count / total_valid) * 100
            logger.info(f"  {symbol_type}: {count} ({percentage:.1f}%)")
    
    # Save results to database
    if risk_profiles:
        insert_count, update_count = save_to_database(session, risk_profiles)
        logger.info(f"Database operation completed: {insert_count} new records, {update_count} updates")
    
    return len(risk_profiles)

def main():
    parser = argparse.ArgumentParser(description='Calculate equity risk profiles')
    parser = add_environment_args(parser)
    
    parser.add_argument('--workers', type=int, default=1,
                      help='Number of worker threads')
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit processing to N symbols (for testing)')
    parser.add_argument('--symbols', type=str, default=None,
                      help='Comma-separated list of symbols to process (overrides database list)')
    parser.add_argument('--include-stocks', action='store_true', default=True,
                      help='Include stocks in the analysis')
    parser.add_argument('--include-etfs', action='store_true', default=True,
                      help='Include ETFs in the analysis')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup database connection
    try:
        engine = get_database_engine(args.env)
        Session = get_session_maker(args.env)
        session = Session()
        
        env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
        logger.info(f"Starting equity risk classification using {env_name} database with {args.workers} worker threads")
        
        # Get symbols if provided via command line
        symbols_dict = None
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(',')]
            logger.info(f"Using {len(symbols)} symbols provided via command line")
            
            # If manually specifying symbols, we need to query their ETF status
            query = """
            SELECT symbol, is_etf FROM symbol_fields 
            WHERE symbol IN :symbols
            """
            
            try:
                result = pd.read_sql(
                    text(query), 
                    session.bind,
                    params={'symbols': tuple(symbols)}
                )
                symbols_dict = dict(zip(result['symbol'], result['is_etf'] == 1))
                
                # Warn about missing symbols
                found_symbols = set(symbols_dict.keys())
                requested_symbols = set(symbols)
                missing_symbols = requested_symbols - found_symbols
                
                if missing_symbols:
                    logger.warning(f"Some requested symbols were not found in database: {', '.join(missing_symbols)}")
                
            except Exception as e:
                logger.error(f"Error fetching symbol types for manual symbols: {e}")
                # Default to treating all symbols as stocks
                symbols_dict = {symbol: False for symbol in symbols}
        
        # Calculate risk profiles
        total_profiles = calculate_risk_profiles(
            session=session, 
            symbols_dict=symbols_dict, 
            num_workers=args.workers, 
            limit=args.limit,
            include_stocks=args.include_stocks,
            include_etfs=args.include_etfs
        )
        
        if total_profiles is None or total_profiles == 0:
            logger.warning("No risk profiles were calculated.")
        else:
            logger.info(f"Successfully calculated {total_profiles} risk profiles")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
    finally:
        if 'session' in locals():
            session.close()
            logger.info("Database session closed")
            
    logger.info("Risk classification process completed")

if __name__ == "__main__":
    main()
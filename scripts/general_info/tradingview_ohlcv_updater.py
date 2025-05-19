import os
import sys
import argparse
import pandas as pd
import datetime
import time
import logging
import requests
import json
import threading
import concurrent.futures
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tvDatafeed import TvDatafeed, Interval
from tradingview_screener import Query, col 
from sqlalchemy import func

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import YfBar1d
from scripts.common_function import (
    get_database_engine, 
    add_environment_args, 
    get_session_maker
)

# --- Configuration ---
DEFAULT_BATCH_SIZE = 300
DEFAULT_WORKERS = 4
PREFERRED_EXCHANGES = ["NASDAQ", "NYSE", "AMEX"]  # Order of preference for auto-detection

# Define paths the same way as company_profile_yfinance.py
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "../logs"
INPUT_FILE = BASE_DIR / "../symbols/stock_symbols.txt"

# --- Logging Setup ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"tradingview_ohlcv_{datetime.datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Thread-local storage for database sessions
thread_local = threading.local()

def get_thread_session():
    """Get thread-local SQLAlchemy session"""
    if not hasattr(thread_local, "session"):
        thread_local.session = Session()
    return thread_local.session

def load_symbols():
    """Load symbols from input file"""
    try:
        with open(INPUT_FILE, "r") as f:
            content = f.read()
            symbols = [symbol.strip() for symbol in content.split(",") if symbol.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {INPUT_FILE}")
        return symbols
    except FileNotFoundError:
        logger.error(f"Input file not found: {INPUT_FILE}")
        return []

def detect_exchange_with_scanner_api(ticker_to_find):
    logger.info(f"Detecting exchange for '{ticker_to_find}' using TradingView Scanner API.")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        "Content-Type": "application/json",
        "Origin": "https://www.tradingview.com",
        "Referer": "https://www.tradingview.com/"
    }
    url = "https://scanner.tradingview.com/america/scan"
    payload = {
        "filter": [
            {"left": "name", "operation": "equal", "right": ticker_to_find}
        ],
        "options": {"lang": "en"},
        "markets": ["america"],
        "columns": ["name", "exchange"],
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
        "range": [0, 10]
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data_json = response.json()

        if data_json and data_json.get("data"):
            potential_matches = []
            for item in data_json["data"]:
                d = item.get("d")
                if d and len(d) == 2:
                    symbol_name, exchange_name = d[0], d[1]
                    if symbol_name == ticker_to_find and exchange_name:
                        potential_matches.append({"symbol": symbol_name, "exchange": exchange_name})
            
            if not potential_matches:
                logger.warning(f"Scanner API: No exact match for '{ticker_to_find}' in response.")
                return None, None

            for pref_ex in PREFERRED_EXCHANGES:
                for match in potential_matches:
                    if match["exchange"] == pref_ex:
                        logger.info(f"Scanner API: Found '{ticker_to_find}' on preferred exchange: {pref_ex}")
                        return match["symbol"], pref_ex
            
            first_valid_match = potential_matches[0]
            logger.info(f"Scanner API: Found '{ticker_to_find}' on exchange: {first_valid_match['exchange']} (first overall match).")
            return first_valid_match["symbol"], first_valid_match["exchange"]
        else:
            logger.warning(f"Scanner API: No data structure in response for '{ticker_to_find}'. Response: {data_json}")
            return None, None
    except requests.RequestException as e:
        logger.error(f"Scanner API request failed for '{ticker_to_find}': {e}")
        if e.response is not None:
            logger.error(f"Scanner API Error Status: {e.response.status_code}, Text: {e.response.text}")
        return None, None
    except Exception as e:
        logger.error(f"Error processing Scanner API response for '{ticker_to_find}': {e}")
        return None, None

def get_date_range(period_arg, at_date=None):
    """
    Determine the date range for data fetching based on the requested period.
    
    Args:
        period_arg (str): One of "last_day", "last_week", "last_month", "last_year", "at_date"
        at_date (str or datetime.date, optional): Specific date when using "at_date" period
        
    Returns:
        tuple: (target_date, n_bars) where target_date is the reference date and 
               n_bars is the number of bars to fetch
    """
    today = datetime.date.today()
    n_bars = 0
    
    # Handle at_date parameter first if provided
    if at_date:
        # Convert at_date string to datetime.date object if it's a string
        if isinstance(at_date, str):
            try:
                target_date = datetime.datetime.strptime(at_date, '%Y-%m-%d').date()
                n_bars = 1
                logger.info(f"Using specific date: {target_date} from at_date argument")
                return target_date, n_bars
            except ValueError as e:
                logger.error(f"Invalid date format for at_date: {at_date}. Error: {e}")
                return None, 0
        else:
            # If it's already a date object, use it directly
            target_date = at_date
            n_bars = 1
            logger.info(f"Using specific date: {target_date} from at_date argument")
            return target_date, n_bars
    
    # Handle regular period arguments
    if period_arg == "last_day":
        # Simply request 1 bar - let tvdatafeed handle trading day logic
        n_bars = 1
        return today, n_bars
    elif period_arg == "last_week":
        n_bars = 5  # Typically 5 trading days in a week
    elif period_arg == "last_month":
        n_bars = 22  # Approximately 22 trading days in a month
    elif period_arg == "last_year":
        n_bars = 252  # Approximately 252 trading days in a year
    else:
        logger.error(f"Invalid period argument: {period_arg}")
        return None, 0
        
    # For periods other than "last_day", return today as the reference date
    # The data provider will count back n_bars trading days from this date
    return today, n_bars

def fetch_ohlcv_with_tvdatafeed(tv, ticker_symbol, exchange, n_bars):
    logger.info(f"Fetching {n_bars} bars for '{ticker_symbol}' on {exchange} using tvdatafeed.")
    try:
        data = tv.get_hist(symbol=ticker_symbol, exchange=exchange, interval=Interval.in_daily, n_bars=n_bars)
        if data is not None and not data.empty:
            data = data.reset_index()
            data = data.rename(columns={"datetime": "timestamp", "symbol": "full_symbol"})
            data["symbol"] = ticker_symbol
            result_df = data[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
            logger.info(f"tvdatafeed: Successfully fetched {len(result_df)} bars for '{ticker_symbol}' on {exchange}.")
            return result_df
        else:
            logger.warning(f"tvdatafeed: No data returned for '{ticker_symbol}' on {exchange}.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"tvdatafeed: Error fetching data for '{ticker_symbol}' on {exchange}: {e}")
        return pd.DataFrame()

def fetch_ohlcv_with_tv_screener(ticker_symbol, exchange, target_date_for_record):
    logger.info(f"Fetching OHLCV for '{ticker_symbol}' on {exchange} for {target_date_for_record} using tradingview-screener.")
    try:
        handler = Query()
        handler.select("name", "open", "high", "low", "close", "volume")
        handler.where(col("name") == ticker_symbol, col("exchange") == exchange)
        screener_data_tuple = handler.get_scanner_data(verbose=False)
        
        if screener_data_tuple and screener_data_tuple[1] is not None and not screener_data_tuple[1].empty:
            data_df = screener_data_tuple[1]
            row = data_df.iloc[0]
            ohlcv_data = {
                "timestamp": pd.to_datetime(target_date_for_record),
                "symbol": ticker_symbol,
                "open": row.get("open"), "high": row.get("high"), "low": row.get("low"),
                "close": row.get("close"), "volume": row.get("volume")
            }
            if all(pd.notna(ohlcv_data[k]) for k in ["open", "high", "low", "close", "volume"]):
                logger.info(f"tradingview-screener: Successfully fetched OHLCV for '{ticker_symbol}' on {exchange}.")
                return pd.DataFrame([ohlcv_data])
            else:
                logger.warning(f"tradingview-screener: Incomplete OHLCV data for '{ticker_symbol}': {row.to_dict()}")
        else:
            logger.warning(f"tradingview-screener: No data returned for '{ticker_symbol}' on {exchange}.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"tradingview-screener: Error fetching data for '{ticker_symbol}': {e}")
        return pd.DataFrame()

def get_latest_timestamp_for_symbol(session, symbol):
    latest_entry = session.query(YfBar1d.timestamp).filter(YfBar1d.symbol == symbol).order_by(YfBar1d.timestamp.desc()).first()
    return latest_entry[0].date() if latest_entry else None

def update_database(session, data_df):
    if data_df.empty:
        logger.info("No data to update in the database.")
        print("No data to update in the database.")
        return
    
    insert_count = 0
    skip_count = 0
    symbol_stats = {}
    
    try:
        for _, row in data_df.iterrows():
            symbol = row["symbol"]
            date = row["timestamp"].date()
            
            # Track statistics by symbol
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {"inserted": 0, "skipped": 0}
            
            exists = session.query(YfBar1d).filter(
                YfBar1d.symbol == symbol, 
                func.date(YfBar1d.timestamp) == date
            ).first()
            
            if not exists:
                try:
                    bar = YfBar1d(
                        symbol=symbol, 
                        timestamp=row["timestamp"], 
                        open=float(row["open"]), 
                        high=float(row["high"]), 
                        low=float(row["low"]), 
                        close=float(row["close"]), 
                        volume=int(row["volume"])
                    )
                    session.add(bar)
                    insert_count += 1
                    symbol_stats[symbol]["inserted"] += 1
                    logger.info(f"Adding new record for '{symbol}' at {date}")
                    # Add print for CLI visibility
                    print(f"DB: Adding new record for '{symbol}' at {date}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Data conversion error for '{symbol}' at {date}: {e}")
                    print(f"ERROR: Data conversion error for '{symbol}' at {date}: {e}")
                    # Continue with other records
            else:
                skip_count += 1
                symbol_stats[symbol]["skipped"] += 1
                logger.info(f"Record for '{symbol}' at {date} already exists. Skipping.")
                # Reduce CLI output by not printing every skipped record
        
        if insert_count > 0: 
            session.commit()
            logger.info(f"Successfully inserted {insert_count} new rows, skipped {skip_count} existing rows.")
            # Make this stand out in CLI
            print(f"\n=== DB UPDATE: Inserted {insert_count} new rows, skipped {skip_count} existing rows ===\n")
            
            # Log detailed stats by symbol to CLI for important info
            for symbol, stats in symbol_stats.items():
                if stats["inserted"] > 0:
                    logger.info(f"Symbol {symbol}: Inserted {stats['inserted']} records, skipped {stats['skipped']} existing records")
                    print(f"Symbol {symbol}: Inserted {stats['inserted']} records, skipped {stats['skipped']} existing records")
            
        else: 
            logger.info(f"No new rows to insert. Skipped {skip_count} existing rows.")
            print(f"\nDB: No new rows to insert. Skipped {skip_count} existing rows.\n")
            
    except Exception as e: 
        session.rollback()
        error_msg = f"Error updating database: {e}"
        logger.error(error_msg)
        print(f"\nERROR: {error_msg}\n")

def process_ticker_batch(tv_instance, tickers_batch, period_arg, provider_arg, smart_update_flag, at_date=None):
    all_data_df = pd.DataFrame()
    session = get_thread_session()
    
    target_date_for_record, n_bars_to_fetch = get_date_range(period_arg, at_date)
    if n_bars_to_fetch <= 0:
        logger.warning(f"Invalid n_bars ({n_bars_to_fetch}) for period {period_arg}. Skipping batch.")
        return

    for ticker_from_file in tickers_batch:
        actual_symbol, exchange = detect_exchange_with_scanner_api(ticker_from_file)
        if not actual_symbol or not exchange:
            logger.warning(f"Could not find exchange for '{ticker_from_file}'. Skipping.")
            continue

        print(f"Processing {ticker_from_file} (Exchange: {exchange if exchange else 'Unknown'})")

        if smart_update_flag and not at_date:  # Skip smart update check if at_date is provided
            latest_db_date = get_latest_timestamp_for_symbol(session, actual_symbol)
            if latest_db_date:
                if period_arg == "last_day" and latest_db_date >= target_date_for_record:
                    logger.info(f"SmartSkip: Data for '{actual_symbol}' on {target_date_for_record} (or newer) already in DB.")
                    continue
                if period_arg != "last_day" and latest_db_date >= (datetime.date.today() - datetime.timedelta(days=n_bars_to_fetch - 1)):
                    logger.info(f"SmartSkip: Data for '{actual_symbol}' seems recent enough for period {period_arg}. Latest: {latest_db_date}. Skipping full fetch.")
                    continue
        elif smart_update_flag and at_date:
            # For at_date with smart_update, check if we already have data for that date
            exact_date_exists = session.query(YfBar1d).filter(
                YfBar1d.symbol == actual_symbol,
                func.date(YfBar1d.timestamp) == target_date_for_record
            ).first()
            if exact_date_exists:
                logger.info(f"SmartSkip: Data for '{actual_symbol}' on {target_date_for_record} already exists in DB.")
                continue

        ticker_df = pd.DataFrame()
        # For at_date, we always just want 1 day of data
        if at_date and provider_arg == "tv_screener":
            ticker_df = fetch_ohlcv_with_tv_screener(actual_symbol, exchange, target_date_for_record)
        elif (period_arg == "last_day" or at_date) and provider_arg == "tv_screener":
            ticker_df = fetch_ohlcv_with_tv_screener(actual_symbol, exchange, target_date_for_record)
        else:
            if tv_instance is None and provider_arg == "tvdatafeed":
                logger.error("tv_instance is None for tvdatafeed provider. This is an issue.")
                tv_instance = TvDatafeed()
            elif tv_instance is None and period_arg != "last_day" and provider_arg == "tv_screener": 
                logger.error("tv_instance is None for tvdatafeed (fallback from tv_screener on non-last_day). This is an issue.")
                tv_instance = TvDatafeed()
            
            if tv_instance:
                ticker_df = fetch_ohlcv_with_tvdatafeed(tv_instance, actual_symbol, exchange, n_bars_to_fetch)
            else:
                logger.error(f"Skipping {actual_symbol} for provider {provider_arg} due to missing tv_instance.")
        
        if not ticker_df.empty:
            # If we're using at_date, filter for just that date
            if at_date and not ticker_df.empty and 'timestamp' in ticker_df.columns:
                target_date = pd.to_datetime(target_date_for_record).normalize()
                ticker_df = ticker_df[ticker_df['timestamp'].dt.normalize() == target_date]
                if ticker_df.empty:
                    logger.warning(f"No data found for '{actual_symbol}' on {target_date_for_record} after filtering.")
            
            all_data_df = pd.concat([all_data_df, ticker_df], ignore_index=True)
        time.sleep(1)

    if not all_data_df.empty:
        # Log data preview before database update
        logger.info(f"Batch data summary before DB update: {len(all_data_df)} rows, {all_data_df['symbol'].nunique()} unique symbols")
        logger.info(f"Symbols to update: {', '.join(all_data_df['symbol'].unique())}")
        
        # Show date range in data
        if 'timestamp' in all_data_df.columns and not all_data_df.empty:
            min_date = all_data_df['timestamp'].min().date()
            max_date = all_data_df['timestamp'].max().date()
            logger.info(f"Date range in batch: {min_date} to {max_date}")
        
        update_database(session, all_data_df)
        print(f"\n--- Batch processing complete: {len(tickers_batch)} symbols processed ---\n")
    else:
        logger.warning(f"No data collected for this batch of {len(tickers_batch)} tickers")
        print(f"\n--- Batch complete: No data collected for {len(tickers_batch)} tickers ---\n")

def main():
    start_time = datetime.datetime.now()
    print(f"Started execution at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    current_timezone = time.tzname
    logger.info(f"Script running in timezone: {current_timezone}")
    print(f"Script running in timezone: {current_timezone}")

    parser = argparse.ArgumentParser(description="Download historical OHLCV data and update database.")
    parser = add_environment_args(parser)
    parser.add_argument("period", choices=["last_day", "last_week", "last_month", "last_year", "at_date"],
                        help="Target period to fetch data for. Use 'at_date' with --date option for specific date.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of tickers per batch.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of worker threads.")
    parser.add_argument("--smart_update", action="store_true", help="Enable smart update (mainly for last_day).")
    parser.add_argument("--provider", choices=["tvdatafeed", "tv_screener"], default="tvdatafeed", 
                        help="Data provider for OHLCV. 'tv_screener' is only applicable for 'last_day' period or at_date.")
    parser.add_argument("--date", type=str, help="Specific date (YYYY-MM-DD) to fetch data for. Used with 'at_date' period.")
    args = parser.parse_args()

    # Validate at_date arguments
    if args.period == "at_date" and not args.date:
        logger.error("Error: When using period 'at_date', you must specify a date with --date. Format: YYYY-MM-DD")
        print("Error: When using period 'at_date', you must specify a date with --date. Format: YYYY-MM-DD")
        return

    if args.date and args.period != "at_date":
        logger.warning(f"Warning: --date argument is specified but period is not 'at_date'. Ignoring --date argument.")
        print(f"Warning: --date argument is specified but period is not 'at_date'. Ignoring --date argument.")
        args.date = None

    if args.provider == "tv_screener" and args.period not in ["last_day", "at_date"]:
        logger.error("--provider tv_screener can only be used with --period last_day or at_date. Exiting.")
        print("Error: --provider tv_screener can only be used with --period last_day or at_date. Exiting.")
        return

    # Setup database connection based on environment
    global engine, Session
    engine = get_database_engine(args.env)
    Session = get_session_maker(args.env)
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    
    # Log the execution configuration
    if args.period == "at_date":
        logger.info(f"Starting OHLCV update for specific date: {args.date}, provider: {args.provider}, smart_update: {args.smart_update} in {env_name} environment")
        print(f"Starting OHLCV update for specific date: {args.date}, provider: {args.provider}, smart_update: {args.smart_update} in {env_name} environment")
    else:
        logger.info(f"Starting OHLCV update for period: {args.period}, provider: {args.provider}, smart_update: {args.smart_update} in {env_name} environment")
        print(f"Starting OHLCV update for period: {args.period}, provider: {args.provider}, smart_update: {args.smart_update} in {env_name} environment")
    
    tickers_from_file = load_symbols()
    if not tickers_from_file:
        logger.info("No tickers to process. Exiting.")
        return

    ticker_batches = [tickers_from_file[i:i + args.batch_size] for i in range(0, len(tickers_from_file), args.batch_size)]
    total_batches = len(ticker_batches)
    
    print(f"Found {len(tickers_from_file)} symbols to process in {total_batches} batches")
    completed_batches = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i, batch in enumerate(ticker_batches):
            logger.info(f"Submitting batch {i + 1}/{total_batches} for processing: {batch}")
            print(f"Submitting batch {i + 1}/{total_batches} ({(i+1)/total_batches*100:.1f}%) for processing: {batch}")
            
            tv_instance_for_batch = None
            if args.provider == "tvdatafeed" or (args.period not in ["last_day", "at_date"] and args.provider == "tv_screener"):
                 tv_instance_for_batch = TvDatafeed()
            
            future = executor.submit(
                process_ticker_batch, 
                tv_instance_for_batch, 
                batch, 
                args.period, 
                args.provider, 
                args.smart_update,
                args.date if args.period == "at_date" else None
            )
            futures.append(future)
            
        for future in concurrent.futures.as_completed(futures):
            completed_batches += 1
            print(f"Completed batch {completed_batches}/{total_batches} ({completed_batches/total_batches*100:.1f}%)")

    logger.info("All ticker batches completed. OHLCV data update process finished.")
    
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    
    print(f"\n=== OHLCV Update Summary ===")
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {execution_time}")
    if args.period == "at_date":
        print(f"Processed {len(tickers_from_file)} symbols for date {args.date} in {total_batches} batches")
        logger.info(f"Execution completed in {execution_time}. Processed {len(tickers_from_file)} symbols for date {args.date}.")
    else:
        print(f"Processed {len(tickers_from_file)} symbols in {total_batches} batches")
        logger.info(f"Execution completed in {execution_time}. Processed {len(tickers_from_file)} symbols.")

if __name__ == "__main__":
    import threading
    main()



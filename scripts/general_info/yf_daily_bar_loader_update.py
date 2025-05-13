import os
import sys
import argparse
import logging
import math
import threading
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import func, distinct

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import YfBar1d
from scripts.common_function import (
    get_database_engine, 
    add_environment_args, 
    get_session_maker,
    YahooFinanceLoader,  
    NorgateDataLoader   
)

# Constants and Configuration
NUM_WORKERS = 2  # Reduced from 4 to 2
BATCH_SIZE = 3   # Reduced to 3 symbols per batch
MIN_DELAY = 1.0  # Increased minimum delay
MAX_DELAY = 3.0  # Increased maximum delay
DEFAULT_START_DATE = "2025-01-01" 
DEFAULT_SOURCE = "yahoo"
DEFAULT_SAFETY_DAYS = 3  # Days to look back when using latest timestamp (to catch any missed days)

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"daily_bar_loader_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Thread-local storage for database sessions
thread_local = threading.local()

class Stats:
    def __init__(self, total_symbols):
        self.processed = 0
        self.successful = 0
        self.bars_added = 0
        self.bars_updated = 0
        self.failed = 0
        self.skipped_bars = 0
        self.lock = threading.Lock()
        self.total_symbols = total_symbols
        self.start_time = datetime.now()
    
    def increment(self, attribute, value=1):
        with self.lock:
            setattr(self, attribute, getattr(self, attribute) + value)
            if attribute == 'processed' and self.processed % 10 == 0:
                elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                symbols_per_minute = self.processed / elapsed if elapsed > 0 else 0
                remaining = (self.total_symbols - self.processed) / symbols_per_minute if symbols_per_minute > 0 else 0
                
                logger.info(f"Rate: {symbols_per_minute:.1f} symbols/min, Est. remaining: {remaining:.1f} minutes")
                logger.info(f"Progress: {self.processed}/{self.total_symbols} ({(self.processed/self.total_symbols)*100:.1f}%)")
                logger.info(f"Bars added: {self.bars_added}, Bars updated: {self.bars_updated}, Failed: {self.failed}, Skipped: {self.skipped_bars}")


def get_thread_session():
    """Get thread-local SQLAlchemy session"""
    if not hasattr(thread_local, "session"):
        thread_local.session = Session()
    return thread_local.session


def normalize_timestamp(timestamp):
    """Convert timezone-aware timestamp to naive datetime"""
    if timestamp is None:
        return None
    
    # If timestamp has timezone info, convert to UTC and remove tz info
    if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
        # Convert to UTC time using the properly imported timezone
        return timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    return timestamp


def get_latest_timestamp(symbol=None, session=None):
    """
    Get the latest timestamp in the database for a symbol or all symbols
    
    Args:
        symbol: Symbol to check (if None, check all symbols)
        session: SQLAlchemy session to use (if None, create a new one)
    
    Returns:
        datetime or None: Latest timestamp in the database
    """
    if session is None:
        session = get_thread_session()
    
    try:
        query = session.query(func.max(YfBar1d.timestamp))
        if symbol:
            query = query.filter(YfBar1d.symbol == symbol)
        
        latest = query.scalar()
        return latest
    except Exception as e:
        logger.error(f"Error getting latest timestamp: {e}")
        return None


def get_date_range(period=None, start_date=None, symbol=None, smart_update=True):
    """
    Get date range based on period, start date, or latest timestamp in database
    
    Args:
        period: Predefined time period (last_day, last_week, etc.)
        start_date: Explicit start date (overrides period)
        symbol: Symbol to check for latest timestamp
        smart_update: Whether to use the latest timestamp in database as reference
    
    Returns:
        tuple: (start_date, end_date)
    """
    end_date = datetime.now()
    
    # Case 1: Explicit start date provided
    if start_date:
        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid start date format: {start_date}. Using default.")
                start_date = datetime.strptime(DEFAULT_START_DATE, '%Y-%m-%d')
        return start_date, end_date
    
    # Case 2: Smart update enabled - check database for latest timestamp
    if smart_update:
        latest_timestamp = get_latest_timestamp(symbol)
        if latest_timestamp:
            # Use latest timestamp minus safety days to catch any missed days
            smart_start_date = latest_timestamp.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=DEFAULT_SAFETY_DAYS)
            logger.info(f"Using latest timestamp from database: {latest_timestamp.date()} (minus {DEFAULT_SAFETY_DAYS} safety days)")
            return smart_start_date, end_date
    
    # Case 3: Use predefined period
    if period == 'last_day':
        start_date = end_date - timedelta(days=3)  # 3 days to account for weekends
    elif period == 'last_week':
        start_date = end_date - timedelta(days=7)
    elif period == 'last_month':
        start_date = end_date - timedelta(days=30)
    elif period == 'last_quarter':
        start_date = end_date - timedelta(days=90)
    elif period == 'ytd':
        start_date = datetime(end_date.year, 1, 1)
    elif period == 'last_year':
        start_date = end_date - timedelta(days=365)
    elif period == 'full':
        start_date = datetime(2000, 1, 1)  # Very early date for full history
    else:
        logger.warning(f"Invalid period '{period}', using default of 1 year")
        start_date = end_date - timedelta(days=365)
    
    return start_date, end_date


def get_data_loader(source, batch_size=BATCH_SIZE):
    """Get the appropriate data loader based on source"""
    if source.lower() == 'norgate':
        return NorgateDataLoader()
    elif source.lower() == 'yahoo':
        # Updated with improved batch size and worker count
        return YahooFinanceLoader(batch_size=batch_size, worker_count=1)
    else:
        logger.warning(f"Unknown source '{source}', using Yahoo Finance")
        return YahooFinanceLoader(batch_size=batch_size, worker_count=1)


def check_symbol_exists(symbol, session):
    """Check if a symbol exists in the symbol_fields table"""
    from models.market_data import SymbolFields
    return session.query(SymbolFields).filter(SymbolFields.symbol == symbol).first() is not None


def add_symbol_to_database(symbol, session):
    """Add a missing symbol to the symbol_fields table with minimal info"""
    from models.market_data import SymbolFields
    try:
        new_symbol = SymbolFields(
            symbol=symbol,
            company_name=f"Auto-added: {symbol}",
            updated_at=datetime.now()
        )
        session.add(new_symbol)
        session.commit()
        logger.info(f"Added missing symbol {symbol} to symbol_fields table")
        return True
    except Exception as e:
        logger.error(f"Failed to add symbol {symbol} to database: {e}")
        session.rollback()
        return False


# ENHANCED RATE-LIMIT HANDLING: Improved download function with better retry logic
def download_with_enhanced_retry(symbol, start_date, end_date, max_retries=5, initial_delay=5):
    """Download stock data with enhanced exponential backoff retry logic"""
    delay = initial_delay
    attempt = 0
    rate_limited = False
    
    while attempt < max_retries:
        attempt += 1
        try:
            # If we've been rate limited before, wait longer
            if rate_limited:
                sleep_time = delay + random.uniform(1.0, 3.0)  # Add randomness to avoid synchronized requests
                logger.info(f"Rate limit hit for {symbol}. Waiting {sleep_time:.2f}s before retry {attempt}...")
                time.sleep(sleep_time)
                rate_limited = False
            
            logger.info(f"Downloading {symbol} (attempt {attempt})...")
            
            # Let YFinance handle its own session
            ticker = yf.Ticker(symbol)
            
            # Check if ticker exists with a minimal query
            check_df = ticker.history(period='1d')
            if check_df.empty:
                logger.info(f"{symbol} has no data available")
                return None
                
            # Use explicit start and end dates - avoid using period parameter
            history_df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if history_df.empty:
                logger.info(f"{symbol} returned empty dataframe")
                return None
                
            # Check for splits and dividends safely
            try:
                # Get splits and dividends directly from ticker
                splits = ticker.splits
                dividends = ticker.dividends
                
                has_recent_event = False
                
                # Handle timezone-aware DatetimeIndex properly
                if len(splits) > 0 and isinstance(splits.index, pd.DatetimeIndex):
                    # Convert dates to UTC for safe comparison
                    start_ts = pd.Timestamp(start_date).tz_localize('UTC')
                    end_ts = pd.Timestamp(end_date).tz_localize('UTC')
                    
                    # Convert splits index to UTC if it has timezone
                    splits_idx = splits.index
                    if splits_idx.tz is not None:
                        # For timezone-aware index, convert the comparison timestamps to match
                        start_ts = start_ts.tz_convert(splits_idx.tz)
                        end_ts = end_ts.tz_convert(splits_idx.tz)
                    
                    # Now do the comparison with appropriate timezone handling
                    recent_splits = splits[(splits_idx >= start_ts) & (splits_idx <= end_ts)]
                    if not recent_splits.empty:
                        has_recent_event = True
                        logger.info(f"{symbol} has recent splits")
                
                # Same for dividends with timezone handling
                if len(dividends) > 0 and isinstance(dividends.index, pd.DatetimeIndex):
                    # Convert dates to UTC for safe comparison
                    start_ts = pd.Timestamp(start_date).tz_localize('UTC')
                    end_ts = pd.Timestamp(end_date).tz_localize('UTC')
                    
                    # Convert dividends index to UTC if it has timezone
                    div_idx = dividends.index
                    if div_idx.tz is not None:
                        # For timezone-aware index, convert the comparison timestamps to match
                        start_ts = start_ts.tz_convert(div_idx.tz)
                        end_ts = end_ts.tz_convert(div_idx.tz)
                    
                    # Now do the comparison with appropriate timezone handling
                    recent_dividends = dividends[(div_idx >= start_ts) & (div_idx <= end_ts)]
                    if not recent_dividends.empty:
                        has_recent_event = True
                        logger.info(f"{symbol} has recent dividends")
                
                # If split or dividend occurred, fetch limited history instead of full history
                if has_recent_event:
                    logger.info(f"Split or dividend happened to {symbol} so fetching more history...")
                    # Use 90 days instead of all history to reduce chance of rate limiting
                    longer_start_date = end_date - timedelta(days=90)
                    
                    # Let YFinance create a new instance
                    event_ticker = yf.Ticker(symbol)
                    
                    # Use explicit date range rather than period='max'
                    all_history_df = event_ticker.history(start=longer_start_date, end=end_date, auto_adjust=True)
                    if not all_history_df.empty:
                        history_df = all_history_df
            except Exception as e:
                logger.warning(f"{symbol}: Error checking splits/dividends: {e}")
            
            # Add jitter to avoid synchronized requests hitting rate limits
            time.sleep(random.uniform(1.0, 2.0))  # Increased delay between requests
            return history_df
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle period max error
            if "period 'max' is invalid" in error_msg:
                logger.warning(f"{symbol}: {e} - Using explicit date range instead")
                try:
                    # Try with explicit date range but with a shorter time span
                    shorter_start_date = end_date - timedelta(days=90)  # Just get 90 days
                    ticker = yf.Ticker(symbol)
                    history_df = ticker.history(start=shorter_start_date, end=end_date, auto_adjust=True)
                    return history_df
                except Exception as retry_error:
                    logger.error(f"{symbol}: Error on retry with explicit dates: {retry_error}")
            
            # Handle rate limiting specifically
            elif "rate limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                rate_limited = True
                logger.warning(f"Rate limit hit for {symbol}. Will retry with backoff.")
                
                # Exponential backoff with jitter
                delay = initial_delay * (2 ** (attempt - 1)) + random.uniform(1, 5)
                
                # Cap the delay but with higher maximum than before (120 seconds)
                delay = min(delay, 120)
                
                # If this is the final retry, save progress and exit
                if attempt >= max_retries - 1:
                    logger.error(f"Rate limit persists for {symbol} after multiple retries. Saving progress and exiting.")
                    save_progress_file()
                    # Wait 10 minutes and then try again
                    time.sleep(600)  # 10 minutes
            else:
                logger.error(f"Error downloading {symbol}: {e}")
                # For non-rate-limit errors, use a moderate delay
                time.sleep(random.uniform(2, 5))
    
    logger.error(f"Failed to download {symbol} after {max_retries} attempts")
    return None


# ENHANCED RATE-LIMIT HANDLING: Save progress to file
def save_progress_file(completed=None, failed=None):
    """Save progress to files for potential resume later"""
    # Use stats to determine what's been processed
    if completed is None:
        completed = []
        failed = []
        session = get_thread_session()
        
        # Get all successfully processed symbols
        try:
            # Find unique symbols in the database from the last day
            yesterday = datetime.now() - timedelta(days=1)
            recent_symbols = session.query(distinct(YfBar1d.symbol)).filter(
                YfBar1d.timestamp >= yesterday
            ).all()
            completed = [row[0] for row in recent_symbols]
            logger.info(f"Identified {len(completed)} recently processed symbols")
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    # Save completed symbols
    progress_file = LOG_DIR / "yf_loader_progress.txt"
    try:
        with open(progress_file, "w") as f:
            f.write(",".join(completed))
            if failed:
                f.write("\nFAILED:")
                f.write(",".join(failed))
        logger.info(f"Progress saved to {progress_file}: {len(completed)} completed, {len(failed) if failed else 0} failed")
    except Exception as e:
        logger.error(f"Error writing progress file: {e}")


# ENHANCED RATE-LIMIT HANDLING: Load saved progress
def load_progress_file():
    """Load saved progress from file if exists"""
    progress_file = LOG_DIR / "yf_loader_progress.txt"
    completed = []
    failed = []
    
    if progress_file.exists():
        try:
            content = progress_file.read_text().strip()
            parts = content.split("\nFAILED:")
            
            if parts[0]:
                completed = [s for s in parts[0].split(",") if s.strip()]
                
            if len(parts) > 1 and parts[1]:
                failed = [s for s in parts[1].split(",") if s.strip()]
                
            logger.info(f"Loaded progress: {len(completed)} completed, {len(failed)} failed symbols")
            return completed, failed
        except Exception as e:
            logger.error(f"Error reading progress file: {e}")
    
    return completed, failed


# ENHANCED RATE-LIMIT HANDLING: Chunk symbols to avoid daily limits
def chunk_symbols(symbols, chunk_size=100):
    """Split symbols into chunks to allow for daily rate limit reset"""
    return [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]


def process_symbol(symbol, start_date, end_date, data_source, force_update=False, auto_add_symbols=True, batch_size=BATCH_SIZE):
    """Process a single symbol's historical data"""
    session = get_thread_session()
    added = 0
    updated = 0
    skipped = 0
    
    try:
        # First check if the symbol exists in the symbol_fields table
        if not check_symbol_exists(symbol, session):
            if auto_add_symbols:
                logger.warning(f"Symbol {symbol} not found in database. Attempting to add it.")
                if not add_symbol_to_database(symbol, session):
                    logger.error(f"Skipping {symbol} as it could not be added to the database")
                    stats.increment('failed')
                    return 0, 0
            else:
                logger.warning(f"Skipping {symbol} as it doesn't exist in symbol_fields table")
                stats.increment('failed')
                return 0, 0
        
        # ENHANCED RATE-LIMIT HANDLING: Use improved download function
        if data_source.lower() == 'yahoo':
            # Use our enhanced rate limit handling method
            df = download_with_enhanced_retry(symbol, start_date, end_date)
        else:
            # Use the original loader for non-yahoo sources
            loader = get_data_loader(data_source, batch_size)
            df = loader.load_symbol_data(symbol, start_date, end_date)
        
        if df is None or df.empty:
            logger.warning(f"No data found for {symbol}")
            stats.increment('failed')
            return 0, 0
        
        # Group by date - extract date only from timestamp (without time)
        df['date_only'] = df.index.date
        # Get the latest record for each date
        latest_records = df.groupby('date_only').last()
        logger.info(f"Found {len(df)} records, using {len(latest_records)} unique days for {symbol}")
        
        # Process each bar (using latest record for each day)
        for index, row in latest_records.iterrows():
            try:
                # Normalize column names if needed
                open_price = row.get('Open', row.get('open', None))
                high_price = row.get('High', row.get('high', None))
                low_price = row.get('Low', row.get('low', None))
                close_price = row.get('Close', row.get('close', None))
                volume = row.get('Volume', row.get('volume', None))
                
                # Check for NaN or None values and skip the record if any are found
                if (pd.isna(open_price) or pd.isna(high_price) or pd.isna(low_price) or 
                    pd.isna(close_price) or pd.isna(volume) or
                    open_price is None or high_price is None or 
                    low_price is None or close_price is None or volume is None):
                    logger.debug(f"Skipping bar with NaN values for {symbol} at {index}")
                    skipped += 1
                    continue
                
                # Create naive datetime at midnight for the date (removes time component)
                timestamp = datetime.combine(index, datetime.min.time())
                
                # Check for existing record
                existing_record = None
                if not force_update:
                    # Check for any record on the same DATE (ignoring time)
                    existing_records = session.query(YfBar1d).filter(
                        YfBar1d.symbol == symbol,
                        func.date(YfBar1d.timestamp) == timestamp.date()
                    ).all()
                    
                    if existing_records:
                        existing_record = existing_records[0]
                
                # Ensure float conversions are safe
                try:
                    open_val = float(open_price)
                    high_val = float(high_price)
                    low_val = float(low_price)
                    close_val = float(close_price)
                    vol_val = int(float(volume))
                    
                    # Additional safety check
                    if (np.isnan(open_val) or np.isnan(high_val) or 
                        np.isnan(low_val) or np.isnan(close_val)):
                        logger.debug(f"Skipping bar with NaN after conversion for {symbol} at {index}")
                        skipped += 1
                        continue
                except (ValueError, TypeError):
                    logger.debug(f"Skipping bar with invalid values for {symbol} at {index}")
                    skipped += 1
                    continue
                
                if existing_record:
                    # Update the existing record
                    existing_record.open = open_val
                    existing_record.high = high_val
                    existing_record.low = low_val
                    existing_record.close = close_val
                    existing_record.volume = vol_val
                    updated += 1
                else:
                    # Insert new record
                    try:
                        bar_data = YfBar1d(
                            symbol=symbol,
                            timestamp=timestamp,  # Using midnight timestamp (00:00:00)
                            open=open_val,
                            high=high_val,
                            low=low_val,
                            close=close_val,
                            volume=vol_val
                        )
                        session.add(bar_data)
                        added += 1
                    except Exception as e:
                        logger.error(f"Error adding bar for {symbol} at {timestamp}: {e}")
                        skipped += 1
                        continue
                
                # Commit in batches to avoid large transactions
                if (added + updated) % 100 == 0:
                    try:
                        session.commit()
                    except Exception as e:
                        logger.error(f"Error committing batch for {symbol}: {e}")
                        session.rollback()
                    
            except Exception as e:
                logger.error(f"Error processing bar for {symbol} at {index}: {e}")
                session.rollback()  # Important: Roll back on error
                continue
                
        # Additional code to clean up any old duplicate entries
        try:
            # First get all dates that have multiple entries
            duplicate_dates = session.query(
                func.date(YfBar1d.timestamp).label('date'),
                func.count('*').label('count')
            ).filter(
                YfBar1d.symbol == symbol
            ).group_by(
                func.date(YfBar1d.timestamp)
            ).having(
                func.count('*') > 1
            ).all()
            
            # For each date with duplicates, keep only the one with latest timestamp
            for date_record in duplicate_dates:
                duplicate_date = date_record.date
                
                # Get all records for this date
                records = session.query(YfBar1d).filter(
                    YfBar1d.symbol == symbol,
                    func.date(YfBar1d.timestamp) == duplicate_date
                ).order_by(
                    YfBar1d.timestamp.desc()
                ).all()
                
                # Keep the first one (latest), delete the rest
                for record in records[1:]:
                    session.delete(record)
                
                logger.info(f"Cleaned up duplicates for {symbol} on {duplicate_date}")
            
            if duplicate_dates:
                session.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning up duplicates for {symbol}: {e}")
            session.rollback()

        # Final commit for any remaining records
        try:
            session.commit()
            stats.increment('bars_added', added)
            stats.increment('bars_updated', updated)
            stats.increment('skipped_bars', skipped)
            stats.increment('successful')
            logger.info(f"Processed {symbol}: {added} bars added, {updated} bars updated, {skipped} bars skipped")
            return added, updated
        except Exception as e:
            logger.error(f"Error in final commit for {symbol}: {e}")
            session.rollback()
            stats.increment('failed')
            return 0, 0
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        try:
            session.rollback()
        except:
            pass  # Ignore if rollback fails
        stats.increment('failed')
        return 0, 0
    finally:
        stats.increment('processed')


# ENHANCED RATE-LIMIT HANDLING: Improved batch processing with better pauses
def process_symbol_batch(symbols, start_date, end_date, data_source, force_update, smart_update, auto_add_symbols=False, batch_size=BATCH_SIZE):
    """Process a batch of symbols with improved rate limit handling"""
    thread_name = threading.current_thread().name
    logger.info(f"[{thread_name}] Starting batch processing of {len(symbols)} symbols")
    
    completed_symbols = []
    failed_symbols = []
    
    for symbol in symbols:
        try:
            # For smart updates, get symbol-specific start_date
            if smart_update:
                symbol_start_date, symbol_end_date = get_date_range(
                    start_date=None, 
                    symbol=symbol,
                    smart_update=True
                )
            else:
                symbol_start_date, symbol_end_date = start_date, end_date
            
            bars_added, bars_updated = process_symbol(
                symbol, symbol_start_date, symbol_end_date, 
                data_source, force_update, auto_add_symbols, batch_size
            )
            
            if bars_added > 0 or bars_updated > 0:
                completed_symbols.append(symbol)
            else:
                failed_symbols.append(symbol)
            
            # ENHANCED RATE-LIMIT HANDLING: Add a significant delay to avoid hammering the API
            time.sleep(random.uniform(MIN_DELAY*2, MAX_DELAY*2))
            
            # ENHANCED RATE-LIMIT HANDLING: Periodically save progress
            if len(completed_symbols) % 10 == 0:
                save_progress_file(completed_symbols, failed_symbols)
                
        except Exception as e:
            logger.error(f"Unexpected error processing {symbol}: {str(e)}")
            failed_symbols.append(symbol)
            
            # If error message contains rate limit indications, pause longer
            if "rate limit" in str(e).lower() or "429" in str(e).lower() or "too many requests" in str(e).lower():
                wait_time = random.uniform(60, 120)  # 1-2 minutes
                logger.warning(f"Rate limit hit! Pausing thread for {wait_time:.1f} seconds...")
                time.sleep(wait_time)
    
    logger.info(f"[{thread_name}] Completed batch processing: {len(completed_symbols)} succeeded, {len(failed_symbols)} failed")
    return completed_symbols, failed_symbols


def get_symbols(input_file=None):
    """Get symbols from input file or use defaults"""
    # Default symbol file location
    if not input_file:
        input_file = script_dir.parent / "symbols/stock_symbols.txt"
    
    logger.info(f"Reading symbols from {input_file}")
    try:
        with open(input_file, 'r') as f:
            content = f.read().strip()
            symbols = [symbol.strip() for symbol in content.split(',') if symbol.strip()]
        
        if not symbols:
            raise ValueError("No symbols found in input file")
            
        logger.info(f"Found {len(symbols)} symbols in input file")
        return symbols
    except Exception as e:
        logger.error(f"Error reading symbol file: {e}")
        # Return some default symbols if file can't be read
        default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
        logger.warning(f"Using default symbols: {', '.join(default_symbols)}")
        return default_symbols


def analyze_database_stats():
    """Analyze and log stats about daily bar data in the database"""
    session = Session()
    try:
        # Get basic counts
        total_bars = session.query(YfBar1d).count()
        total_symbols = session.query(YfBar1d.symbol).distinct().count()
        
        logger.info("\nDaily Bar Database Statistics:")
        logger.info(f"Total bars stored: {total_bars:,}")
        logger.info(f"Total symbols: {total_symbols}")
        
        if total_symbols > 0:
            bars_per_symbol = total_bars / total_symbols
            logger.info(f"Average bars per symbol: {bars_per_symbol:.1f}")
        
        # Get date range
        oldest_bar = session.query(YfBar1d).order_by(YfBar1d.timestamp.asc()).first()
        newest_bar = session.query(YfBar1d).order_by(YfBar1d.timestamp.desc()).first()
        
        if oldest_bar and newest_bar:
            days_span = (newest_bar.timestamp - oldest_bar.timestamp).days
            logger.info(f"Date range: {oldest_bar.timestamp.date()} to {newest_bar.timestamp.date()} ({days_span} days)")
        
        # Get symbols with most and least data
        symbol_counts = session.query(
            YfBar1d.symbol, 
            func.count(YfBar1d.symbol).label('count')
        ).group_by(YfBar1d.symbol).order_by(func.count(YfBar1d.symbol).desc()).all()
        
        if symbol_counts:
            most_data = symbol_counts[0]
            least_data = symbol_counts[-1]
            logger.info(f"Symbol with most bars: {most_data[0]} ({most_data[1]} bars)")
            logger.info(f"Symbol with least bars: {least_data[0]} ({least_data[1]} bars)")
        
        # Get recent activity
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)
        
        bars_today = session.query(YfBar1d).filter(YfBar1d.timestamp >= today).count()
        bars_yesterday = session.query(YfBar1d).filter(
            YfBar1d.timestamp >= yesterday,
            YfBar1d.timestamp < today
        ).count()
        
        logger.info(f"Bars added today: {bars_today}")
        logger.info(f"Bars added yesterday: {bars_yesterday}")
        
        # Get data holes (days with missing data)
        if newest_bar and oldest_bar:
            # This query is for more advanced monitoring - check if there are any missing days
            # in the most active symbols (ones with the most data points)
            logger.info("\nChecking for data gaps in top 10 symbols...")
            
            top_symbols = [symbol for symbol, _ in symbol_counts[:10]]
            for symbol in top_symbols:
                # Get count of days for this symbol
                day_count = session.query(
                    func.date(YfBar1d.timestamp),
                    func.count(YfBar1d.id)
                ).filter(
                    YfBar1d.symbol == symbol
                ).group_by(
                    func.date(YfBar1d.timestamp)
                ).count()
                
                # Get the date range for this symbol
                symbol_oldest = session.query(
                    func.min(YfBar1d.timestamp)
                ).filter(
                    YfBar1d.symbol == symbol
                ).scalar()
                
                symbol_newest = session.query(
                    func.max(YfBar1d.timestamp)
                ).filter(
                    YfBar1d.symbol == symbol
                ).scalar()
                
                if symbol_oldest and symbol_newest:
                    calendar_days = (symbol_newest.date() - symbol_oldest.date()).days + 1
                    weekdays = sum(1 for i in range(calendar_days) if (symbol_oldest.date() + timedelta(i)).weekday() < 5)
                    coverage = day_count / weekdays if weekdays > 0 else 0
                    
                    logger.info(f"Symbol {symbol}: {day_count} days of data, {coverage:.1%} weekday coverage")
                
    except Exception as e:
        logger.error(f"Error analyzing database statistics: {e}")
    finally:
        session.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Daily bar data loader')
    parser = add_environment_args(parser)
    parser.add_argument('--period', type=str, 
                        choices=['last_day', 'last_week', 'last_month', 'last_quarter', 'ytd', 'last_year', 'full'],
                        help='Predefined time period for data loading')
    parser.add_argument('--start-date', type=str, 
                        help=f'Custom start date in YYYY-MM-DD format (default: {DEFAULT_START_DATE})')
    parser.add_argument('--source', type=str, default=DEFAULT_SOURCE,
                        choices=['yahoo', 'norgate'],
                        help=f'Data source to use (default: {DEFAULT_SOURCE})')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help=f'Number of worker threads (default: {NUM_WORKERS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for API requests to avoid rate limiting (default: {BATCH_SIZE})')
    parser.add_argument('--input', type=str,
                        help='Input file with comma-separated symbols')
    parser.add_argument('--force', action='store_true',
                        help='Force update existing records')
    parser.add_argument('--symbol', type=str,
                        help='Process a single symbol')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only show database statistics, do not update data')
    parser.add_argument('--no-smart-update', action='store_true',
                        help='Disable smart update (ignore database last timestamp)')
    parser.add_argument('--auto-add-symbols', action='store_true',
                        help='Automatically add missing symbols to the database')
    # ENHANCED RATE-LIMIT HANDLING: New option to resume from saved progress
    parser.add_argument('--resume', action='store_true',
                        help='Resume from saved progress file')
    # ENHANCED RATE-LIMIT HANDLING: New option for chunk size
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='Number of symbols to process before taking a longer break (default: 100)')
    
    args = parser.parse_args()
    
    # Setup database connection based on environment
    global engine, Session, stats
    engine = get_database_engine(args.env)
    Session = get_session_maker(args.env)
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting daily bar data loader using {env_name} database")
    
    # Show stats only if requested
    if args.stats_only:
        analyze_database_stats()
        return
    
    # Smart update setting
    smart_update = not args.no_smart_update
    
    # Determine date range - if smart_update is False, use standard date range calculation
    if args.start_date:
        start_date, end_date = get_date_range(
            start_date=args.start_date, 
            smart_update=False
        )
        logger.info(f"Using custom start date: {start_date.strftime('%Y-%m-%d')}")
    elif args.period and not smart_update:
        start_date, end_date = get_date_range(
            period=args.period, 
            smart_update=False
        )
        logger.info(f"Using period: {args.period} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
    else:
        # If using smart update with single symbol, get symbol-specific date range
        if args.symbol and smart_update:
            start_date, end_date = get_date_range(
                period=args.period,
                symbol=args.symbol.upper(),
                smart_update=True
            )
        # Otherwise get global date range for initial setting (individual symbols will use their own in batch mode)
        else:
            start_date, end_date = get_date_range(
                period=args.period or 'last_year',
                smart_update=smart_update
            )
        
        if smart_update:
            logger.info(f"Using smart update with start date: {start_date.strftime('%Y-%m-%d')}")
        else:
            period_name = args.period or 'last_year'
            logger.info(f"Using default period of {period_name} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
    
    # Get symbols to process
    if args.symbol:
        symbols = [args.symbol.upper()]
        logger.info(f"Processing single symbol: {args.symbol}")
    else:
        symbols = get_symbols(args.input)
    
    # ENHANCED RATE-LIMIT HANDLING: Handle resume from saved progress
    if args.resume:
        completed, failed = load_progress_file()
        if completed:
            # Remove completed symbols from the list
            symbols = [s for s in symbols if s not in completed]
            logger.info(f"Resuming operation: {len(completed)} symbols already processed, {len(symbols)} remaining")
            
            # If there were failures from previous run, log them
            if failed:
                logger.warning(f"Note: {len(failed)} symbols failed in previous run")
    
    # Initialize stats
    stats = Stats(len(symbols))
    
    # ENHANCED RATE-LIMIT HANDLING: Chunk symbols to avoid daily limits
    if len(symbols) > args.chunk_size and not args.symbol:
        logger.info(f"Splitting {len(symbols)} symbols into chunks of {args.chunk_size}")
        symbol_chunks = chunk_symbols(symbols, args.chunk_size)
        logger.info(f"Created {len(symbol_chunks)} chunks")
    else:
        # For single symbol or small lists, use one chunk
        symbol_chunks = [symbols]
    
    # Process symbols
    if len(symbols) == 1:
        # Single symbol mode - no threading needed
        process_symbol(symbols[0], start_date, end_date, args.source, args.force, args.auto_add_symbols)
    else:
        # Process each chunk with a longer break in between
        all_completed = []
        all_failed = []
        
        for chunk_idx, symbol_chunk in enumerate(symbol_chunks):
            logger.info(f"\n=== Processing chunk {chunk_idx+1}/{len(symbol_chunks)} with {len(symbol_chunk)} symbols ===")
            
            # Multi-symbol mode with threading
            batch_size = math.ceil(len(symbol_chunk) / args.workers)
            symbol_batches = [symbol_chunk[i:i + batch_size] for i in range(0, len(symbol_chunk), batch_size)]
            
            logger.info(f"Processing {len(symbol_chunk)} symbols with {args.workers} workers")
            logger.info(f"Data source: {args.source}")
            logger.info(f"Force update: {'Yes' if args.force else 'No'}")
            logger.info(f"Smart update: {'Yes' if smart_update else 'No'}")
            logger.info(f"Auto-add symbols: {'Yes' if args.auto_add_symbols else 'No'}")
            
            chunk_completed = []
            chunk_failed = []
            
            try:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = [
                        executor.submit(
                            process_symbol_batch, 
                            batch, 
                            start_date, 
                            end_date, 
                            args.source,
                            args.force,
                            smart_update,
                            args.auto_add_symbols,
                            args.batch_size
                        ) 
                        for batch in symbol_batches
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            batch_completed, batch_failed = future.result()
                            chunk_completed.extend(batch_completed)
                            chunk_failed.extend(batch_failed)
                        except Exception as e:
                            logger.error(f"Error in worker thread: {e}")
                            
            except KeyboardInterrupt:
                logger.warning("\nProcess interrupted by user!")
                # Save progress before exiting
                save_progress_file(all_completed + chunk_completed, all_failed + chunk_failed)
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                # Save progress before exiting
                save_progress_file(all_completed + chunk_completed, all_failed + chunk_failed)
                break
            
            # Add chunk results to overall totals
            all_completed.extend(chunk_completed)
            all_failed.extend(chunk_failed)
            
            # Save progress after each chunk
            save_progress_file(all_completed, all_failed)
            
            # ENHANCED RATE-LIMIT HANDLING: Take a longer break between chunks 
            if chunk_idx < len(symbol_chunks) - 1:
                wait_time = random.uniform(180, 300)  # 3-5 minutes between chunks
                logger.info(f"Taking a longer break between chunks ({wait_time:.1f} seconds)...")
                time.sleep(wait_time)
    
    # Print summary
    elapsed_time = (datetime.now() - stats.start_time).total_seconds() / 60
    
    logger.info("\nProcessing Summary:")
    logger.info(f"Total symbols processed: {stats.processed}/{len(symbols)}")
    logger.info(f"Successfully processed: {stats.successful}")
    logger.info(f"Failed: {stats.failed}")
    logger.info(f"Total bars added: {stats.bars_added}")
    logger.info(f"Total bars updated: {stats.bars_updated}")
    logger.info(f"Total bars skipped: {stats.skipped_bars}")
    logger.info(f"Total processing time: {elapsed_time:.1f} minutes")
    
    if elapsed_time > 0:
        logger.info(f"Processing rate: {stats.processed/elapsed_time:.1f} symbols per minute")
    
    # Show database statistics
    analyze_database_stats()


if __name__ == "__main__":
    # Import yfinance here rather than at the top
    # This way if there's a rate limit error path before this, 
    # we don't waste the import
    import yfinance as yf
    
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        # Save progress before exiting
        save_progress_file()
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        # Save progress before exiting
        save_progress_file()


        
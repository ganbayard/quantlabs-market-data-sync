import os
import sys
import argparse
import logging
import math
import threading
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import YfBar1d
from scripts.common_function import get_database_engine, add_environment_args, get_session_maker
from scripts.common_function import NorgateDataLoader

# Constants and Configuration
NUM_WORKERS = 6
BATCH_SIZE = 100
MIN_DELAY = 0.2
MAX_DELAY = 0.5

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"norgate_bar_loader_{datetime.now().strftime('%Y%m%d')}.log"

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
                logger.info(f"Bars added: {self.bars_added}, Bars updated: {self.bars_updated}, Failed: {self.failed}")


def get_thread_session():
    """Get thread-local SQLAlchemy session"""
    if not hasattr(thread_local, "session"):
        thread_local.session = Session()
    return thread_local.session


def get_date_range(period=None, start_date=None):
    """Get date range based on period or start date"""
    end_date = datetime.now()
    
    if start_date:
        # Parse start date from string
        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid start date format: {start_date}. Using default.")
                start_date = datetime.now() - timedelta(days=365)  # Default to 1 year
        return start_date, end_date
    
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


def process_symbol_data(symbol, df):
    """Process data for a single symbol and add to database"""
    session = get_thread_session()
    added = 0
    updated = 0
    
    try:
        for index, row in df.iterrows():
            try:
                # Prepare data
                bar_data = {
                    'symbol': symbol,
                    'timestamp': index,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
                }
                
                # Check if the record already exists
                existing_record = session.query(YfBar1d).filter_by(
                    symbol=symbol, timestamp=index
                ).first()
                
                if existing_record:
                    # Update the existing record
                    for key, value in bar_data.items():
                        setattr(existing_record, key, value)
                    updated += 1
                else:
                    # Insert new record
                    session.add(YfBar1d(**bar_data))
                    added += 1
                
                # Commit every 100 records to avoid large transactions
                if (added + updated) % 100 == 0:
                    session.commit()
                    
            except Exception as e:
                logger.error(f"Error processing bar for {symbol} at {index}: {e}")
                continue
        
        # Final commit for any remaining records
        session.commit()
        logger.info(f"Processed {symbol}: {added} bars added, {updated} bars updated")
        
        # Update global stats
        stats.increment('bars_added', added)
        stats.increment('bars_updated', updated)
        stats.increment('successful')
        return True
        
    except Exception as e:
        logger.error(f"Error processing symbol {symbol}: {e}")
        session.rollback()
        stats.increment('failed')
        return False
    finally:
        stats.increment('processed')


def process_symbol_batch(symbols, start_date, end_date):
    """Process a batch of symbols"""
    logger.info(f"Starting batch processing of {len(symbols)} symbols")
    
    try:
        # Create loader for this batch
        loader = NorgateDataLoader()
        
        # Load data for all symbols in batch
        assets = loader.load_data(start_date=start_date, end_date=end_date, symbols=symbols)
        
        # Process each symbol's data
        for symbol, df in assets.items():
            if df is not None and not df.empty:
                process_symbol_data(symbol, df)
                # Small delay to avoid overloading database
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            else:
                logger.warning(f"No data found for symbol {symbol}")
                stats.increment('failed')
                stats.increment('processed')
        
        logger.info(f"Completed batch processing of {len(symbols)} symbols")
    except ImportError as e:
        logger.error(f"Norgate SDK import error: {e}. Please install the Norgate Data SDK.")
        for symbol in symbols:
            stats.increment('failed')
            stats.increment('processed')
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        for symbol in symbols:
            stats.increment('failed')
            stats.increment('processed')


def get_symbols(input_file=None):
    """Get symbols to process from file or use default from data loader"""
    # If input file specified, read from it
    if (input_file):
        try:
            with open(input_file, 'r') as f:
                content = f.read().strip()
                symbols = [symbol.strip() for symbol in content.split(',') if symbol.strip()]
            logger.info(f"Loaded {len(symbols)} symbols from {input_file}")
            return symbols
        except Exception as e:
            logger.error(f"Error reading symbols from {input_file}: {e}")
    
    # Otherwise, get symbols from the NorgateDataLoader
    loader = NorgateDataLoader()
    symbols = loader.symbols
    logger.info(f"Using {len(symbols)} symbols from NorgateDataLoader")
    return symbols


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Norgate daily bar data loader')
    parser = add_environment_args(parser)
    parser.add_argument('--period', type=str, 
                      choices=['last_day', 'last_week', 'last_month', 'last_quarter', 'ytd', 'last_year', 'full'],
                      help='Predefined time period for data loading')
    parser.add_argument('--start-date', type=str, 
                      help='Custom start date in YYYY-MM-DD format')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                      help=f'Number of worker threads (default: {NUM_WORKERS})')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, 
                      help=f'Number of symbols per batch (default: {BATCH_SIZE})')
    parser.add_argument('--input', type=str,
                      help='Input file with comma-separated symbols')
    
    args = parser.parse_args()
    
    # Setup database connection based on environment
    global engine, Session, stats
    engine = get_database_engine(args.env)
    Session = get_session_maker(args.env)
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting Norgate daily bar loader using {env_name} database")
    
    # Get date range based on period or start date
    if args.start_date:
        start_date, end_date = get_date_range(start_date=args.start_date)
        logger.info(f"Using custom start date: {start_date}")
    elif args.period:
        start_date, end_date = get_date_range(period=args.period)
        logger.info(f"Using period: {args.period}")
    else:
        # Default to last month if neither is specified
        start_date, end_date = get_date_range(period='last_month')
        logger.info(f"Using default period of last month")
    
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Get symbols to process
    symbols = get_symbols(args.input)
    
    # Initialize global stats
    stats = Stats(len(symbols))
    
    # Split symbols into batches
    batch_size = args.batch
    symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    logger.info(f"Split {len(symbols)} symbols into {len(symbol_batches)} batches of {batch_size}")
    
    # Process batches with threading
    logger.info(f"Starting processing with {args.workers} workers...")
    
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_symbol_batch, batch, start_date, end_date) 
                for batch in symbol_batches
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in worker thread: {e}")
    
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    # Print final statistics
    elapsed_time = (datetime.now() - stats.start_time).total_seconds() / 60
    
    logger.info("\nProcessing Summary:")
    logger.info(f"Total symbols processed: {stats.processed}/{len(symbols)}")
    logger.info(f"Successfully processed: {stats.successful}")
    logger.info(f"Failed: {stats.failed}")
    logger.info(f"Total bars added: {stats.bars_added}")
    logger.info(f"Total bars updated: {stats.bars_updated}")
    logger.info(f"Total processing time: {elapsed_time:.1f} minutes")
    
    if elapsed_time > 0:
        logger.info(f"Processing rate: {stats.processed/elapsed_time:.1f} symbols per minute")
    
    logger.info("Norgate bar data update completed")


if __name__ == "__main__":
    main()

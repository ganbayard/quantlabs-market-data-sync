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
    get_session_maker,
    YahooFinanceLoader,  
    NorgateDataLoader   
)

# Constants and Configuration
NUM_WORKERS = 6
MIN_DELAY = 0.2
MAX_DELAY = 1.0
DEFAULT_START_DATE = "2020-01-01"  
DEFAULT_SOURCE = "yahoo"  

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
                start_date = datetime.strptime(DEFAULT_START_DATE, '%Y-%m-%d')
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


def get_data_loader(source):
    """Get the appropriate data loader based on source"""
    if source.lower() == 'norgate':
        return NorgateDataLoader()
    elif source.lower() == 'yahoo':
        return YahooFinanceLoader()
    else:
        logger.warning(f"Unknown source '{source}', using Yahoo Finance")
        return YahooFinanceLoader()


def process_symbol(symbol, start_date, end_date, data_source, force_update=False):
    """Process a single symbol's historical data"""
    session = get_thread_session()
    added = 0
    updated = 0
    
    try:
        # Load data from source
        loader = get_data_loader(data_source)
        df = loader.load_symbol_data(symbol, start_date, end_date)
        
        if df is None or df.empty:
            logger.warning(f"No data found for {symbol}")
            stats.increment('failed')
            return 0, 0
        
        # Process each bar
        for index, row in df.iterrows():
            try:
                # Normalize column names if needed
                open_price = row.get('Open', row.get('open', None))
                high_price = row.get('High', row.get('high', None))
                low_price = row.get('Low', row.get('low', None))
                close_price = row.get('Close', row.get('close', None))
                volume = row.get('Volume', row.get('volume', None))
                
                # Skip rows with missing data
                if any(x is None for x in [open_price, high_price, low_price, close_price, volume]):
                    continue
                
                # Check if the record already exists
                timestamp = index.to_pydatetime() if hasattr(index, 'to_pydatetime') else index
                
                existing_record = None
                if not force_update:
                    existing_record = session.query(YfBar1d).filter_by(
                        symbol=symbol, timestamp=timestamp
                    ).first()
                
                if existing_record:
                    # Update the existing record
                    existing_record.open = float(open_price)
                    existing_record.high = float(high_price)
                    existing_record.low = float(low_price)
                    existing_record.close = float(close_price)
                    existing_record.volume = float(volume)
                    updated += 1
                else:
                    # Insert new record
                    bar_data = YfBar1d(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=float(open_price),
                        high=float(high_price),
                        low=float(low_price),
                        close=float(close_price),
                        volume=float(volume)
                    )
                    session.add(bar_data)
                    added += 1
                
                # Commit in batches to avoid large transactions
                if (added + updated) % 100 == 0:
                    session.commit()
                    
            except Exception as e:
                logger.error(f"Error processing bar for {symbol} at {index}: {e}")
                continue
        
        # Final commit for any remaining records
        session.commit()
        stats.increment('bars_added', added)
        stats.increment('bars_updated', updated)
        stats.increment('successful')
        logger.info(f"Processed {symbol}: {added} bars added, {updated} bars updated")
        return added, updated
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        stats.increment('failed')
        return 0, 0
    finally:
        stats.increment('processed')


def process_symbol_batch(symbols, start_date, end_date, data_source, force_update):
    """Process a batch of symbols"""
    thread_name = threading.current_thread().name
    logger.info(f"[{thread_name}] Starting batch processing of {len(symbols)} symbols")
    
    for symbol in symbols:
        process_symbol(symbol, start_date, end_date, data_source, force_update)
        # Add a small delay to avoid hammering the API
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    
    logger.info(f"[{thread_name}] Completed batch processing")


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
    parser.add_argument('--input', type=str,
                        help='Input file with comma-separated symbols')
    parser.add_argument('--force', action='store_true',
                        help='Force update existing records')
    parser.add_argument('--symbol', type=str,
                        help='Process a single symbol')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only show database statistics, do not update data')
    
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
    
    # Determine date range
    if args.start_date:
        start_date, end_date = get_date_range(start_date=args.start_date)
        logger.info(f"Using custom start date: {start_date.strftime('%Y-%m-%d')}")
    elif args.period:
        start_date, end_date = get_date_range(period=args.period)
        logger.info(f"Using period: {args.period} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
    else:
        # Default to last year if neither is specified
        start_date, end_date = get_date_range(period='last_year')
        logger.info(f"Using default period of last year ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
    
    # Get symbols to process
    if args.symbol:
        symbols = [args.symbol.upper()]
        logger.info(f"Processing single symbol: {args.symbol}")
    else:
        symbols = get_symbols(args.input)
    
    # Initialize stats
    stats = Stats(len(symbols))
    
    # Process symbols
    if len(symbols) == 1:
        # Single symbol mode - no threading needed
        process_symbol(symbols[0], start_date, end_date, args.source, args.force)
    else:
        # Multi-symbol mode with threading
        batch_size = math.ceil(len(symbols) / args.workers)
        symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        logger.info(f"Processing {len(symbols)} symbols with {args.workers} workers")
        logger.info(f"Data source: {args.source}")
        logger.info(f"Force update: {'Yes' if args.force else 'No'}")
        
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        process_symbol_batch, 
                        batch, 
                        start_date, 
                        end_date, 
                        args.source,
                        args.force
                    ) 
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
    
    # Print summary
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
    
    # Show database statistics
    analyze_database_stats()


if __name__ == "__main__":
    main()
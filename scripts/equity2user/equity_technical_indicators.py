import pandas as pd
import numpy as np
import argparse
import logging
import math
import random
import threading
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from sqlalchemy import func, distinct

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import EquityTechnicalIndicator, YfBar1d
from scripts.common_function import (
    get_database_engine, 
    add_environment_args, 
    get_session_maker
)

# Constants and Configuration
NUM_WORKERS = 6
BATCH_SIZE = 100
MIN_DELAY = 0.5
MAX_DELAY = 1.5
MAX_RETRIES = 2
DEFAULT_SAFETY_DAYS = 3  # Days to look back when using latest timestamp (to catch any missed days)

# Period constants (in days)
PERIOD_DAY = 3  # 3 days to account for weekends
PERIOD_WEEK = 9  # 9 days to account for weekends and holidays
PERIOD_MONTH = 30
PERIOD_YEAR = 365
PERIOD_FIVE_YEAR = 1825
DEFAULT_PERIOD = PERIOD_DAY

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs" 
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"technical_indicators_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define default input file path
DEFAULT_INPUT_FILE = project_root / "scripts/symbols/stock_symbols.txt"

# Thread-local storage
thread_local = threading.local()

class Stats:
    def __init__(self, total_symbols):
        self.processed = 0
        self.successful = 0
        self.indicators_added = 0
        self.indicators_updated = 0
        self.failed = 0
        self.skipped = 0
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
                logger.info(f"Indicators added: {self.indicators_added}, updated: {self.indicators_updated}, failed: {self.failed}")

def get_thread_session():
    """Get or create a thread-local database session"""
    if not hasattr(thread_local, "session"):
        thread_local.session = Session()
    return thread_local.session

def get_latest_date(symbol=None, session=None):
    """
    Get the latest date in the database for a symbol or all symbols
    
    Args:
        symbol: Symbol to check (if None, check all symbols)
        session: SQLAlchemy session to use (if None, create a new one)
    
    Returns:
        date or None: Latest date in the database
    """
    if session is None:
        session = get_thread_session()
    
    try:
        query = session.query(func.max(EquityTechnicalIndicator.date))
        if symbol:
            query = query.filter(EquityTechnicalIndicator.symbol == symbol)
        
        latest = query.scalar()
        return latest
    except Exception as e:
        logger.error(f"Error getting latest date: {e}")
        return None

def get_date_range(period=None, start_date=None, symbol=None, smart_update=True):
    """
    Get date range based on period, start date, or latest timestamp in database
    
    Args:
        period: Predefined time period (last_day, last_week, etc.)
        start_date: Explicit start date (overrides period)
        symbol: Symbol to check for latest timestamp
        smart_update: Whether to use the latest date in database as reference
    
    Returns:
        tuple: (start_date, end_date)
    """
    end_date = datetime.now().date()
    
    # Case 1: Explicit start date provided
    if start_date:
        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            except ValueError:
                logger.error(f"Invalid start date format: {start_date}. Using default.")
                start_date = datetime.now().date() - timedelta(days=PERIOD_YEAR)
        return start_date, end_date
    
    # Case 2: Smart update enabled - check database for latest date
    if smart_update:
        latest_date = get_latest_date(symbol)
        if latest_date:
            if isinstance(latest_date, datetime):
                latest_date = latest_date.date()
                
            # Use latest date minus safety days to catch any missed days
            smart_start_date = latest_date - timedelta(days=DEFAULT_SAFETY_DAYS)
            logger.info(f"Using latest date from database: {latest_date} (minus {DEFAULT_SAFETY_DAYS} safety days)")
            return smart_start_date, end_date
    
    # Case 3: Use predefined period
    if period == 'last_day':
        start_date = end_date - timedelta(days=PERIOD_DAY)
    elif period == 'last_week':
        start_date = end_date - timedelta(days=PERIOD_WEEK)
    elif period == 'last_month':
        start_date = end_date - timedelta(days=PERIOD_MONTH)
    elif period == 'last_year':
        start_date = end_date - timedelta(days=PERIOD_YEAR)
    elif period == '5_year':
        start_date = end_date - timedelta(days=PERIOD_FIVE_YEAR)
    else:
        logger.warning(f"Invalid period '{period}', using default of 1 day")
        start_date = end_date - timedelta(days=PERIOD_DAY)
    
    return start_date, end_date

def money_flow_index(high, low, close, volume, length=14): 
    """Calculate Money Flow Index (MFI)"""
    tp = (high + low + close) / 3
    raw_money_flow = tp * volume
    pos_flow = raw_money_flow.where(tp > tp.shift(1), 0)
    neg_flow = raw_money_flow.where(tp < tp.shift(1), 0)
    pos_mf_sum = pos_flow.rolling(window=length).sum()
    negat_mf_sum = neg_flow.rolling(window=length).sum()
    money_flow_ratio = pos_mf_sum / negat_mf_sum
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi

def trend_intensity_calc(close_series):
    """Calculate Trend Intensity based on moving averages"""
    avgc7 = close_series.rolling(7).mean()
    avgc65 = close_series.rolling(65).mean()
    return avgc7/avgc65

def process_stock_data(symbol, historical_data):
    """Process historical stock data and store in database"""
    session = get_thread_session()
    thread_name = threading.current_thread().name
    added = 0
    updated = 0
    skipped = 0
    
    try:
        # Process each day's data
        for day_data in historical_data:
            day_date = day_data['date']
            
            # Check if we already have data for this symbol and date
            existing = session.query(EquityTechnicalIndicator).filter_by(
                symbol=symbol, 
                date=day_date
            ).first()
            
            if existing:
                # Update existing record
                existing.mfi = day_data['mfi']
                existing.trend_intensity = day_data['trend_intensity']
                existing.persistent_ratio = day_data['persistent_ratio']
                existing.updated_at = datetime.now()
                updated += 1
            else:
                # Create new record
                new_record = EquityTechnicalIndicator(
                    symbol=symbol,
                    date=day_date,
                    mfi=day_data['mfi'],
                    trend_intensity=day_data['trend_intensity'],
                    persistent_ratio=day_data['persistent_ratio'],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                session.add(new_record)
                added += 1

            # Commit in batches to avoid large transactions
            if (added + updated) % 50 == 0:
                try:
                    session.commit()
                except Exception as e:
                    logger.error(f"Error committing batch for {symbol}: {e}")
                    session.rollback()
                    skipped += (added + updated) % 50
                    
        # Final commit for any remaining records
        try:
            session.commit()
            stats.increment('indicators_added', added)
            stats.increment('indicators_updated', updated)
            stats.increment('skipped', skipped)
            stats.increment('successful')
            logger.info(f"[{thread_name}] Successfully processed {symbol}: {added} added, {updated} updated, {skipped} skipped")
            return True
        except Exception as e:
            logger.error(f"Error in final commit for {symbol}: {e}")
            session.rollback()
            stats.increment('failed')
            return False
        
    except Exception as e:
        logger.error(f"[{thread_name}] Error processing {symbol}: {str(e)}")
        session.rollback()
        stats.increment('failed')
        return False

def process_symbols(symbols, start_date, end_date, smart_update=True):
    """Process a batch of symbols"""
    thread_name = threading.current_thread().name
    logger.info(f"[{thread_name}] Starting processing of {len(symbols)} symbols")
    
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
            
            # Calculate technical indicators from price data
            historical_data = calculate_historical_data_for_symbol(symbol, symbol_start_date, symbol_end_date)
            
            if historical_data:
                process_stock_data(symbol, historical_data)
            else:
                logger.warning(f"No data found for symbol {symbol}")
                stats.increment('skipped')
                
        except Exception as e:
            logger.error(f"Unexpected error processing {symbol}: {str(e)}")
            stats.increment('failed')
        finally:
            stats.increment('processed')
            # Random delay between symbols
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            time.sleep(delay)
    
    logger.info(f"[{thread_name}] Completed batch processing")

def calculate_historical_data_for_symbol(symbol, start_date, end_date):
    """
    Calculate technical indicators from historical price data for a specific symbol and date range
    """
    session = get_thread_session()
    
    try:
        # Query OHLC data from yf_daily_bar table for the specific date range
        bars = session.query(YfBar1d).filter(
            YfBar1d.symbol == symbol,
            YfBar1d.timestamp >= start_date,
            YfBar1d.timestamp <= end_date
        ).order_by(YfBar1d.timestamp.asc()).all()
        
        if not bars:
            logger.warning(f"No price data found for {symbol} between {start_date} and {end_date}")
            return []
            
        # Get additional historical data for accurate calculations (lookback window)
        lookback = 65  # Maximum lookback needed for calculations (trend intensity uses 65-day MA)
        lookback_date = start_date - timedelta(days=lookback * 2)  # Double to account for weekends/holidays
        
        lookback_bars = session.query(YfBar1d).filter(
            YfBar1d.symbol == symbol,
            YfBar1d.timestamp >= lookback_date,
            YfBar1d.timestamp < start_date
        ).order_by(YfBar1d.timestamp.asc()).all()
        
        # Combine lookback and target period data
        all_bars = lookback_bars + bars
        
        if len(all_bars) < lookback:
            logger.warning(f"Insufficient data for {symbol} to calculate indicators (need at least {lookback} days, got {len(all_bars)})")
            return []
            
        # Convert to pandas DataFrame for easier calculation
        data = pd.DataFrame([
            {
                'date': bar.timestamp.date(),
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            } for bar in all_bars
        ])
        
        # Calculate technical indicators
        
        # 1. Money Flow Index
        try:
            mfi_values = money_flow_index(data['high'], data['low'], data['close'], data['volume'])
            data['mfi'] = mfi_values
        except Exception as e:
            logger.error(f"Error calculating MFI for {symbol}: {e}")
            data['mfi'] = np.nan
        
        # 2. Trend Intensity
        try:
            data['trend_intensity'] = trend_intensity_calc(data['close'])
        except Exception as e:
            logger.error(f"Error calculating trend intensity for {symbol}: {e}")
            data['trend_intensity'] = np.nan
        
        # 3. Persistence Ratio (measures consistency of price movements)
        try:
            # Calculate daily returns
            data['return'] = data['close'].pct_change()
            
            # Calculate 14-day cumulative return
            data['cum_return_14d'] = data['return'].rolling(window=14).sum()
            
            # Calculate persistence ratio (absolute cumulative return / sum of absolute daily returns)
            abs_cum_return = data['cum_return_14d'].abs()
            sum_abs_returns = data['return'].abs().rolling(window=14).sum()
            
            # Avoid division by zero
            data['persistent_ratio'] = np.where(
                sum_abs_returns > 0,
                abs_cum_return / sum_abs_returns,
                1.0  # Default value when denominator is zero
            )
        except Exception as e:
            logger.error(f"Error calculating persistence ratio for {symbol}: {e}")
            data['persistent_ratio'] = np.nan
        
        # Filter to only include data for the requested date range
        target_data = data[data['date'] >= start_date]
        
        # Convert to list of dictionaries with only the needed fields
        symbol_data = []
        for _, row in target_data.iterrows():
            # Skip days with missing indicators
            if pd.isna(row['mfi']) or pd.isna(row['trend_intensity']) or pd.isna(row['persistent_ratio']):
                continue
                
            day_data = {
                'date': row['date'],
                'mfi': float(row['mfi']),
                'trend_intensity': float(row['trend_intensity']),
                'persistent_ratio': float(row['persistent_ratio'])
            }
            symbol_data.append(day_data)
        
        if not symbol_data:
            logger.warning(f"No valid technical indicators calculated for {symbol}")
            
        return symbol_data
        
    except Exception as e:
        logger.error(f"Error calculating historical data for {symbol}: {e}")
        return []

def get_symbols(input_file=None):
    """Get symbols from input file or use defaults"""
    # Default symbol file location
    if not input_file:
        input_file = DEFAULT_INPUT_FILE
    
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

def clean_old_data(history_days):
    """Remove data older than history_days to maintain database size"""
    session = get_thread_session()
    cutoff_date = date.today() - timedelta(days=history_days)
    
    try:
        # Count records to be deleted
        old_count = session.query(EquityTechnicalIndicator).filter(
            EquityTechnicalIndicator.date < cutoff_date
        ).count()
        
        # Delete old records
        if old_count > 0:
            session.query(EquityTechnicalIndicator).filter(
                EquityTechnicalIndicator.date < cutoff_date
            ).delete()
            session.commit()
            logger.info(f"Removed {old_count} historical records older than {cutoff_date}")
    except Exception as e:
        logger.error(f"Error cleaning old data: {str(e)}")
        session.rollback()

def analyze_technical_indicators():
    """Analyze and log stats about technical indicator data in the database"""
    session = Session()
    try:
        # Get basic counts
        total_indicators = session.query(EquityTechnicalIndicator).count()
        total_symbols = session.query(EquityTechnicalIndicator.symbol).distinct().count()
        
        logger.info("\nTechnical Indicator Database Statistics:")
        logger.info(f"Total indicators stored: {total_indicators:,}")
        logger.info(f"Total symbols: {total_symbols}")
        
        if total_symbols > 0:
            indicators_per_symbol = total_indicators / total_symbols
            logger.info(f"Average indicators per symbol: {indicators_per_symbol:.1f}")
        
        # Get date range
        oldest_indicator = session.query(EquityTechnicalIndicator).order_by(EquityTechnicalIndicator.date.asc()).first()
        newest_indicator = session.query(EquityTechnicalIndicator).order_by(EquityTechnicalIndicator.date.desc()).first()
        
        if oldest_indicator and newest_indicator:
            days_span = (newest_indicator.date - oldest_indicator.date).days if isinstance(newest_indicator.date, date) else 0
            logger.info(f"Date range: {oldest_indicator.date} to {newest_indicator.date} ({days_span} days)")
        
        # Get symbols with most and least data
        symbol_counts = session.query(
            EquityTechnicalIndicator.symbol, 
            func.count(EquityTechnicalIndicator.symbol).label('count')
        ).group_by(EquityTechnicalIndicator.symbol).order_by(func.count(EquityTechnicalIndicator.symbol).desc()).all()
        
        if symbol_counts:
            most_data = symbol_counts[0]
            least_data = symbol_counts[-1]
            logger.info(f"Symbol with most indicators: {most_data[0]} ({most_data[1]} indicators)")
            logger.info(f"Symbol with least indicators: {least_data[0]} ({least_data[1]} indicators)")
        
        # Get recent activity
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        indicators_today = session.query(EquityTechnicalIndicator).filter(EquityTechnicalIndicator.date == today).count()
        indicators_yesterday = session.query(EquityTechnicalIndicator).filter(EquityTechnicalIndicator.date == yesterday).count()
        
        logger.info(f"Indicators for today: {indicators_today}")
        logger.info(f"Indicators for yesterday: {indicators_yesterday}")
        
    except Exception as e:
        logger.error(f"Error analyzing database statistics: {e}")
    finally:
        session.close()

# Initialize global variables
engine = None
Session = None
stats = None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Historical Technical Indicators Processor')
    parser = add_environment_args(parser)
    parser.add_argument('--period', type=str, 
                  choices=['last_day', 'last_week', 'last_month', 'last_year', '5_year'],
                  default='last_day',
                  help='Period for historical data (default: last_day)')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                      help=f'Number of worker threads (default: {NUM_WORKERS})')
    parser.add_argument('--input', type=str,
                      help=f'Input file with comma-separated symbols (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--no-smart-update', action='store_true',
                      help='Disable smart update (ignore database last date)')
    parser.add_argument('--stats-only', action='store_true',
                      help='Only show database statistics, do not update data')
    parser.add_argument('--start-date', type=str,
                      help='Custom start date in YYYY-MM-DD format')
    parser.add_argument('--symbol', type=str,
                      help='Process a single symbol')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup database connection based on environment
    global engine, Session, stats
    env = getattr(args, 'env', 'dev')
    engine = get_database_engine(env)
    Session = get_session_maker(env)
    
    env_name = "PRODUCTION" if env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting technical indicators with {env_name} database")
    
    # Show stats only if requested
    if args.stats_only:
        analyze_technical_indicators()
        return
    
    # Smart update setting
    smart_update = not args.no_smart_update
   
    # Determine date range
    if args.start_date:
        start_date, end_date = get_date_range(start_date=args.start_date, smart_update=False)
        logger.info(f"Using custom start date: {start_date}")
    elif args.symbol and smart_update:
        start_date, end_date = get_date_range(symbol=args.symbol.upper(), smart_update=True)
    else:
        start_date, end_date = get_date_range(period=args.period, smart_update=smart_update)
        
        if smart_update:
            logger.info(f"Using smart update with start date: {start_date}")
        else:
            period_name = args.period or 'last_day'
            logger.info(f"Using period of {period_name} ({start_date} to {end_date})")
    
    # Get symbols to process
    if args.symbol:
        symbols = [args.symbol.upper()]
        logger.info(f"Processing single symbol: {args.symbol}")
    else:
        symbols = get_symbols(args.input)
    
    # Initialize stats
    stats = Stats(len(symbols))
    
    try:
        # Clean old data if needed
        if args.period == '5_year':
            clean_old_data(PERIOD_FIVE_YEAR)
        elif args.period == 'last_year':
            clean_old_data(PERIOD_YEAR)
            
        # Process symbols
        if len(symbols) == 1 or args.workers == 1:
            # Single symbol or single worker mode
            process_symbols(symbols, start_date, end_date, smart_update)
        else:
            # Multi-symbol mode with threading
            batch_size = math.ceil(len(symbols) / args.workers)
            symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
            
            logger.info(f"Processing {len(symbols)} symbols with {args.workers} workers")
            logger.info(f"Smart update: {'Yes' if smart_update else 'No'}")
            
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        process_symbols, 
                        batch, 
                        start_date, 
                        end_date, 
                        smart_update
                    ) 
                    for batch in symbol_batches
                ]
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error in worker thread: {e}")
        
        # Print summary
        elapsed_time = (datetime.now() - stats.start_time).total_seconds() / 60
        
        logger.info("\nProcessing Summary:")
        logger.info(f"Total symbols processed: {stats.processed}/{len(symbols)}")
        logger.info(f"Successfully processed: {stats.successful}")
        logger.info(f"Failed: {stats.failed}")
        logger.info(f"Total indicators added: {stats.indicators_added}")
        logger.info(f"Total indicators updated: {stats.indicators_updated}")
        logger.info(f"Total processing time: {elapsed_time:.1f} minutes")
        
        if elapsed_time > 0:
            logger.info(f"Processing rate: {stats.processed/elapsed_time:.1f} symbols per minute")
        
        # Show database statistics
        analyze_technical_indicators()
        
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        
    logger.info("Processing completed")

if __name__ == "__main__":
    main()
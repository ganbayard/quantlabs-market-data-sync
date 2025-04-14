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

# Period constants (in days)
PERIOD_DAY = 3  # 3 days to account for weekends
PERIOD_MONTH = 30
PERIOD_YEAR = 365
PERIOD_FIVE_YEAR = 1825
DEFAULT_PERIOD = PERIOD_DAY 

# Base directory setup
BASE_DIR = Path(__file__).parent
LOG_DIR = project_root / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"technical_indicators_{datetime.now().strftime('%Y%m%d')}.log"

# Define default input file path
DEFAULT_INPUT_FILE = project_root / "scripts/symbols/stock_symbols.txt"

# Thread-local storage
thread_local = threading.local()

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class Stats:
    def __init__(self, total_symbols):
        self.processed = 0
        self.successful = 0
        self.lock = threading.Lock()
        self.total_symbols = total_symbols
        self.start_time = datetime.now()
    
    def increment(self, attribute):
        with self.lock:
            setattr(self, attribute, getattr(self, attribute) + 1)
            if attribute == 'processed' and self.processed % 10 == 0:
                logger.info(f"Progress: {self.processed}/{self.total_symbols} ({(self.processed/self.total_symbols)*100:.1f}%)")
                logger.info(f"Successful: {self.successful}")

def get_thread_session():
    """Get or create a thread-local database session"""
    if not hasattr(thread_local, "session"):
        thread_local.session = Session()
    return thread_local.session

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
            else:
                # Create new record
                new_record = EquityTechnicalIndicator(
                    symbol=symbol,
                    date=day_date,
                    mfi=day_data['mfi'],
                    trend_intensity=day_data['trend_intensity'],
                    persistent_ratio=day_data['persistent_ratio']
                )
                session.add(new_record)
        
        # Commit all changes
        session.commit()
        stats.increment('successful')
        logger.info(f"[{thread_name}] Successfully processed {symbol} with {len(historical_data)} data points")
        return True
        
    except Exception as e:
        logger.error(f"[{thread_name}] Error processing {symbol}: {str(e)}")
        session.rollback()
        return False

def process_symbols(symbols, historical_data_dict):
    """Process a batch of symbols"""
    thread_name = threading.current_thread().name
    logger.info(f"[{thread_name}] Starting processing of {len(symbols)} symbols")
    
    for symbol in symbols:
        if symbol in historical_data_dict:
            process_stock_data(symbol, historical_data_dict[symbol])
        else:
            logger.warning(f"[{thread_name}] No data found for symbol {symbol}")
        
        stats.increment('processed')
        # Random delay between symbols
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)
    
    logger.info(f"[{thread_name}] Completed batch processing")

def calculate_historical_data(symbols, days=60):
    """
    Calculate real technical indicators from historical price data in the database
    """
    session = get_thread_session()
    historical_data = {}
    cutoff_date = date.today() - timedelta(days=days)
    
    logger.info(f"Fetching price data from yf_daily_bar for {len(symbols)} symbols...")
    
    # Process each symbol
    for symbol in symbols:
        try:
            # Query OHLC data from yf_daily_bar table
            bars = session.query(YfBar1d).filter(
                YfBar1d.symbol == symbol,
                YfBar1d.timestamp >= cutoff_date
            ).order_by(YfBar1d.timestamp.asc()).all()
            
            if not bars:
                logger.warning(f"No price data found for {symbol}, skipping")
                continue
                
            # Convert to pandas DataFrame for easier calculation
            data = pd.DataFrame([
                {
                    'date': bar.timestamp.date(),
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                } for bar in bars
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
            
            # Convert to list of dictionaries with only the needed fields
            symbol_data = []
            for _, row in data.iterrows():
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
            
            if symbol_data:
                historical_data[symbol] = symbol_data
                
        except Exception as e:
            logger.error(f"Error processing historical data for {symbol}: {e}")
    
    return historical_data

def get_technical_data(input_file_path=None, days=365):
    """
    Read symbols from input file and calculate technical indicators from real price data
    """
    # Use provided input file or default
    if input_file_path:
        input_file = Path(input_file_path)
    else:
        input_file = DEFAULT_INPUT_FILE
    
    logger.info(f"Reading symbols from {input_file}")
    try:
        with open(input_file, 'r') as f:
            content = f.read()
            symbols = [symbol.strip() for symbol in content.split(',') if symbol.strip()]
            logger.info(f"Found {len(symbols)} symbols in input file")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        # Fallback to a smaller set of symbols if file not found
        logger.warning("Using fallback symbol list")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "BAC", "WMT"]
    
    logger.info(f"Calculating technical indicators for {days} days of price data...")
    
    # Calculate indicators using real price data
    historical_data = calculate_historical_data(symbols, days)
    
    logger.info(f"Generated technical indicators for {len(historical_data)} symbols")
    return historical_data, symbols

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

def analyze_historical_indicators(history_days=365):
    """Analyze the historical technical indicators in the database"""
    from sqlalchemy import func
    
    session = Session()
    
    try:
        today = date.today()
        analysis_period = today - timedelta(days=min(history_days, 30))  # Analyze at most 30 days 
        
        # Count total records
        total_symbols = session.query(func.count(func.distinct(EquityTechnicalIndicator.symbol))).scalar()
        total_records = session.query(EquityTechnicalIndicator).count()
        
        logger.info("\nHistorical Technical Indicator Analysis:")
        logger.info(f"Total symbols tracked: {total_symbols}")
        logger.info(f"Total data points: {total_records}")
        
        # Calculate average records per symbol
        if total_symbols > 0:
            avg_records = total_records / total_symbols
            logger.info(f"Average data points per symbol: {avg_records:.1f}")
        
        # Calculate average MFI, trend intensity, and persistence ratio for the last month
        recent_avg_mfi = session.query(func.avg(EquityTechnicalIndicator.mfi)).filter(
            EquityTechnicalIndicator.date >= analysis_period
        ).scalar()
        
        recent_avg_trend = session.query(func.avg(EquityTechnicalIndicator.trend_intensity)).filter(
            EquityTechnicalIndicator.date >= analysis_period
        ).scalar()
        
        recent_avg_persistence = session.query(func.avg(EquityTechnicalIndicator.persistent_ratio)).filter(
            EquityTechnicalIndicator.date >= analysis_period
        ).scalar()
        
        logger.info("\nLast Month Average Values:")
        logger.info(f"Average MFI: {recent_avg_mfi:.2f}")
        logger.info(f"Average Trend Intensity: {recent_avg_trend:.2f}")
        logger.info(f"Average Persistence Ratio: {recent_avg_persistence:.2f}")
        
    except Exception as e:
        logger.error(f"Error analyzing technical indicators: {str(e)}")
    finally:
        session.close()

# Initialize global variables
engine = None
Session = None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Historical Technical Indicators Processor')
    parser = add_environment_args(parser)
    parser.add_argument('--period', type=str, 
                      choices=['last_day', 'last_month', 'last_year', '5_year'],
                      default='last_year',
                      help='Period for historical data (default: last_year)')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                      help=f'Number of worker threads (default: {NUM_WORKERS})')
    parser.add_argument('--input', type=str,
                      help=f'Input file with comma-separated symbols (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # More explicit environment handling with detailed logging
    if '--env prod' in sys.argv or '-e prod' in sys.argv:
        logger.warning("PRODUCTION environment specified in command line arguments!")
        args.env = 'prod'
    else:
        # Force to dev environment unless explicitly requested
        args.env = 'dev'
        logger.info("Using DEVELOPMENT environment (default)")
    
    # Setup database connection based on environment - MUST COME BEFORE ANY DB OPERATIONS
    global engine, Session, stats
    logger.info(f"Connecting to database using environment: {args.env}")
    engine = get_database_engine(args.env)
    Session = get_session_maker(args.env)
    
    # Set history days based on period argument
    if args.period == 'last_day':
        history_days = PERIOD_DAY
    elif args.period == 'last_month':
        history_days = PERIOD_MONTH
    elif args.period == 'last_year':
        history_days = PERIOD_YEAR
    elif args.period == '5_year':
        history_days = PERIOD_FIVE_YEAR
    else:
        history_days = DEFAULT_PERIOD
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting technical indicators history tracker using {env_name} database")
    logger.info(f"Processing period: {args.period} ({history_days} days)")
    
    try:
        # Clean old data first
        clean_old_data(history_days)  # Use history_days instead of DEFAULT_PERIOD
        
        # Get historical technical indicator data
        historical_data, symbols = get_technical_data(args.input, history_days)  # Pass the parameters!
        
        # Initialize global stats
        global stats
        stats = Stats(len(symbols))
        
        # Split symbols into batches for workers
        batch_size = math.ceil(len(symbols) / NUM_WORKERS)
        symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        logger.info(f"Starting processing with {NUM_WORKERS} workers...")
        logger.info(f"Total symbols to process: {len(symbols)}")
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit batches to thread pool
            futures = [executor.submit(process_symbols, batch, historical_data) for batch in symbol_batches]
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in worker thread: {str(e)}")
        
        # Print final statistics
        logger.info("\nProcessing Summary:")
        logger.info(f"Total symbols processed: {stats.processed}")
        logger.info(f"Successfully processed: {stats.successful}")
        logger.info(f"Failed: {stats.processed - stats.successful}")
        
        # Analyze the historical indicators
        analyze_historical_indicators()
        
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user!")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main()
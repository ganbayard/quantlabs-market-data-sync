import pandas as pd
import numpy as np
import argparse
import logging
import math
import threading
import time
import random
from datetime import datetime, timedelta, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Add project root to Python path
script_dir = Path(__file__).parent.parent.parent.parent
project_root = script_dir
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import Equity2User
from scripts.common_function import (
    get_database_engine, 
    add_environment_args, 
    get_session_maker
)

# Import from local modules
from scripts.equity2user.equity_recommendation.technical_analysis import (
    classify_stock, get_stock_metrics, get_sector_data
)
from scripts.equity2user.equity_recommendation.scoring import get_all_scores
from scripts.equity2user.equity_recommendation.liquiditation import calculate_buy_short_points
from scripts.equity2user.equity_recommendation.overbuy_oversold import determine_overbuy_oversold

# Constants and Configuration
NUM_WORKERS = 6
BATCH_SIZE = 100
MIN_DELAY = 0.5
MAX_DELAY = 1.5
MAX_RETRIES = 2

# Configure logging
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"equity_history_{datetime.now().strftime('%Y%m%d')}.log"

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

# Initialize global variables
engine = None
Session = None
stats = None

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

def process_stock_data(symbol, stock_data, sector_data):
    """Process stock data and store in database"""
    session = get_thread_session()
    thread_name = threading.current_thread().name
    
    try:
        # Get existing entry or create new one
        entry = session.query(Equity2User).filter_by(symbol=symbol).first()
        if not entry:
            entry = Equity2User(symbol=symbol)
            entry.created_at = datetime.now()
        
        # Core entity fields
        entry.symbol = symbol
        entry.is_active = True
        entry.status = 'active'
        entry.updated_at = datetime.now()
        
        # Technical Analysis fields
        adr_value = stock_data.get('ADR', 0)
        entry.risk_type = classify_stock(stock_data)
        entry.sector = sector_data.get(symbol, 'Unknown')
        entry.volume_spike = stock_data.get('Volume', '')
        
        # Handle NaN values for numeric fields
        mfi_value = stock_data.get('MFI', 50)
        entry.RSI = float(mfi_value) if not pd.isna(mfi_value) else 50.0
        
        entry.ADR = float(adr_value) if not pd.isna(adr_value) else 0.0
        
        # Set long term persistence (using Persistence Ratio)
        persistence_ratio = stock_data.get('Persistence Ratio', 1.0)
        entry.long_term_persistance = float(persistence_ratio) if not pd.isna(persistence_ratio) else 1.0
        
        # Set long term divergence (using Trend Intensity)
        trend_intensity_value = stock_data.get('Trend Intensity', 50.0)
        entry.long_term_divergence = float(trend_intensity_value) if not pd.isna(trend_intensity_value) else 50.0
        
        # Calculate overbuy/oversold value
        overbuy_value = determine_overbuy_oversold(
            entry.RSI, 
            trend_intensity_value / 50 if not pd.isna(trend_intensity_value) else 1.0,
            persistence_ratio if not pd.isna(persistence_ratio) else 1.0
        )
        entry.overbuy_oversold = float(overbuy_value) if not pd.isna(overbuy_value) else 0.0
        
        # Calculate buy and short points
        price = stock_data.get('Price', 0)
        volatility = stock_data.get('Volatility', 0)
        if pd.isna(price) or pd.isna(volatility) or pd.isna(adr_value):
            buy_point, short_point = 0.0, 0.0
        else:
            buy_point, short_point = calculate_buy_short_points(price, volatility, adr_value)
        
        entry.buy_point = float(buy_point) if not pd.isna(buy_point) else 0.0
        entry.short_point = float(short_point) if not pd.isna(short_point) else 0.0
        
        # Get and set all financial scores - these are generated, so shouldn't be NaN
        scores = get_all_scores()
        entry.earnings_date_score = scores['earnings_date_score']
        entry.income_statement_score = scores['income_statement_score']
        entry.cashflow_statement_score = scores['cashflow_statement_score']
        entry.balance_sheet_score = scores['balance_sheet_score']
        entry.rate_scoring = scores['rate_scoring']
        
        # Set dates 
        entry.recommended_date = datetime.now()
        
        # Save to database
        session.add(entry)
        session.commit()
        
        stats.increment('successful')
        logger.info(f"[{thread_name}] Successfully processed {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"[{thread_name}] Error processing {symbol}: {str(e)}")
        session.rollback()
        return False

def process_symbols(symbols, stock_data_dict, sector_data):
    """Process a batch of symbols"""
    thread_name = threading.current_thread().name
    logger.info(f"[{thread_name}] Starting processing of {len(symbols)} symbols")
    
    for symbol in symbols:
        if symbol in stock_data_dict:
            process_stock_data(symbol, stock_data_dict[symbol], sector_data)
        else:
            logger.warning(f"[{thread_name}] No data found for symbol {symbol}")
        
        stats.increment('processed')
        # Random delay between symbols
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)
    
    logger.info(f"[{thread_name}] Completed batch processing")

def get_stock_data_from_scanner(input_file_path=None, days=30):
    """
    Read symbols from input file and calculate real metrics from YfBar1d data
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
    
    logger.info("Calculating stock metrics using YfBar1d data...")
    
    # Get real metrics and sector data
    stock_data, sector_data = get_stock_metrics(symbols, days)
    
    logger.info(f"Finished preparing data for {len(stock_data)} symbols")
    return stock_data, sector_data, symbols

def analyze_equity_data():
    """Analyze the equity user data in the database"""
    from sqlalchemy import func
    
    session = get_thread_session()
    
    try:
        # Count total records
        total_records = session.query(Equity2User).count()
        
        if total_records == 0:
            logger.info("No equity user data found in database")
            return
            
        logger.info("\nEquity User Analysis:")
        logger.info(f"Total equity records: {total_records}")
        
        # Count by risk type
        aggressive_count = session.query(Equity2User).filter_by(risk_type="Aggressive").count()
        moderate_count = session.query(Equity2User).filter_by(risk_type="Moderate").count()
        conservative_count = session.query(Equity2User).filter_by(risk_type="Conservative").count()
        
        logger.info("\nRisk Distribution:")
        logger.info(f"Aggressive: {aggressive_count} ({aggressive_count/total_records*100:.1f}%)")
        logger.info(f"Moderate: {moderate_count} ({moderate_count/total_records*100:.1f}%)")
        logger.info(f"Conservative: {conservative_count} ({conservative_count/total_records*100:.1f}%)")
        
        # Analyze overbuy/oversold distribution using numeric ranges
        strong_sell_count = session.query(Equity2User).filter(Equity2User.overbuy_oversold > 60).count()
        sell_count = session.query(Equity2User).filter(Equity2User.overbuy_oversold.between(20, 60)).count()
        neutral_count = session.query(Equity2User).filter(Equity2User.overbuy_oversold.between(-20, 20)).count()
        buy_count = session.query(Equity2User).filter(Equity2User.overbuy_oversold.between(-60, -20)).count()
        strong_buy_count = session.query(Equity2User).filter(Equity2User.overbuy_oversold < -60).count()
        
        logger.info("\nOverbuy/Oversold Distribution:")
        logger.info(f"Strong Sell (> 60): {strong_sell_count} ({strong_sell_count/total_records*100:.1f}%)")
        logger.info(f"Sell (20 to 60): {sell_count} ({sell_count/total_records*100:.1f}%)")
        logger.info(f"Neutral (-20 to 20): {neutral_count} ({neutral_count/total_records*100:.1f}%)")
        logger.info(f"Buy (-60 to -20): {buy_count} ({buy_count/total_records*100:.1f}%)")
        logger.info(f"Strong Buy (< -60): {strong_buy_count} ({strong_buy_count/total_records*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error analyzing equity data: {str(e)}")
    finally:
        session.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Equity User History Generator')
    parser = add_environment_args(parser)
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
    
    # Set the session factory in technical_analysis module
    from scripts.equity2user.equity_recommendation import technical_analysis
    technical_analysis.set_session_factory(Session)
    
    # Fixed history days for weekly execution
    history_days = 7
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting equity user history generator using {env_name} database")
    logger.info(f"Processing period: {history_days} days")
    
    try:
        # Get stock data and sectors
        stock_data, sector_data, symbols = get_stock_data_from_scanner(args.input, history_days)
        
        # Initialize global stats
        stats = Stats(len(symbols))
        
        # Split symbols into batches for workers
        batch_size = math.ceil(len(symbols) / args.workers)
        symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        logger.info(f"Starting processing with {args.workers} workers...")
        logger.info(f"Total symbols to process: {len(symbols)}")
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit batches to thread pool
            futures = [executor.submit(process_symbols, batch, stock_data, sector_data) for batch in symbol_batches]
            
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
        
        # Analyze the equity data
        analyze_equity_data()
        
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user!")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main()
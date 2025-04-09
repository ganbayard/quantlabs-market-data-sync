import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Date, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta, date
import time
import logging
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import math
from data_helpers import get_us_stocks_with_sector
from dotenv import load_dotenv
import os

load_dotenv()

# Constants and Configuration
NUM_WORKERS = 6
BATCH_SIZE = 100
MIN_DELAY = 0.5
MAX_DELAY = 1.5
MAX_RETRIES = 2
HISTORY_DAYS = 60  # Store 2 months of history

# Base directory setup
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"

thread_local = threading.local()

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

# Database setup

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "stock_data")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
engine = create_engine(DATABASE_URL, pool_recycle=3600, pool_size=20, max_overflow=0)
Session = sessionmaker(bind=engine)
Base = declarative_base()

def setup_logging():
    """Setup logging configuration"""
    log_file = LOG_DIR / f"technical_indicators_{datetime.now().strftime('%Y%m%d')}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
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

class TechnicalIndicator(Base):
    __tablename__ = 'equity_technical_indicators_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), index=True)
    date = Column(Date, index=True)  # Date for this technical data point
    mfi = Column(Float)  # Money Flow Index
    trend_intensity = Column(Float)
    persistent_ratio = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

Base.metadata.create_all(engine)

def get_thread_session():
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
            existing = session.query(TechnicalIndicator).filter_by(
                symbol=symbol, 
                date=day_date
            ).first()
            
            if existing:
                # Update existing record
                existing.mfi = day_data['mfi']
                existing.trend_intensity = day_data['trend_intensity']
                existing.persistent_ratio = day_data['persistent_ratio']
                existing.updatedAt = datetime.now()
            else:
                # Create new record
                new_record = TechnicalIndicator(
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

def generate_historical_data(symbols, days=60):
    """
    Generate simulated historical technical indicator data for the given symbols
    
    In a real implementation, this would calculate actual indicators from historical price data
    """
    historical_data = {}
    today = date.today()
    
    # Generate random but somewhat realistic data for each symbol
    for symbol in symbols:
        # Base values with some randomness
        base_mfi = random.uniform(40, 60)
        base_trend = random.uniform(0.95, 1.05)
        base_persistence = random.uniform(0.8, 1.2)
        
        symbol_data = []
        
        # Generate data for each day with some continuity
        for i in range(days):
            day_date = today - timedelta(days=days-i-1)
            
            # Add some random walk to the values for realistic movement
            mfi_change = random.uniform(-5, 5)
            trend_change = random.uniform(-0.03, 0.03)
            persistence_change = random.uniform(-0.1, 0.1)
            
            # Update values with some constraints to keep them realistic
            base_mfi = max(0, min(100, base_mfi + mfi_change))
            base_trend = max(0.7, min(1.3, base_trend + trend_change))
            base_persistence = max(0.3, min(2.5, base_persistence + persistence_change))
            
            # Store the day's data
            day_data = {
                'date': day_date,
                'mfi': base_mfi,
                'trend_intensity': base_trend,
                'persistent_ratio': base_persistence
            }
            
            symbol_data.append(day_data)
        
        historical_data[symbol] = symbol_data
    
    return historical_data

def get_technical_data():
    """
    Read symbols from input file and generate technical indicator data
    In a real implementation, this would calculate actual indicators from historical price data
    """
    # Read symbols from the input file
    input_file = BASE_DIR  / "symbols.txt"
    
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
    
    logger.info(f"Generating {HISTORY_DAYS} days of historical technical data...")
    
    # Generate historical data for each symbol
    historical_data = generate_historical_data(symbols, HISTORY_DAYS)
    
    logger.info(f"Generated historical technical data for {len(historical_data)} symbols")
    return historical_data, symbols

def analyze_historical_indicators():
    """Analyze the historical technical indicators in the database"""
    session = Session()
    
    try:
        today = date.today()
        one_month_ago = today - timedelta(days=30)
        
        # Count total records
        total_symbols = session.query(func.count(func.distinct(TechnicalIndicator.symbol))).scalar()
        total_records = session.query(TechnicalIndicator).count()
        
        logger.info("\nHistorical Technical Indicator Analysis:")
        logger.info(f"Total symbols tracked: {total_symbols}")
        logger.info(f"Total data points: {total_records}")
        
        # Calculate average records per symbol
        if total_symbols > 0:
            avg_records = total_records / total_symbols
            logger.info(f"Average data points per symbol: {avg_records:.1f}")
        
        # Calculate average MFI, trend intensity, and persistence ratio for the last month
        recent_avg_mfi = session.query(func.avg(TechnicalIndicator.mfi)).filter(
            TechnicalIndicator.date >= one_month_ago
        ).scalar()
        
        recent_avg_trend = session.query(func.avg(TechnicalIndicator.trend_intensity)).filter(
            TechnicalIndicator.date >= one_month_ago
        ).scalar()
        
        recent_avg_persistence = session.query(func.avg(TechnicalIndicator.persistent_ratio)).filter(
            TechnicalIndicator.date >= one_month_ago
        ).scalar()
        
        logger.info("\nLast Month Average Values:")
        logger.info(f"Average MFI: {recent_avg_mfi:.2f}")
        logger.info(f"Average Trend Intensity: {recent_avg_trend:.2f}")
        logger.info(f"Average Persistence Ratio: {recent_avg_persistence:.2f}")
        
    except Exception as e:
        logger.error(f"Error analyzing technical indicators: {str(e)}")
    finally:
        session.close()

def clean_old_data():
    """Remove data older than HISTORY_DAYS to maintain database size"""
    session = get_thread_session()
    cutoff_date = date.today() - timedelta(days=HISTORY_DAYS)
    
    try:
        # Count records to be deleted
        old_count = session.query(TechnicalIndicator).filter(
            TechnicalIndicator.date < cutoff_date
        ).count()
        
        # Delete old records
        if old_count > 0:
            session.query(TechnicalIndicator).filter(
                TechnicalIndicator.date < cutoff_date
            ).delete()
            session.commit()
            logger.info(f"Removed {old_count} historical records older than {cutoff_date}")
    except Exception as e:
        logger.error(f"Error cleaning old data: {str(e)}")
        session.rollback()

def main():
    logger.info("Starting technical indicators history tracker")
    
    try:
        # Clean old data first
        clean_old_data()
        
        # Get historical technical indicator data
        historical_data, symbols = get_technical_data()
        
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
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import random
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from data_helpers import get_us_stocks_with_sector
from dotenv import load_dotenv
import os

# Constants and Configuration
NUM_WORKERS = 6
BATCH_SIZE = 100
MIN_DELAY = 0.5
MAX_DELAY = 1.5
MAX_RETRIES = 2

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
    log_file = LOG_DIR / f"equity_history_{datetime.now().strftime('%Y%m%d')}.log"
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

class EquityUserHistory(Base):
    __tablename__ = 'equity_2_user_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), index=True)
    risk_type = Column(String(20))  # 'Conservative', 'Aggressive', 'Moderate'
    sector = Column(String(100))
    volume_spike = Column(String(20))
    RSI = Column(Float)
    ADR = Column(Float)
    long_term_persistance = Column(Float)
    long_term_divergence = Column(Float)
    earnings_date_score = Column(Float, default=0)
    income_statement_score = Column(Float, default=0)
    cashflow_statement_score = Column(Float, default=0)
    balance_sheet_score = Column(Float, default=0)
    rate_scoring = Column(Float, default=0)
    buy_point = Column(Float)
    short_point = Column(Float)
    recommended_date = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)
    status = Column(String(20), default='')
    overbuy_oversold = Column(Float)  # Scale from -100 to 100: negative = oversold, positive = overbought
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

Base.metadata.create_all(engine)

def get_thread_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = Session()
    return thread_local.session

def money_flow_index(high, low, close, volume, length=14): 
    tp = (high + low + close) / 3
    raw_money_flow = tp * volume
    pos_flow = raw_money_flow.where(tp > tp.shift(1), 0)
    neg_flow = raw_money_flow.where(tp < tp.shift(1), 0)
    pos_mf_sum = pos_flow.rolling(window=length).sum()
    negat_mf_sum = neg_flow.rolling(window=length).sum()
    money_flow_ratio = pos_mf_sum / negat_mf_sum
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi

def trend_intensity(close_series):
    avgc7 = close_series.rolling(7).mean()
    avgc65 = close_series.rolling(65).mean()
    return avgc7/avgc65

def calculate_rsi(close_series, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def classify_stock(stock_data):
    """
    Classify a stock as 'Aggressive', 'Moderate', or 'Conservative' based on various metrics
    """
    adr_value = stock_data.get('ADR', 0)
    persistence_ratio = stock_data.get('Persistence Ratio', 0)
    volatility = stock_data.get('Volatility', 0)
    trend_intensity_value = stock_data.get('Trend Intensity', 0)
    
    # ADR Classification
    if adr_value >= 5:
        adr_class = 'Aggressive'
    elif 3 <= adr_value < 5:
        adr_class = 'Moderate'
    else:
        adr_class = 'Conservative'

    # Persistence Ratio Classification
    if persistence_ratio > 2.0:
        persistence_class = 'Aggressive'
    elif 0.8 <= persistence_ratio <= 2.0:
        persistence_class = 'Moderate'
    else:
        persistence_class = 'Conservative'

    # Volatility Classification
    if volatility > 0.15:
        volatility_class = 'Aggressive'
    elif 0.06 < volatility <= 0.15:
        volatility_class = 'Moderate'
    else:
        volatility_class = 'Conservative'
    
    # Trend intensity Classification
    if trend_intensity_value > 70:
        trend_class = 'Aggressive'
    elif 40 <= trend_intensity_value <= 70:
        trend_class = 'Moderate'
    else:
        trend_class = 'Conservative'
    
    classifications = [adr_class, persistence_class, volatility_class, trend_class]
 
    if classifications.count('Aggressive') >= 2:
        return 'Aggressive'
    elif classifications.count('Moderate') >= 2:
        return 'Moderate'
    else:
        return 'Conservative'

def determine_overbuy_oversold(mfi, trend_intensity_value, persistence_ratio):
    """
    Calculate a numeric value representing overbought/oversold status on a scale from -100 to 100
    
    - Negative values: Oversold (more negative = stronger oversold)
    - Positive values: Overbought (more positive = stronger overbought)
    - Zero area: Neutral
    
    MFI is scaled and adjusted based on trend intensity and persistence ratio
    """
    # Base calculation from MFI (convert 0-100 scale to -100 to 100 scale)
    base_value = (mfi - 50) * 2
    
    # Adjustments based on trend intensity and persistence
    # For oversold conditions (negative base_value)
    if base_value < 0:
        # Strong trend + high persistence makes it more strongly oversold (buy signal)
        if trend_intensity_value > 1.05 and persistence_ratio > 1.5:
            # Amplify the signal for strong buy conditions
            adjustment = -25  # Push further negative for stronger buy signal
        else:
            adjustment = 0
    # For overbought conditions (positive base_value)
    elif base_value > 0:
        # Weak trend + low persistence makes it more strongly overbought (sell signal)
        if trend_intensity_value < 0.95 and persistence_ratio < 0.7:
            adjustment = 20  # Push further positive for stronger sell signal
        else:
            adjustment = 0
    else:
        adjustment = 0
    
    # Apply adjustment and ensure within bounds (-100 to 100)
    final_value = max(-100, min(100, base_value + adjustment))
    
    return final_value

def calculate_buy_short_points(price, volatility, adr):
    """Calculate simple buy and short points based on price, volatility and ADR"""
    buy_point = price * (1 - (volatility * 0.5))
    short_point = price * (1 + (adr * 0.01 * 0.5))
    return buy_point, short_point

def process_stock_data(symbol, stock_data, sector_data):
    """Process stock data and store in database"""
    session = get_thread_session()
    thread_name = threading.current_thread().name
    
    try:
        # Get existing entry or create new one
        entry = session.query(EquityUserHistory).filter_by(symbol=symbol).first()
        if not entry:
            entry = EquityUserHistory(symbol=symbol)
        
        # Get ADR value
        adr_value = stock_data.get('ADR', 0)
        
        # Calculate Risk Type
        risk_type = classify_stock(stock_data)
        entry.risk_type = risk_type
        
        # Get Sector from sector data
        if symbol in sector_data:
            entry.sector = sector_data[symbol]
        
        # Set Volume Spike
        entry.volume_spike = stock_data.get('Volume', '')
        
        # Set RSI (dummy calculation, should be replaced with actual RSI)
        mfi_value = float(stock_data.get('MFI', 50))
        entry.RSI = mfi_value
        
        # Set ADR
        entry.ADR = adr_value
        
        # Set long term persistence (using Persistence Ratio)
        persistence_ratio = stock_data.get('Persistence Ratio', 1.0)
        entry.long_term_persistance = persistence_ratio
        
        # Set long term divergence (using Trend Intensity)
        trend_intensity_value = stock_data.get('Trend Intensity', 1.0)
        entry.long_term_divergence = trend_intensity_value
        
        # Set overbuy_oversold status
        entry.overbuy_oversold = determine_overbuy_oversold(
            mfi_value, 
            trend_intensity_value, 
            persistence_ratio
        )
        
        # Calculate buy and short points
        price = stock_data.get('Price', 0)
        volatility = stock_data.get('Volatility', 0)
        buy_point, short_point = calculate_buy_short_points(price, volatility, adr_value)
        entry.buy_point = buy_point
        entry.short_point = short_point
        
        # Set other default scores
        # These would normally be calculated from additional data
        entry.earnings_date_score = random.uniform(1, 10)
        entry.income_statement_score = random.uniform(1, 10)
        entry.cashflow_statement_score = random.uniform(1, 10)
        entry.balance_sheet_score = random.uniform(1, 10)
        
        # Calculate rate scoring as average of component scores
        entry.rate_scoring = (entry.earnings_date_score + 
                             entry.income_statement_score + 
                             entry.cashflow_statement_score + 
                             entry.balance_sheet_score) / 4
        
        # Set dates and status
        entry.recommended_date = datetime.now()
        entry.is_active = True
        entry.status = 'active'
        entry.updatedAt = datetime.now()
        
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

def get_stock_data_from_scanner():
    """
    Read symbols from input file and generate stock data
    In a real implementation, this would run the scanner logic or read its output
    """
    # Read symbols from the same input file used in company_profile_yfinance.py
    input_file = BASE_DIR / "symbols.txt"
    
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
    
    logger.info("Getting stock data from scanner...")
    
    # Create stock dataset
    stock_data = {}
    
    # Get sectors from data_helpers
    sector_info = get_us_stocks_with_sector()
    sector_dict = {item['symbol']: item['sector'] for item in sector_info}
    
    # Process all symbols from the input file
    for symbol in symbols:
        # In production, this would call the actual scanner logic or read its output
        # For now, we'll generate sample data that matches the scanner output format
        stock_data[symbol] = {
            'Symbol': symbol,
            'Price': random.uniform(10, 500),
            'ADR': random.uniform(1, 8),
            'Volatility': random.uniform(0.03, 0.2),
            'Trend Intensity': random.uniform(30, 90),
            'Persistence Ratio': random.uniform(0.5, 3.0),
            'MFI': random.uniform(30, 70),
            'Volume': 'O O' if random.random() > 0.5 else 'O O O',
        }
    
    logger.info(f"Got data for {len(stock_data)} symbols")
    return stock_data, sector_dict, symbols

def analyze_equity_data():
    """Analyze the equity user data in the database"""
    session = Session()
    
    try:
        # Count total records
        total_records = session.query(EquityUserHistory).count()
        
        if total_records == 0:
            logger.info("No equity user history found in database")
            return
            
        logger.info("\nEquity User Analysis:")
        logger.info(f"Total equity records: {total_records}")
        
        # Count by risk type
        aggressive_count = session.query(EquityUserHistory).filter_by(risk_type="Aggressive").count()
        moderate_count = session.query(EquityUserHistory).filter_by(risk_type="Moderate").count()
        conservative_count = session.query(EquityUserHistory).filter_by(risk_type="Conservative").count()
        
        logger.info("\nRisk Distribution:")
        logger.info(f"Aggressive: {aggressive_count} ({aggressive_count/total_records*100:.1f}%)")
        logger.info(f"Moderate: {moderate_count} ({moderate_count/total_records*100:.1f}%)")
        logger.info(f"Conservative: {conservative_count} ({conservative_count/total_records*100:.1f}%)")
        
        # Analyze overbuy/oversold distribution using numeric ranges
        strong_sell_count = session.query(EquityUserHistory).filter(EquityUserHistory.overbuy_oversold > 60).count()
        sell_count = session.query(EquityUserHistory).filter(EquityUserHistory.overbuy_oversold.between(20, 60)).count()
        neutral_count = session.query(EquityUserHistory).filter(EquityUserHistory.overbuy_oversold.between(-20, 20)).count()
        buy_count = session.query(EquityUserHistory).filter(EquityUserHistory.overbuy_oversold.between(-60, -20)).count()
        strong_buy_count = session.query(EquityUserHistory).filter(EquityUserHistory.overbuy_oversold < -60).count()
        
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
    logger.info("Starting equity user history generator")
    
    try:
        # Get stock data and sectors
        stock_data, sector_data, symbols = get_stock_data_from_scanner()
        
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
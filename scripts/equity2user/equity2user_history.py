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
from models.market_data import Equity2User, YfBar1d  # Add YfBar1d import
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
DEFAULT_PERIOD = PERIOD_DAY

# Analysis periods for calculations
ADR_PERIOD = 20  # Days to use for ADR calculation
MFI_PERIOD = 14  # Period for Money Flow Index
TREND_PERIOD = 65  # Period for trend intensity calculation

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"equity_history_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # Standardized format
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

def trend_intensity(close_series):
    """Calculate Trend Intensity based on moving averages"""
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

def get_price_data(symbol, days=60):
    """Fetch real price data from YfBar1d table"""
    session = get_thread_session()
    cutoff_date = date.today() - timedelta(days=days)
    
    try:
        # Query price data from YfBar1d table
        bars = session.query(YfBar1d).filter(
            YfBar1d.symbol == symbol,
            YfBar1d.timestamp >= cutoff_date
        ).order_by(YfBar1d.timestamp.asc()).all()
        
        if not bars:
            logger.warning(f"No price data found for {symbol}")
            return None
            
        # Convert to pandas DataFrame for calculations
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
        
        return data
    
    except Exception as e:
        logger.error(f"Error fetching price data for {symbol}: {str(e)}")
        return None

def calculate_metrics(symbol, days=30):
    """Calculate all metrics for a symbol from real price data"""
    price_data = get_price_data(symbol, days=max(days, 90))  # Get more data for better calculations
    
    if price_data is None or len(price_data) < 20:  # Need minimum data for calculations
        logger.warning(f"Insufficient price data for {symbol}")
        return None
    
    try:
        # Calculate ADR (Average Daily Range)
        price_data['daily_range'] = (price_data['high'] - price_data['low']) / price_data['close'] * 100
        adr = price_data['daily_range'].tail(ADR_PERIOD).mean()
        
        # Calculate volatility (20-day)
        price_data['return'] = price_data['close'].pct_change()
        volatility = price_data['return'].tail(20).std() * (252 ** 0.5)  # Annualized
        
        # Calculate MFI (Money Flow Index)
        mfi_values = money_flow_index(
            price_data['high'], 
            price_data['low'], 
            price_data['close'], 
            price_data['volume'],
            length=MFI_PERIOD
        )
        current_mfi = mfi_values.iloc[-1] if not mfi_values.empty else 50
        
        # Calculate Trend Intensity
        trend_intensity_values = trend_intensity(price_data['close'])
        # FIX: Handle NaN values properly
        if trend_intensity_values.empty:
            current_trend = 1.0
        else:
            current_trend = trend_intensity_values.iloc[-1]
            if np.isnan(current_trend):
                current_trend = 1.0
        
        # Calculate Persistence Ratio
        try:
            # Calculate 14-day cumulative return
            cum_return_14d = price_data['return'].rolling(window=14).sum()
            
            # Calculate persistence ratio (absolute cumulative return / sum of absolute daily returns)
            abs_cum_return = cum_return_14d.abs()
            sum_abs_returns = price_data['return'].abs().rolling(window=14).sum()
            
            # Avoid division by zero
            persistence_ratio = np.where(
                sum_abs_returns > 0,
                abs_cum_return / sum_abs_returns,
                1.0  # Default value when denominator is zero
            )
            
            current_persistence = persistence_ratio[-1] if len(persistence_ratio) > 0 else 1.0
        except Exception as e:
            logger.error(f"Error calculating persistence ratio for {symbol}: {e}")
            current_persistence = 1.0  # Default value
        
        # Volume spike detection
        try:
            # Calculate average volume
            avg_volume = price_data['volume'].tail(20).mean()
            recent_volumes = price_data['volume'].tail(3)
            
            # Count how many recent days had volume spikes
            spike_days = sum(vol > avg_volume * 1.5 for vol in recent_volumes)
            
            # Format volume spike indicators
            if spike_days == 0:
                volume_spike = ''
            else:
                volume_spike = 'O ' * spike_days
        except Exception as e:
            logger.error(f"Error calculating volume spike for {symbol}: {e}")
            volume_spike = ''
        
        # Get current price
        current_price = price_data['close'].iloc[-1] if not price_data.empty else 0
        
        # Create metrics dictionary
        metrics = {
            'Symbol': symbol,
            'Price': current_price,
            'ADR': adr,
            'Volatility': volatility,
            'Trend Intensity': current_trend * 50,  # Scale to match expected range
            'Persistence Ratio': current_persistence,
            'MFI': current_mfi,
            'Volume': volume_spike,
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
        return None

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
        entry = session.query(Equity2User).filter_by(symbol=symbol).first()
        if not entry:
            entry = Equity2User(symbol=symbol)
        
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
        
        # Set RSI (using MFI as this is what we calculated)
        mfi_value = float(stock_data.get('MFI', 50))
        entry.RSI = mfi_value
        
        # Set ADR
        entry.ADR = adr_value
        
        # Set long term persistence (using Persistence Ratio)
        persistence_ratio = stock_data.get('Persistence Ratio', 1.0)
        entry.long_term_persistance = persistence_ratio
        
        # Set long term divergence (using Trend Intensity)
        trend_intensity_value = stock_data.get('Trend Intensity', 1.0)
        # FIX: Handle NaN values in trend_intensity_value
        if trend_intensity_value is None or (isinstance(trend_intensity_value, float) and np.isnan(trend_intensity_value)):
            trend_intensity_value = 1.0  # Default value for NaN
        entry.long_term_divergence = trend_intensity_value
        
        # Set overbuy_oversold status
        entry.overbuy_oversold = determine_overbuy_oversold(
            mfi_value, 
            trend_intensity_value / 50 if not np.isnan(trend_intensity_value) else 1.0,  # Also handle NaN here
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
        entry.updated_at = datetime.now()
        
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

def get_sector_data(symbols):
    """
    Create a simple sector dictionary from symbol prefixes
    In a real implementation, this would use a proper sector database
    """
    sector_dict = {}
    
    # Simple mapping of first letter to sectors for demonstration
    sectors = {
        "A": "Technology",
        "B": "Financials",
        "C": "Healthcare",
        "D": "Consumer Cyclical",
        "E": "Energy",
        "F": "Utilities",
        "G": "Basic Materials",
        "H": "Real Estate",
        "I": "Communication Services",
        "J": "Industrials",
        "K": "Consumer Defensive",
        "L": "Technology",
        "M": "Financials",
        "N": "Healthcare",
        "O": "Energy",
        "P": "Consumer Cyclical",
        "Q": "Communication Services",
        "R": "Industrials",
        "S": "Consumer Defensive",
        "T": "Technology",
        "U": "Utilities",
        "V": "Consumer Cyclical",
        "W": "Industrials",
        "X": "Basic Materials",
        "Y": "Communication Services",
        "Z": "Technology"
    }
    
    # Assign sectors based on first letter
    for symbol in symbols:
        first_letter = symbol[0].upper() if symbol else "A"
        sector_dict[symbol] = sectors.get(first_letter, "Unknown")
    
    return sector_dict

def get_stock_metrics(symbols, days=30):
    """
    Calculate real metrics for symbols using YfBar1d data
    """
    logger.info(f"Calculating metrics for {len(symbols)} symbols...")
    
    stock_data = {}
    calculated = 0
    failed = 0
    
    for symbol in symbols:
        metrics = calculate_metrics(symbol, days)
        if metrics:
            stock_data[symbol] = metrics
            calculated += 1
        else:
            failed += 1
        
        # Log progress
        if (calculated + failed) % 20 == 0 or (calculated + failed) == len(symbols):
            logger.info(f"Metrics progress: {calculated + failed}/{len(symbols)} - Success: {calculated}, Failed: {failed}")
    
    logger.info(f"Finished calculating metrics for {calculated} symbols")
    return stock_data, get_sector_data(symbols)

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
    
    session = Session()
    
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

# Initialize global variables
engine = None
Session = None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Equity User History Generator')
    parser = add_environment_args(parser)
    parser.add_argument('--period', type=str, 
                      choices=['last_day', 'last_month'],
                      default='last_day',
                      help='Period for history generation (default: last_day)')
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
    else:
        history_days = DEFAULT_PERIOD
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting equity user history generator using {env_name} database")
    logger.info(f"Processing period: {args.period} ({history_days} days)")
    
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
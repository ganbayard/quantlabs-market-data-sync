import os
import argparse
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import yfinance as yf
import logging
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from cachetools import TTLCache
import concurrent.futures

# Cache for data to avoid redundant API calls
data_cache = {}
cache_lock = threading.Lock()

class YahooFinanceLoader:
    """Yahoo Finance data loader implementation for price data"""
    
    def __init__(self, batch_size=5, worker_count=2):
        """
        Yahoo Finance data loader with robust download handling
        
        Args:
            batch_size: Number of symbols to process in a batch (default: 5)
            worker_count: Number of concurrent download workers (default: 2)
        """
        self.batch_size = batch_size
        self.worker_count = worker_count
        self.logger = logging.getLogger(__name__)
    
    def load_symbol_data(self, symbol, start_date, end_date):
        """
        Load historical data for a single symbol
        
        Args:
            symbol: Stock symbol to download
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical data or None if download failed
        """
        return self._download_with_retry(symbol, start_date, end_date)
    
    def _download_with_retry(self, symbol, start_date, end_date, max_retries=5, initial_delay=2):
        """Download stock data with exponential backoff retry logic"""
        delay = initial_delay
        attempt = 0
        rate_limited = False
        
        while attempt < max_retries:
            attempt += 1
            try:
                # If we've been rate limited before, wait longer
                if rate_limited:
                    self.logger.info(f"Waiting {delay:.2f}s before retry {attempt} for {symbol}...")
                    time.sleep(delay)
                    rate_limited = False
                
                self.logger.info(f"Downloading {symbol} (attempt {attempt})...")
                ticker = yf.Ticker(symbol)
                
                # Check if ticker exists
                check_df = ticker.history(period='1d')
                if check_df.empty:
                    self.logger.info(f"{symbol} has no data available")
                    return None
                
                # Get historical data with explicit date range
                history_df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                
                if history_df.empty:
                    self.logger.info(f"{symbol} returned empty dataframe")
                    return None
                
                # Check for splits and dividends
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
                            self.logger.info(f"{symbol} has recent splits")
                    
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
                            self.logger.info(f"{symbol} has recent dividends")
                    
                    # If split or dividend occurred, fetch full history
                    if has_recent_event:
                        self.logger.info(f"Split or dividend happened to {symbol} so fetching all history again...")
                        # Always use explicit date range
                        all_history_df = ticker.history(start="2000-01-01", end=end_date, auto_adjust=True)
                        if not all_history_df.empty:
                            history_df = all_history_df
                except Exception as e:
                    self.logger.warning(f"{symbol}: Error checking splits/dividends: {e}")
                
                # Add jitter to avoid synchronized requests hitting rate limits
                time.sleep(random.uniform(0.2, 0.5))
                return history_df
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Special handling for the 'period max is invalid' error
                if "period 'max' is invalid" in error_msg:
                    self.logger.warning(f"{symbol}: {e} - Using explicit date range instead")
                    # Try again with explicit date range instead of 'max'
                    try:
                        ticker = yf.Ticker(symbol)
                        history_df = ticker.history(start=datetime(2000, 1, 1), end=end_date, auto_adjust=True)
                        return history_df
                    except Exception as retry_error:
                        self.logger.error(f"{symbol}: Error on retry with explicit dates: {retry_error}")
                
                # Handle rate limiting specifically
                elif "rate limit" in error_msg or "too many requests" in error_msg:
                    rate_limited = True
                    self.logger.warning(f"Rate limit hit for {symbol}. Will retry with backoff.")
                    
                    # Exponential backoff with jitter
                    delay = initial_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    
                    # Cap the delay at a reasonable maximum (30 seconds)
                    delay = min(delay, 30)
                else:
                    self.logger.error(f"Error downloading {symbol}: {e}")
                    # For non-rate-limit errors, use a shorter delay
                    time.sleep(1)
        
        self.logger.error(f"Failed to download {symbol} after {max_retries} attempts")
        return None
        
    def load_multiple_symbols(self, symbols, start_date, end_date):
        """
        Load data for multiple symbols with concurrency
        
        Args:
            symbols: List of stock symbols to download
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            Dictionary mapping symbols to their dataframes
        """
        results = {}
        
        # Process in small batches to avoid rate limits
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i:i+self.batch_size]
            batch_num = i//self.batch_size + 1
            total_batches = (len(symbols) + self.batch_size - 1)//self.batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} - Symbols: {', '.join(batch)}")
            
            # Process each symbol in the batch with concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_count) as executor:
                future_to_symbol = {
                    executor.submit(self._download_with_retry, symbol, start_date, end_date): symbol 
                    for symbol in batch
                }
                
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    
                    try:
                        result = future.result()
                        results[symbol] = result
                    except Exception as e:
                        self.logger.error(f"Exception processing {symbol}: {e}")
            
            # Add delay between batches to prevent rate limiting
            if i + self.batch_size < len(symbols):
                wait_time = random.uniform(3, 5)
                self.logger.info(f"Waiting {wait_time:.2f}s before next batch...")
                time.sleep(wait_time)
                
        return results


class NorgateDataLoader:
    """Norgate data loader implementation for price data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = TTLCache(maxsize=10000, ttl=3600)
        self.cache_lock = threading.Lock()
        
        # Load symbols from file
        try:
            symbols_file = Path(__file__).parent / "symbols/stock_symbols.txt"
            with open(symbols_file, 'r') as f:
                content = f.read()
                self.symbols = [symbol.strip() for symbol in content.split(',') if symbol.strip()]
            self.logger.info(f"Loaded {len(self.symbols)} symbols for Norgate data")
        except Exception as e:
            self.logger.error(f"Error loading symbols for Norgate: {e}")
            self.symbols = []
    
    def load_symbol_data(self, symbol, start_date, end_date):
        """
        Method required by daily bar loader script
        This bridges to our fetch_symbol_data implementation
        """
        try:
            import norgatedata
            
            if not start_date:
                start_date = datetime.now() - timedelta(days=365)
            if not end_date:
                end_date = datetime.now()
            
            cache_key = (symbol, start_date, end_date)
            
            # Check if data is in cache
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            # Fetch data from Norgate
            df = norgatedata.price_timeseries(
                symbol,
                stock_price_adjustment_setting = norgatedata.StockPriceAdjustmentType.TOTALRETURN,
                padding_setting                = norgatedata.PaddingType.NONE,
                start_date                     = start_date,
                end_date                       = end_date,
                timeseriesformat               = 'pandas-dataframe',
            )
            
            if df is not None and not df.empty:
                # Ensure standard column naming
                column_map = {
                    'open': 'Open', 
                    'high': 'High', 
                    'low': 'Low', 
                    'close': 'Close', 
                    'volume': 'Volume'
                }
                
                df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
                
                # Cache the result
                with self.cache_lock:
                    self.cache[cache_key] = df
                
                return df
            else:
                self.logger.warning(f"No data found for {symbol} in Norgate")
                return None
                
        except ImportError:
            self.logger.error("Norgate Data SDK not installed. Please install the Norgate Data SDK.")
            return None
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol} from Norgate: {e}")
            return None
    
    def load_data(self, start_date=None, end_date=None, symbols=None):
        """
        Method required by norgate batch loader script
        Load price data for multiple symbols
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            symbols (list): List of symbols to fetch; uses self.symbols if None
            
        Returns:
            dict: Dictionary mapping symbols to their price DataFrame
        """
        start_time = time.perf_counter()
        self.logger.info("Loading stock data from Norgate...")
        
        if symbols is None:
            symbols = self.symbols
        
        # Dictionary to hold all data
        assets = {}
        
        # Use ThreadPoolExecutor for concurrent fetching
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.load_symbol_data, symbol, start_date, end_date): symbol 
                for symbol in symbols
            }
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        assets[symbol] = df
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        self.logger.info(f"Norgate data loading completed in {elapsed_time:.2f} seconds")
        
        return assets

def get_connection_details(env='dev'):
    """
    Get database connection details based on environment
    Args:
        env (str): Environment name ('dev' or 'prod')
    Returns:
        dict: Dictionary with connection parameters
    """
    load_dotenv()
    
    if env == 'prod':
        # Production database connection
        return {
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT', '3306'),
            'database': os.getenv('DB_NAME')
        }
    else:
        # Development database connection (Docker)
        return {
            'user': os.getenv('LOCAL_DB_USER'),
            'password': os.getenv('LOCAL_DB_PASSWORD'),
            'host': os.getenv('LOCAL_DB_HOST', 'localhost'),
            'port': os.getenv('LOCAL_DB_PORT', '3306'),
            'database': os.getenv('LOCAL_DB_NAME')
        }

def get_database_engine(env="dev"):
    """Get SQLAlchemy engine based on environment"""
    load_dotenv()
    
    if env == "prod":
        # Production database connection
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        db_host = os.getenv('DB_HOST')
        db_port = os.getenv('DB_PORT', '3306')
        db_name = os.getenv('DB_NAME')
    else:
        # Development database connection (Docker)
        db_user = os.getenv('LOCAL_DB_USER')
        db_password = os.getenv('LOCAL_DB_PASSWORD')
        db_host = os.getenv('LOCAL_DB_HOST', 'localhost')
        db_port = os.getenv('LOCAL_DB_PORT', '3306')
        db_name = os.getenv('LOCAL_DB_NAME')
    
    # Create engine with increased pool size
    connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # INCREASE THESE VALUES for higher concurrency
    engine = create_engine(
        connection_string, 
        pool_size=30,           # Increase from 10 to 30
        max_overflow=20,        # Increase from 10 to 20
        pool_recycle=3600,
        pool_pre_ping=True      # Add this for connection health checks
    )
    
    return engine

def get_session_maker(env='dev'):
    """
    Create a SQLAlchemy session maker based on environment
    Args:
        env (str): Environment name ('dev' or 'prod')
    Returns:
        sessionmaker: SQLAlchemy session maker
    """
    engine = get_database_engine(env)
    return sessionmaker(bind=engine)

def add_environment_args(parser):
    """
    Add environment selection arguments to an ArgumentParser
    Args:
        parser (ArgumentParser): ArgumentParser instance
    Returns:
        ArgumentParser: The updated parser
    """
    parser.add_argument('--env', choices=['dev', 'prod'], default='dev',
                        help='Target environment (dev or prod)')
    return parser
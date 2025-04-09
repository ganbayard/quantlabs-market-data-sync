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

# Cache for data to avoid redundant API calls
data_cache = {}
cache_lock = threading.Lock()

class YahooFinanceLoader:
    """Yahoo Finance data loader implementation for price data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_symbol_data(self, symbol, start_date, end_date):
        """
        Load price data for a specific symbol within date range
        
        Args:
            symbol (str): Stock symbol
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            DataFrame: DataFrame with OHLCV data or None on failure
        """
        # Generate cache key
        cache_key = f"{symbol}_{start_date}_{end_date}_yahoo"
        
        # Check cache
        with cache_lock:
            if cache_key in data_cache:
                return data_cache[cache_key]
                
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            # Normalize column names
            if not df.empty:
                df.columns = [col.title() for col in df.columns]  # Capitalize first letter
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                # Check if required columns exist
                if all(col in df.columns for col in required_columns):
                    # Only keep essential columns
                    df = df[required_columns]
                    
                    # Store in cache
                    with cache_lock:
                        data_cache[cache_key] = df
                    
                    return df
            
            self.logger.warning(f"No data found for {symbol} in Yahoo Finance")
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol} from Yahoo Finance: {e}")
            return None
    
    def load_multiple_symbols(self, symbols, start_date, end_date, max_workers=4):
        """
        Load price data for multiple symbols with threading
        
        Args:
            symbols (list): List of stock symbols
            start_date (datetime): Start date
            end_date (datetime): End date
            max_workers (int): Maximum number of concurrent workers
            
        Returns:
            dict: Dictionary mapping symbols to their price DataFrames
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for each symbol
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
                        results[symbol] = df
                    
                    # Add small delay to avoid rate limiting
                    time.sleep(random.uniform(0.1, 0.3))
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
        
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
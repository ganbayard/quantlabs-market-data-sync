import sys
import time
import os
import pandas as pd
import yfinance as yf
import sqlalchemy as sa
import concurrent.futures
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import insert as mysql_insert
from dotenv import load_dotenv
import logging

from models.market_data import YfBar1d

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# MySQL database configuration
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "stock_data")

# Create MySQL connection string
MYSQL_CONNECTION_STRING = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Create engine
engine = sa.create_engine(MYSQL_CONNECTION_STRING)


def initialize_database():
    """Create database and tables if they don't exist"""
    try:
        # Create database if it doesn't exist
        engine_no_db = sa.create_engine(
            f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"
        )
        with engine_no_db.connect() as conn:
            conn.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")

        # Create tables
        Base.metadata.create_all(engine)
        logging.info("Database and tables initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise


def load_symbols():
    """Load symbols from config files"""
    symbols = []

    # # Load symbols from DATABASE_UPDATE_SYMBOLS
    # try:
    #     with open(os.getenv("DATABASE_UPDATE_SYMBOLS", "symbols.txt")) as f:
    #         content = f.read()
    #         symbols = content.strip().split(",")
    #         logging.info(f"symbols {len(symbols)}")
    # except FileNotFoundError:
    #     logging.warning("DATABASE_UPDATE_SYMBOLS file not found")

    # Load ETF symbols
    try:
        with open(os.getenv("DATABASE_UPDATE_ETF_SYMBOLS", "etf_symbols.txt")) as f:
            content = f.read()
            etf_symbols = [
                symbol.strip()
                for symbol in content.strip().split(",")
                if symbol.strip()
            ]
            symbols.extend(etf_symbols)
    except FileNotFoundError:
        logging.warning("DATABASE_UPDATE_ETF_SYMBOLS file not found")

    # Remove duplicates
    symbols = list(dict.fromkeys(symbols))
    return symbols


def download_latest_stock(symbol):
    """Download latest stock data for a given symbol"""
    logging.info(f"Downloading {symbol} ...")
    try:
        ticker = yf.Ticker(symbol)
        if ticker.history(period="1d").empty:
            logging.warning(f"No history found for {symbol}")
            return None

        end_date = datetime.today()
        start_date = end_date - timedelta(days=14)

        # Fetch history
        history_df = ticker.history(
            start=start_date, end=end_date, auto_adjust=True, period="1d"
        )

        # Check for splits and dividends
        splits = ticker.splits
        dividends = ticker.dividends
        recent_splits = splits[
            (splits.index.date >= start_date.date())
            & (splits.index.date <= end_date.date())
        ]
        recent_dividends = dividends[
            (dividends.index.date >= start_date.date())
            & (dividends.index.date <= end_date.date())
        ]

        # If splits or dividends occurred, fetch full history
        if not recent_splits.empty or not recent_dividends.empty:
            logging.info(
                f"Split or dividend happened to {symbol}. Fetching full history..."
            )
            history_df = ticker.history(
                start="2000-01-01", auto_adjust=True, period="1d"
            )

        return history_df

    except Exception as e:
        logging.error(f"Exception downloading {symbol}: {e}")
        return None


def insert_update_df_to_db(symbol, df, batch_size=300):
    """Insert or update stock data in the database"""
    for i in range(0, len(df), batch_size):
        batch_start_index = i
        batch_end_index = min(i + batch_size, len(df))

        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Prepare rows for batch insert
            rows_to_insert = [
                {
                    "symbol": symbol,
                    "timestamp": timestamp.date(),
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"],
                }
                for timestamp, row in df.iloc[
                    batch_start_index:batch_end_index
                ].iterrows()
            ]

            # MySQL-specific upsert
            for row in rows_to_insert:
                insert_stmt = mysql_insert(YfBar1d).values(row)
                upsert_stmt = insert_stmt.on_duplicate_key_update(
                    open=insert_stmt.inserted.open,
                    high=insert_stmt.inserted.high,
                    low=insert_stmt.inserted.low,
                    close=insert_stmt.inserted.close,
                    volume=insert_stmt.inserted.volume,
                )
                session.execute(upsert_stmt)

            session.commit()
            logging.info(f"{symbol} batch insert {batch_start_index}:{batch_end_index}")

        except Exception as e:
            session.rollback()
            logging.error(f"Error inserting data for {symbol}: {e}")

        finally:
            session.close()


def main():
    # Initialize database
    # initialize_database()

    # Load symbols
    symbols = load_symbols()
    logging.info(f"Total symbols to process: {len(symbols)}")

    # Concurrent download
    max_workers = min(6, len(symbols))
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_latest_stock, symbols))

    # Create mapping of symbols to results
    assets = {
        symbol: result for symbol, result in zip(symbols, results) if result is not None
    }

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logging.info(f"Fetching completed in {elapsed_time:.2f} seconds.")

    # Save to database
    logging.info("Saving data to database...")

    for symbol, df in assets.items():
        try:
            insert_update_df_to_db(symbol, df)
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")

    logging.info("Data update completed.")


if __name__ == "__main__":
    main()

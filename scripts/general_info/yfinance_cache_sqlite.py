import os
import sys
import time
import logging
import pandas as pd
import sqlalchemy as sa
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy.dialects.mysql import insert as mysql_insert
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


def download_stock_data(ticker, start_date):
    try:
        stock = yf.Ticker(ticker)
        if not stock.history(period="1d").empty:
            data = stock.history(start=start_date, auto_adjust=True)
            return data[["Open", "High", "Low", "Close", "Volume"]]
        else:
            return None
    except Exception as e:
        return None


def insert_df_to_db(symbol, df, batch_size=1000):
    df = df[:-1]  # remove unclosed bar
    for i in range(0, len(df), batch_size):
        batch_start_index = i
        batch_end_index = min(i + batch_size, len(df))
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            with session:
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
            logging.error(f"Error inserting data for {symbol}: {e}")
            session.rollback()
        finally:
            session.close()


# symbols = symbols[:5]
# print(symbols)
def main():
    start_time = time.perf_counter()
    symbols = load_symbols()

    assets = {}
    for idx, symbol in enumerate(symbols):
        df = download_stock_data(symbol, "2000-01-01")
        if df is None:
            logging.warning(f"{symbol} doesn't exist.")
        else:
            logging.info(f"{symbol} is fetched into memory.")
            assets[symbol] = df

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logging.info(f"fetching is done. {elapsed_time:.2f} seconds.")

    logging.info("saving df into the DB...")

    for symbol in assets:
        df = assets[symbol]
        insert_df_to_db(symbol, df)

    logging.info("FIN.")


if __name__ == "__main__":
    main()

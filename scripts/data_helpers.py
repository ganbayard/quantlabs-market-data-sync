import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import or_
from sqlalchemy.sql import func
from models.market_data import (
    IShareETF,
    SymbolFields,
)
from models.market_data import MmtvDailyBar
from models.market_data import YfBar1d
import os
from dotenv import load_dotenv

# Load environment variables for database configuration
load_dotenv()

# MySQL database connection configuration
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "etf_data")

# Create MySQL connection string
MYSQL_CONNECTION_STRING = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Create engine
engine = create_engine(MYSQL_CONNECTION_STRING)

# Create session factory
Session = sessionmaker(bind=engine)


def get_us_stocks_with_sector():
    Session = sessionmaker(bind=engine)
    session = Session()
    """
    Retrieve symbol, sector, and industry for US stocks from the SymbolFields table.
    
    :param session: SQLAlchemy session
    :return: List of dictionaries containing symbol, sector, and industry
    """
    query = sa.select(
        SymbolFields.symbol,
        SymbolFields.sector,
        SymbolFields.industry,
        SymbolFields.indexes,
    )

    result = session.execute(query)

    us_stocks_sector = [
        {
            "symbol": row.symbol,
            "sector": row.sector,
            "industry": row.industry,
            "indexes": row.indexes,
        }
        for row in result
    ]

    return us_stocks_sector


def get_mmtv_bar_with_sector():
    Session = sessionmaker(bind=engine)
    session = Session()
    """
    Retrieve symbol, sector, and industry for US stocks from the sector_update table.
    Also includes records where field_type is 'total_market'.
    
    :param session: SQLAlchemy session
    :return: List of dictionaries containing date, field_name, field_type, open, high, low, close
    """
    query = sa.select(
        MmtvDailyBar.date,
        MmtvDailyBar.field_name,
        MmtvDailyBar.field_type,
        MmtvDailyBar.open,
        MmtvDailyBar.high,
        MmtvDailyBar.low,
        MmtvDailyBar.close,
    ).where(
        or_(
            MmtvDailyBar.field_type == "sector",
            MmtvDailyBar.field_type == "total_market",
        )
    )

    result = session.execute(query)

    mmtv_bar_sector = [
        {
            "Date": row.date,
            "Field_name": row.field_name,
            "Field_type": row.field_type,
            "Open": row.open,
            "High": row.high,
            "Low": row.low,
            "Close": row.close,
            "Volume": np.nan,
        }
        for row in result
    ]

    return mmtv_bar_sector


def get_mmtv_bar_with_industry():
    Session = sessionmaker(bind=engine)
    session = Session()
    """
    Retrieve symbol, sector, and industry for US stocks from the sector_update table.
    Also includes records where field_type is 'total_market'.
    
    :param session: SQLAlchemy session
    :return: List of dictionaries containing date, field_name, field_type, open, high, low, close
    """
    query = sa.select(
        MmtvDailyBar.date,
        MmtvDailyBar.field_name,
        MmtvDailyBar.field_type,
        MmtvDailyBar.open,
        MmtvDailyBar.high,
        MmtvDailyBar.low,
        MmtvDailyBar.close,
    ).where(MmtvDailyBar.field_type == "industry")

    result = session.execute(query)

    mmtv_bar_sector = [
        {
            "Date": row.date,
            "Field_name": row.field_name,
            "Field_type": row.field_type,
            "Open": row.open,
            "High": row.high,
            "Low": row.low,
            "Close": row.close,
            "Volume": np.nan,
        }
        for row in result
    ]

    return mmtv_bar_sector


def get_ishare_etf_symbols():
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        etf_symbols = [etf.ticker for etf in session.query(IShareETF.ticker).all()]
        return etf_symbols
    finally:
        session.close()


def get_ishare_symbols_by_etf(etf_symbol):
    """Get holdings for a specific ETF with the latest date"""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        query = sa.text("""
            SELECT ticker, weight 
            FROM ishare_etf_holding 
            WHERE fund_ticker = :etf_symbol
            AND as_of_date = (
                SELECT MAX(as_of_date) 
                FROM ishare_etf_holding 
                WHERE fund_ticker = :etf_symbol
            )
        """)

        result = session.execute(query, {"etf_symbol": etf_symbol})
        holdings = []
        for row in result:
            if row.ticker and row.weight is not None:
                holdings.append(
                    {"ticker": str(row.ticker).strip(), "weight": float(row.weight)}
                )
        return holdings
    except Exception as e:
        print(f"Error querying holdings for {etf_symbol}: {str(e)}")
        return None
    finally:
        session.close()


# def get_ishare_symbols_by_etf(etf_symbol):
#     """
#     Get holdings for an iShares ETF using direct SQL query to match database structure.
#     """
#     Session = sessionmaker(bind=engine)
#     session = Session()
#     try:
#         query = sa.text("""
#             SELECT ticker, weight
#             FROM ishare_etf_holding
#             WHERE fund_ticker = :etf_symbol
#             AND as_of_date = (
#                 SELECT MAX(as_of_date)
#                 FROM ishare_etf_holding
#                 WHERE fund_ticker = :etf_symbol
#             )
#         """)

#         result = session.execute(query, {"etf_symbol": etf_symbol})

#         symbols = []
#         for row in result:
#             # Ensure both ticker and weight are present
#             if row.ticker and row.weight is not None:
#                 symbols.append({
#                     'ticker': str(row.ticker).strip(),
#                     'weight': float(row.weight)
#                 })

#         if not symbols:
#             print(f"No holdings found for ETF: {etf_symbol}")
#             return None

#         return symbols

#     except Exception as e:
#         print(f"Error querying iShares ETF data: {str(e)}")
#         return None
#     finally:
#         session.close()

# def get_ishare_symbols_by_etf(etf_symbol):
#     Session = sessionmaker(bind=engine)
#     session = Session()
#     with session:
#         etf_instance = session.query(IShareETF).filter_by(ticker=etf_symbol).first()
#         if etf_instance:
#             symbols = []
#             for record in session.query(IShareETFHolding).filter_by(ishare_etf_id=etf_instance.id).all():
#                 symbols.append({'ticker': record.ticker, 'weight': record.weight})
#             return symbols
#         else:
#             return None


def get_ath_per_symbol():
    Session = sessionmaker(bind=engine)
    session = Session()

    query = sa.select(YfBar1d.symbol, func.max(YfBar1d.high).label("ATH")).group_by(
        YfBar1d.symbol
    )

    result = session.execute(query)

    ath_data = pd.DataFrame([{"Symbol": row.symbol, "ATH": row.ATH} for row in result])

    session.close()
    return ath_data

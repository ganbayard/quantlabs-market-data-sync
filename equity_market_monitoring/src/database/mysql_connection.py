import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import streamlit as st
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_mysql_connection(use_ssh=False):
    """
    Create a database connection to MySQL using environment variables

    Args:
        use_ssh: Whether to use SSH tunneling

    Returns:
        tuple: (engine, session)
    """
    try:
        # Determine environment based on env var or default to dev
        env = os.environ.get("MIGRATION_ENV", "dev")

        # Use appropriate connection details based on environment
        if env == "prod":
            db_host = os.environ.get("DB_HOST")
            db_port = os.environ.get("DB_PORT", "3306")
            db_user = os.environ.get("DB_USER")
            db_password = os.environ.get("DB_PASSWORD")
            db_name = os.environ.get("DB_NAME")
        else:  # dev environment
            db_host = os.environ.get("LOCAL_DB_HOST")
            db_port = os.environ.get("LOCAL_DB_PORT", "3306")
            db_user = os.environ.get("LOCAL_DB_USER")
            db_password = os.environ.get("LOCAL_DB_PASSWORD")
            db_name = os.environ.get("LOCAL_DB_NAME")

        # Create connection string
        conn_str = (
            f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        # Create engine and session
        engine = create_engine(conn_str)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Test connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            for row in result:
                logger.info(f"Connection successful: {row}")

        logger.info(f"Successfully connected to {env.upper()} MySQL database")
        return engine, session

    except Exception as e:
        logger.error(f"Error connecting to MySQL database: {e}")
        st.error(f"Database connection error: {e}")
        return None, None


def load_risk_profile_data(session, include_stocks=True, include_etfs=True):
    """
    Load risk profile data from database

    Args:
        session: Database session
        include_stocks: Whether to include stocks
        include_etfs: Whether to include ETFs

    Returns:
        DataFrame with risk profile data
    """
    try:
        conditions = []
        if include_stocks:
            conditions.append("sf.is_etf = 0")
        if include_etfs:
            conditions.append("sf.is_etf = 1")

        if not conditions:
            logger.warning("No asset types selected (stocks or ETFs)")
            return pd.DataFrame()

        where_clause = " OR ".join(conditions)

        query = f"""
        SELECT 
            sf.symbol, 
            sf.is_etf, 
            sf.company_name,
            sf.price,
            sf.sector,
            sf.industry,
            sf.asset_class,
            sf.focus,
            sf.market_cap,
            sf.exchange,
            sf.country,
            erp.risk_type,
            erp.average_score,
            erp.volatility_3yr_avg,
            erp.beta_3yr_avg,
            erp.max_drawdown_3yr_avg,
            erp.adr_3yr_avg,
            erp.volatility_score,
            erp.beta_score,
            erp.drawdown_score,
            erp.adr_score,
            erp.volatility_year3,
            erp.volatility_year2,
            erp.volatility_year1,
            erp.beta_year3,
            erp.beta_year2,
            erp.beta_year1,
            erp.max_drawdown_year3,
            erp.max_drawdown_year2,
            erp.max_drawdown_year1,
            erp.adr_year3,
            erp.adr_year2,
            erp.adr_year1,
            erp.classified_at
        FROM 
            symbol_fields sf
        JOIN 
            equity_risk_profile erp ON sf.symbol = erp.symbol
        WHERE 
            ({where_clause})
        """

        df = pd.read_sql(text(query), session.bind)

        if df.empty:
            logger.warning("No risk profile data found")
            return pd.DataFrame()

        # Convert is_etf to boolean for easier filtering
        df["is_etf"] = df["is_etf"] == 1
        df["asset_type"] = df["is_etf"].apply(lambda x: "ETF" if x else "Stock")

        # Fill missing values for better visualization
        df["focus"] = df["focus"].fillna("Not Specified")
        df["asset_class"] = df["asset_class"].fillna("Not Specified")
        df["sector"] = df["sector"].fillna("Not Specified")
        df["industry"] = df["industry"].fillna("Not Specified")

        logger.info(
            f"Loaded {len(df)} risk profiles ({df['is_etf'].sum()} ETFs, {(~df['is_etf']).sum()} Stocks)"
        )
        return df

    except Exception as e:
        logger.error(f"Error loading risk profile data: {e}")
        st.error(f"Error loading risk profile data: {e}")
        return pd.DataFrame()


def get_risk_distribution_metrics(risk_data):
    """
    Calculate risk distribution metrics from risk data

    Args:
        risk_data: DataFrame with risk profile data

    Returns:
        Tuple of (total_symbols, etf_count, stock_count, risk_distribution)
    """
    total_symbols = len(risk_data)
    etf_count = risk_data["is_etf"].sum()
    stock_count = total_symbols - etf_count

    # Calculate risk distributions
    risk_distribution = risk_data["risk_type"].value_counts(normalize=True)

    return total_symbols, etf_count, stock_count, risk_distribution


def load_price_data(session, symbol, lookback_days=1500):
    """
    Load price data for a specific symbol

    Args:
        session: Database session
        symbol: The ticker symbol
        lookback_days: Number of days to look back

    Returns:
        DataFrame with price data
    """
    try:
        from sqlalchemy import text
        import pandas as pd
        from datetime import datetime, timedelta
        import logging
        import streamlit as st

        logger = logging.getLogger(__name__)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        query = """
        SELECT 
            timestamp, 
            open, 
            high, 
            low, 
            close, 
            volume 
        FROM 
            yf_daily_bar
        WHERE 
            symbol = :symbol
            AND timestamp BETWEEN :start_date AND :end_date
        ORDER BY 
            timestamp
        """

        df = pd.read_sql(
            text(query),
            session.bind,
            params={
                "symbol": symbol,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            },
        )

        if df.empty:
            logger.warning(f"No price data found for {symbol}")
            return pd.DataFrame()

        # Convert columns
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])

        # Calculate moving averages
        df["MA21"] = df["close"].rolling(window=21).mean()
        df["MA200"] = df["close"].rolling(window=200).mean()

        # Calculate daily returns
        df["daily_return"] = df["close"].pct_change()

        # Calculate drawdown
        df["high_watermark"] = df["close"].cummax()
        df["drawdown"] = (df["close"] - df["high_watermark"]) / df["high_watermark"]

        logger.info(f"Loaded {len(df)} price records for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error loading price data for {symbol}: {e}")
        st.error(f"Error loading price data for {symbol}: {e}")
        return pd.DataFrame()


def load_etf_holdings(session, etf_symbol):
    """
    Load ETF holdings data

    Args:
        session: Database session
        etf_symbol: The ETF symbol

    Returns:
        DataFrame with ETF holdings data
    """
    try:
        from sqlalchemy import text
        import pandas as pd
        import logging
        import streamlit as st

        logger = logging.getLogger(__name__)

        query = """
        SELECT 
            ih.ticker as stock_symbol,
            ih.name,
            ih.sector,
            ih.weight,
            ih.market_value,
            ih.price,
            erp.max_drawdown_3yr_avg as drawdown,
            erp.risk_type
        FROM 
            ishare_etf e
        JOIN 
            ishare_etf_holding ih ON e.id = ih.ishare_etf_id
        LEFT JOIN 
            equity_risk_profile erp ON ih.ticker = erp.symbol
        WHERE 
            e.ticker = :etf_symbol
        ORDER BY 
            ih.weight DESC
        """

        df = pd.read_sql(text(query), session.bind, params={"etf_symbol": etf_symbol})

        if df.empty:
            logger.warning(f"No holdings data found for ETF {etf_symbol}")
            return pd.DataFrame()

        # Convert weight to percentage if it's in decimal form
        if df["weight"].max() < 1:
            df["weight"] = df["weight"] * 100

        logger.info(f"Loaded {len(df)} holdings for ETF {etf_symbol}")
        return df

    except Exception as e:
        logger.error(f"Error loading holdings data for ETF {etf_symbol}: {e}")
        st.error(f"Error loading holdings data for ETF {etf_symbol}: {e}")
        return pd.DataFrame()

import pandas as pd
import numpy as np
import streamlit as st
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from sqlalchemy import text

# Set up logging
logger = logging.getLogger(__name__)


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


def load_price_data(session, symbol, lookback_days=365 * 2):
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
        df["MA50"] = df["close"].rolling(window=50).mean()
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
        st.error(f"Error loading price data: {e}")
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

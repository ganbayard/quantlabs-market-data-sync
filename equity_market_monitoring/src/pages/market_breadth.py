import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import logging

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import database connection
from src.database.mysql_connection import get_mysql_connection
from src.database.models import SymbolFields, YfBar1d

# Define risk color scheme for consistent styling
RISK_COLORS = {
    "Conservative": "#4CAF50",  # Green
    "Moderate": "#FFC107",  # Deep golden yellow (amber)
    "Aggressive": "#F44336",  # Red
}


def apply_chart_theme(fig):
    """Apply consistent theme to all charts"""
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="#FFFFFF"),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
        ),
    )
    return fig


# Update the create_ohlc_chart function to support reference lines and scaling
def create_ohlc_chart(data, title, add_ma=False, normalize=False, add_ref_lines=False):
    """Create OHLC candlestick chart"""
    fig = go.Figure()

    # Normalize data if requested
    plot_data = data.copy()
    if normalize:
        # Rescale OHLC to 0-100 range
        min_val = min(plot_data["low"].min(), plot_data["close"].min())
        max_val = max(plot_data["high"].max(), plot_data["close"].max())
        range_val = max_val - min_val

        for col in ["open", "high", "low", "close"]:
            plot_data[col] = (
                100 * (plot_data[col] - min_val) / range_val if range_val > 0 else 50
            )
    else:
        plot_data = data

    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=plot_data["date"],
            open=plot_data["open"],
            high=plot_data["high"],
            low=plot_data["low"],
            close=plot_data["close"],
            name="OHLC",
            increasing_line_color="#26A69A",  # Green for increasing candles
            decreasing_line_color="#EF5350",  # Red for decreasing candles
        )
    )

    # Add moving averages if requested and data has enough points
    if add_ma and len(data) > 21:
        plot_data["MA21"] = plot_data["close"].rolling(window=21).mean()
        fig.add_trace(
            go.Scatter(
                x=plot_data["date"],
                y=plot_data["MA21"],
                mode="lines",
                name="21-day MA",
                line=dict(color="#1E88E5", width=1.5),
            )
        )

    if add_ma and len(data) > 200:
        plot_data["MA200"] = plot_data["close"].rolling(window=200).mean()
        fig.add_trace(
            go.Scatter(
                x=plot_data["date"],
                y=plot_data["MA200"],
                mode="lines",
                name="200-day MA",
                line=dict(color="#FFC107", width=1.5),
            )
        )

    # Add reference lines if requested
    if add_ref_lines:
        # Add overbought line (80)
        fig.add_shape(
            type="line",
            x0=plot_data["date"].min(),
            y0=80,
            x1=plot_data["date"].max(),
            y1=80,
            line=dict(color="#F44336", width=1.5, dash="dash"),  # Red
        )

        # Add middle line (50)
        fig.add_shape(
            type="line",
            x0=plot_data["date"].min(),
            y0=50,
            x1=plot_data["date"].max(),
            y1=50,
            line=dict(color="#FFC107", width=1.5, dash="dash"),  # Deep golden yellow
        )

        # Add oversold line (20)
        fig.add_shape(
            type="line",
            x0=plot_data["date"].min(),
            y0=20,
            x1=plot_data["date"].max(),
            y1=20,
            line=dict(color="#4CAF50", width=1.5, dash="dash"),  # Green
        )

        # Annotations for the lines
        fig.add_annotation(
            x=plot_data["date"].max(),
            y=80,
            text="80",
            showarrow=False,
            xshift=10,
            font=dict(color="#F44336"),
        )

        fig.add_annotation(
            x=plot_data["date"].max(),
            y=50,
            text="50",
            showarrow=False,
            xshift=10,
            font=dict(color="#FFC107"),
        )

        fig.add_annotation(
            x=plot_data["date"].max(),
            y=20,
            text="20",
            showarrow=False,
            xshift=10,
            font=dict(color="#4CAF50"),
        )

        # Fix the y-axis range to 0-100
        fig.update_layout(yaxis_range=[0, 100])

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value (0-100)" if normalize else "Price",
        xaxis_rangeslider_visible=False,
        height=450,
    )

    return apply_chart_theme(fig)


def load_mmtv_daily_bars(session, field_type, field_name=None, lookback_days=180):
    """
    Load MMTV daily bars data for sector or industry

    Args:
        session: Database session
        field_type: 'sector' or 'industry'
        field_name: Specific sector or industry name (if None, get all)
        lookback_days: Number of days to look back

    Returns:
        DataFrame with MMTV data
    """
    try:
        from sqlalchemy import text
        import pandas as pd
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        if field_name is None:
            # Get distinct field_names of specified type
            query = """
            SELECT DISTINCT field_name 
            FROM mmtv_daily_bars 
            WHERE field_type = :field_type
            """

            df = pd.read_sql(
                text(query), session.bind, params={"field_type": field_type}
            )

            return df["field_name"].tolist() if not df.empty else []

        else:
            # Get OHLC data for specific field_name
            query = """
            SELECT 
                date, 
                open, 
                high, 
                low, 
                close
            FROM 
                mmtv_daily_bars
            WHERE 
                field_type = :field_type
                AND field_name = :field_name
                AND date BETWEEN :start_date AND :end_date
            ORDER BY 
                date
            """

            df = pd.read_sql(
                text(query),
                session.bind,
                params={
                    "field_type": field_type,
                    "field_name": field_name,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                },
            )

            if df.empty:
                logger.warning(f"No MMTV data found for {field_type} {field_name}")
                return pd.DataFrame()

            # Convert columns
            df["date"] = pd.to_datetime(df["date"])
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col])

            logger.info(f"Loaded {len(df)} MMTV records for {field_type} {field_name}")
            return df

    except Exception as e:
        logger.error(f"Error loading MMTV data for {field_type} {field_name}: {e}")
        st.error(f"Error loading MMTV data: {e}")
        return pd.DataFrame()


# 1. First, create a function to check if price data exists for a symbol
def has_price_data(session, symbol, min_days=1):
    """
    Check if a stock has price data available

    Args:
        session: Database session
        symbol: Stock symbol
        min_days: Minimum number of days of data required

    Returns:
        Boolean indicating if price data exists
    """
    try:
        from sqlalchemy import text

        query = """
        SELECT COUNT(*) as count
        FROM yf_daily_bar
        WHERE symbol = :symbol
        """

        result = session.execute(text(query), {"symbol": symbol}).fetchone()

        if result and result[0] >= min_days:
            return True
        return False

    except Exception as e:
        logger.error(f"Error checking price data for {symbol}: {e}")
        return False


# 2. Modify the load_stocks_by_industry function to filter stocks with no data
def load_stocks_by_industry(session, industry, filter_no_data=True):
    """
    Load stocks for a specific industry

    Args:
        session: Database session
        industry: Industry name
        filter_no_data: Whether to filter out stocks with no price data

    Returns:
        DataFrame with stocks data
    """
    try:
        from sqlalchemy import text
        import pandas as pd

        query = """
        SELECT 
            symbol, 
            company_name, 
            price,
            sector,
            industry
        FROM 
            symbol_fields
        WHERE 
            industry = :industry
        ORDER BY 
            symbol
        """

        df = pd.read_sql(text(query), session.bind, params={"industry": industry})

        if filter_no_data and not df.empty:
            # Filter out stocks with no price data
            has_data = []
            for symbol in df["symbol"]:
                has_data.append(has_price_data(session, symbol))

            # Only keep rows where has_data is True
            df = df[has_data]
            logger.info(
                f"Filtered out {sum(1 for x in has_data if not x)} stocks with no price data"
            )

        logger.info(f"Loaded {len(df)} stocks for industry {industry}")
        return df

    except Exception as e:
        logger.error(f"Error loading stocks for industry {industry}: {e}")
        st.error(f"Error loading stocks: {e}")
        return pd.DataFrame()


def load_stock_price_data(session, symbol, lookback_days=365):
    """
    Load price data for a specific symbol using yf_daily_bar
    """
    try:
        from sqlalchemy import text
        import pandas as pd
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        query = """
        SELECT 
            timestamp as date, 
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
            logger.warning(f"No price data found for symbol {symbol}")
            return pd.DataFrame()

        # Convert columns
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])

        logger.info(f"Loaded {len(df)} price records for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error loading price data for {symbol}: {e}")
        st.error(f"Error loading price data: {e}")
        return pd.DataFrame()


def show_market_breadth():
    """Main function to display the Market Breadth Analysis page"""
    st.title("Market Breadth Analysis")

    # Get database session from session state - this is the key fix
    try:
        if "db_session" not in st.session_state:
            st.error("Database session not found. Please restart the application.")
            return

        session = st.session_state["db_session"]

    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return

    try:
        # Create a 4-column layout with adjusted widths:
        # 1st column small, 2nd and 3rd medium, and 4th larger
        v1, v2, v3, v4 = st.columns([0.8, 2.5, 2.5, 4.2])

        # Column 1: Control panel
        with v1:
            st.markdown("### Analysis Controls")

            # Add buttons for the first vertical
            st.button(
                "Market Breadth - Sector",
                key="btn_sector",
                type="primary",
                use_container_width=True,
            )
            st.button(
                "Market Breadth - ETF",
                key="btn_etf",
                type="primary",
                use_container_width=True,
            )
            st.button(
                "Global Market Shift",
                key="btn_global",
                type="primary",
                use_container_width=True,
            )

            # Add information section at the bottom
            st.markdown("---")
            st.markdown("### How to use")
            st.markdown(
                """
            1. Click on a sector in the second column to view its industries
            2. Click on an industry in the third column to view its stocks
            3. Click on a stock to view detailed price data
            """
            )

        # Initialize session state variables if they don't exist
        if "selected_sector" not in st.session_state:
            # Get all available sectors
            sectors = load_mmtv_daily_bars(session, field_type="sector")
            if sectors:
                st.session_state.selected_sector = sectors[0]  # Default to first sector
            else:
                st.session_state.selected_sector = None

        if "selected_industry" not in st.session_state:
            st.session_state.selected_industry = None

        if "selected_stock" not in st.session_state:
            st.session_state.selected_stock = None

        # Column 2: Sector Analysis
        with v2:
            st.markdown("### Sector Analysis")

            # Get all available sectors
            sectors = load_mmtv_daily_bars(session, field_type="sector")

            if not sectors:
                st.warning("No sector data available")
            else:
                # Load chart data for the selected sector
                sector_data = load_mmtv_daily_bars(
                    session,
                    field_type="sector",
                    field_name=st.session_state.selected_sector,
                )

                if not sector_data.empty:
                    # Create OHLC chart for the sector with reference lines
                    fig = create_ohlc_chart(
                        sector_data,
                        f"{st.session_state.selected_sector} Sector OHLC",
                        normalize=True,  # Scale to 0-100
                        add_ref_lines=True,  # Add the reference lines
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(
                        f"No data available for sector {st.session_state.selected_sector}"
                    )

                # Create a clickable table of sectors
                st.markdown("#### Available Sectors")

                # Create a simple list of buttons for sectors
                for sector in sectors:
                    if st.button(
                        sector,
                        key=f"sector_{sector}",
                        use_container_width=True,
                        type=(
                            "secondary"
                            if sector != st.session_state.selected_sector
                            else "primary"
                        ),
                    ):
                        st.session_state.selected_sector = sector
                        st.session_state.selected_industry = (
                            None  # Reset industry selection
                        )
                        st.session_state.selected_stock = None  # Reset stock selection
                        st.rerun()

        # Column 3: Industry Analysis
        with v3:
            st.markdown("### Industry Analysis")

            if st.session_state.selected_sector:
                # Get all industries in the selected sector
                query = """
                SELECT DISTINCT industry 
                FROM symbol_fields 
                WHERE sector = :sector 
                ORDER BY industry
                """

                from sqlalchemy import text

                industries_df = pd.read_sql(
                    text(query),
                    session.bind,
                    params={"sector": st.session_state.selected_sector},
                )

                industries = (
                    industries_df["industry"].tolist()
                    if not industries_df.empty
                    else []
                )

                # Default selection if needed
                if not st.session_state.selected_industry and industries:
                    st.session_state.selected_industry = industries[0]

                if st.session_state.selected_industry:
                    # Get MMTV data for the industry
                    industry_data = load_mmtv_daily_bars(
                        session,
                        field_type="industry",
                        field_name=st.session_state.selected_industry,
                    )

                    if not industry_data.empty:
                        # Create OHLC chart for the industry with reference lines
                        fig = create_ohlc_chart(
                            industry_data,
                            f"{st.session_state.selected_industry} Industry OHLC",
                            normalize=True,  # Scale to 0-100
                            add_ref_lines=True,  # Add the reference lines
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(
                            f"No MMTV data available for industry {st.session_state.selected_industry}"
                        )

                # Create a clickable list of industries
                st.markdown("#### Available Industries")

                for industry in industries:
                    if st.button(
                        industry,
                        key=f"industry_{industry}",
                        use_container_width=True,
                        type=(
                            "secondary"
                            if industry != st.session_state.selected_industry
                            else "primary"
                        ),
                    ):
                        st.session_state.selected_industry = industry
                        st.session_state.selected_stock = None  # Reset stock selection
                        st.rerun()
            else:
                st.info("Please select a sector first")

        # Column 4: Stock Analysis
        with v4:
            st.markdown("### Stock Analysis")

            if st.session_state.selected_industry:
                # Get all stocks in the selected industry
                stocks_df = load_stocks_by_industry(
                    session, st.session_state.selected_industry, filter_no_data=True
                )

                if stocks_df.empty:
                    st.info(
                        f"No stocks found for industry {st.session_state.selected_industry}"
                    )
                else:
                    # Default selection if needed
                    if not st.session_state.selected_stock:
                        st.session_state.selected_stock = stocks_df["symbol"].iloc[0]

                    # Get price data for the selected stock
                    stock_data = load_stock_price_data(
                        session, st.session_state.selected_stock
                    )

                    if not stock_data.empty:
                        # Create OHLC chart for the stock with MAs
                        fig = create_ohlc_chart(
                            stock_data,
                            f"{st.session_state.selected_stock} Price Chart",
                            add_ma=True,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Get stock details
                        selected_stock_info = stocks_df[
                            stocks_df["symbol"] == st.session_state.selected_stock
                        ].iloc[0]

                        # Show stock info
                        st.markdown(
                            f"**Company:** {selected_stock_info['company_name']}"
                        )
                        st.markdown(
                            f"**Current Price:** ${selected_stock_info['price']:.2f}"
                        )

                        # Show news placeholder
                        st.markdown("### Latest News")
                        st.info("News feed integration coming soon...")
                    else:
                        st.warning(
                            f"No price data available for {st.session_state.selected_stock}"
                        )

                    # Create a table of stocks
                    st.markdown("#### Stocks in this Industry")

                    # Display clickable table
                    for i, row in stocks_df.iterrows():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button(
                                row["symbol"],
                                key=f"stock_{row['symbol']}",
                                type=(
                                    "secondary"
                                    if row["symbol"] != st.session_state.selected_stock
                                    else "primary"
                                ),
                            ):
                                st.session_state.selected_stock = row["symbol"]
                                st.rerun()
                        with col2:
                            st.write(f"{row['company_name']} (${row['price']:.2f})")
            else:
                st.info("Please select an industry first")

    except Exception as e:
        st.error(f"Error in Market Breadth Analysis: {e}")
        logger.exception("Error in Market Breadth Analysis")

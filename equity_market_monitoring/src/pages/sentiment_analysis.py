import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import logging
import json
import webbrowser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define color scheme for sentiment visualization
SENTIMENT_COLORS = {
    "positive": "#26A69A",  # Green
    "neutral": "#64B5F6",  # Blue
    "negative": "#EF5350",  # Red
}


def apply_chart_theme(fig):
    """Apply consistent theme to charts"""
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


def get_top_market_cap_symbols(session, limit=10):
    """Get top symbols by market cap for default selection"""
    try:
        from sqlalchemy import text

        query = """
        SELECT 
            symbol, 
            company_name,
            market_cap
        FROM 
            symbol_fields
        WHERE 
            market_cap IS NOT NULL
        ORDER BY 
            market_cap DESC
        LIMIT :limit
        """

        df = pd.read_sql(text(query), session.bind, params={"limit": limit})
        return df
    except Exception as e:
        logger.error(f"Error getting top market cap symbols: {e}")
        return pd.DataFrame()


def load_news_with_symbols(
    session, symbols=None, publisher=None, limit=50, days_back=30
):
    """Load news articles with related symbols and join with symbol_fields for price data"""
    try:
        from sqlalchemy import text

        # Calculate the date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Base query with left join to get symbol data
        query = """
        SELECT 
            n.id,
            n.symbol AS primary_symbol,
            n.title,
            n.publisher,
            n.link,
            n.published_date,
            n.related_symbols,
            n.preview_text,
            n.thumbnail,
            sf.price,
            sf.change,
            sf.market_cap
        FROM 
            news_articles n
        LEFT JOIN
            symbol_fields sf ON n.symbol = sf.symbol
        WHERE 
            n.published_date >= :start_date
        """

        # Add publisher filter if specified
        if publisher and publisher != "All":
            query += " AND n.publisher = :publisher"

        # Add symbol filter if specified
        if symbols and len(symbols) > 0:
            symbol_placeholders = ", ".join(
                [":sym_" + str(i) for i in range(len(symbols))]
            )
            query += f" AND (n.symbol IN ({symbol_placeholders}) OR n.related_symbols LIKE :symbol_pattern)"

        # Add ordering and limit
        query += """
        ORDER BY 
            n.published_date DESC
        LIMIT :limit
        """

        # Build parameters
        params = {"start_date": start_date.strftime("%Y-%m-%d"), "limit": limit}

        if publisher and publisher != "All":
            params["publisher"] = publisher

        if symbols and len(symbols) > 0:
            # Add individual symbol parameters
            for i, symbol in enumerate(symbols):
                params[f"sym_{i}"] = symbol

            # Add pattern for related_symbols LIKE search
            # This is a basic approach - for production, consider a more robust search
            params["symbol_pattern"] = (
                "%" + symbols[0] + "%"
            )  # Just using first symbol for LIKE

        # Execute query
        df = pd.read_sql(text(query), session.bind, params=params)

        # Parse related_symbols safely
        def parse_related_symbols(symbols_str):
            if pd.isnull(symbols_str) or not symbols_str:
                return []
            try:
                # Try parsing as JSON
                return json.loads(symbols_str)
            except:
                # If JSON parsing fails, try splitting by comma or return as singleton
                if "," in str(symbols_str):
                    return [s.strip() for s in str(symbols_str).split(",")]
                return [str(symbols_str)]

        df["related_symbols_list"] = df["related_symbols"].apply(parse_related_symbols)

        logger.info(f"Loaded {len(df)} news articles")
        return df

    except Exception as e:
        logger.error(f"Error loading news articles: {e}")
        st.error(f"Error loading news: {e}")
        return pd.DataFrame()


def load_news_articles(session, limit=50, days_back=30):
    """Load latest news articles without trying to parse related_symbols as JSON"""
    try:
        from sqlalchemy import text

        # Calculate the date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        query = """
        SELECT 
            n.id,
            n.symbol,
            n.title,
            n.publisher,
            n.link,
            n.published_date,
            n.related_symbols,
            n.preview_text,
            n.thumbnail,
            s.price,
            s.change
        FROM 
            news_articles n
        LEFT JOIN
            symbol_fields s ON n.symbol = s.symbol
        WHERE 
            n.published_date >= :start_date
        ORDER BY 
            n.published_date DESC
        LIMIT :limit
        """

        df = pd.read_sql(
            text(query),
            session.bind,
            params={"start_date": start_date.strftime("%Y-%m-%d"), "limit": limit},
        )

        # Process related symbols safely - don't try to parse as JSON
        def extract_symbols(related_str):
            if pd.isnull(related_str) or not related_str:
                return []

            # Just return the string as a single symbol if no special processing needed
            if "," not in str(related_str):
                return [str(related_str).strip()]

            # Split by comma if the string contains commas
            return [s.strip() for s in str(related_str).split(",")]

        # Apply the extraction function
        df["related_symbols_list"] = df["related_symbols"].apply(extract_symbols)

        logger.info(f"Loaded {len(df)} news articles")
        return df

    except Exception as e:
        logger.error(f"Error loading news articles: {e}")
        st.error(f"Error loading news: {e}")
        return pd.DataFrame()


def load_symbols_data(session, symbols):
    """Load data for specific symbols"""
    if not symbols or len(symbols) == 0:
        return {}

    try:
        from sqlalchemy import text

        # Create placeholders for IN clause
        placeholders = ", ".join([":sym_" + str(i) for i in range(len(symbols))])

        query = f"""
        SELECT 
            symbol,
            price,
            change,
            market_cap
        FROM 
            symbol_fields
        WHERE 
            symbol IN ({placeholders})
        """

        # Build parameters dictionary
        params = {}
        for i, symbol in enumerate(symbols):
            params[f"sym_{i}"] = symbol

        df = pd.read_sql(text(query), session.bind, params=params)

        # Convert to dictionary for easy lookup
        symbols_data = {}
        for _, row in df.iterrows():
            symbols_data[row["symbol"]] = {
                "price": row["price"],
                "change": row["change"],
                "market_cap": row["market_cap"],
            }

        return symbols_data

    except Exception as e:
        logger.error(f"Error loading symbols data: {e}")
        return {}


def load_historical_prices(session, symbol, days=180):
    """Load historical price data for a symbol"""
    try:
        from sqlalchemy import text

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
        ORDER BY 
            timestamp DESC
        LIMIT :days
        """

        df = pd.read_sql(
            text(query), session.bind, params={"symbol": symbol, "days": days}
        )

        # Sort by date ascending for charting
        df = df.sort_values("timestamp")

        # Calculate moving averages
        if len(df) > 21:
            df["MA21"] = df["close"].rolling(window=21).mean()

        if len(df) > 200:
            df["MA200"] = df["close"].rolling(window=200).mean()

        return df
    except Exception as e:
        logger.error(f"Error loading historical prices for {symbol}: {e}")
        return pd.DataFrame()


def create_price_chart(price_df, symbol):
    """Create a price chart with MA lines"""
    if price_df.empty:
        return None

    # Create figure
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=price_df["timestamp"],
            open=price_df["open"],
            high=price_df["high"],
            low=price_df["low"],
            close=price_df["close"],
            name="Price",
            increasing_line_color="#26A69A",  # Green
            decreasing_line_color="#EF5350",  # Red
        )
    )

    # Add MA21 line
    if "MA21" in price_df.columns:
        fig.add_trace(
            go.Scatter(
                x=price_df["timestamp"],
                y=price_df["MA21"],
                name="MA21",
                line=dict(color="#64B5F6", width=1.5),  # Blue
            )
        )

    # Add MA200 line
    if "MA200" in price_df.columns:
        fig.add_trace(
            go.Scatter(
                x=price_df["timestamp"],
                y=price_df["MA200"],
                name="MA200",
                line=dict(color="#FFD54F", width=1.5),  # Yellow
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{symbol} Price History",
        height=300,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return apply_chart_theme(fig)


def show_sentiment_analysis():
    """Main function to display the Sentiment Analysis page"""
    st.title("Market Sentiment Analysis")

    # Get database session
    try:
        if "db_session" not in st.session_state:
            st.error("Database session not found. Please restart the application.")
            return

        session = st.session_state["db_session"]

    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return

    try:
        # Create a 3-column layout: filters, news, chart
        filter_col, news_col, chart_col = st.columns([1, 3, 2])

        # First column: Filters
        with filter_col:
            st.markdown("### News Filters")

            # Date range filter
            st.markdown("#### Date Range")
            days_back = st.slider("Days back", min_value=1, max_value=90, value=30)

            # Symbol filter
            st.markdown("#### Symbol Filter")

            # Get default symbols (most mentioned in news)
            if "default_symbols" not in st.session_state:
                try:
                    default_query = """
                    SELECT symbol, COUNT(*) as count
                    FROM news_articles
                    WHERE symbol IS NOT NULL
                    GROUP BY symbol
                    ORDER BY count DESC
                    LIMIT 20
                    """
                    default_symbols_df = pd.read_sql(default_query, session.bind)
                    st.session_state.default_symbols = default_symbols_df[
                        "symbol"
                    ].tolist()
                except:
                    st.session_state.default_symbols = [
                        "OHI",
                        "SOFI",
                        "A",
                        "MAT",
                        "JNJ",
                    ]

            # Let user select symbol from dropdown or input custom
            symbol_options = (
                ["All"] + st.session_state.default_symbols + ["Custom Input"]
            )
            symbol_choice = st.selectbox(
                "Select symbol filter", options=symbol_options, index=0
            )

            if symbol_choice == "Custom Input":
                symbol_filter = st.text_input("Enter symbol(s) (comma separated)", "")
            elif symbol_choice == "All":
                symbol_filter = ""
            else:
                symbol_filter = symbol_choice

            symbol_list = (
                [s.strip().upper() for s in symbol_filter.split(",")]
                if symbol_filter
                else []
            )

            # Publisher filter
            st.markdown("#### Publisher")
            try:
                publishers_query = """
                SELECT DISTINCT publisher FROM news_articles
                ORDER BY publisher
                """
                publishers_df = pd.read_sql(publishers_query, session.bind)
                publishers = ["All"] + publishers_df["publisher"].tolist()
            except:
                publishers = [
                    "All",
                    "TipRanks",
                    "Benzinga",
                    "Zacks",
                    "Motley Fool",
                    "Reuters",
                    "Insider Monkey",
                    "Simply Wall St.",
                ]

            selected_publisher = st.selectbox("Select news source", publishers)

            # News count
            st.markdown("#### News Count")
            news_count = st.slider(
                "Number of articles", min_value=10, max_value=200, value=50
            )

            # Apply filters button
            apply_filters = st.button("Apply Filters")

        # Second column: News
        with news_col:
            if apply_filters or "news_df" not in st.session_state:
                with st.spinner("Loading news articles..."):
                    # Load news articles
                    news_df = load_news_articles(
                        session, limit=news_count, days_back=days_back
                    )

                    # Apply symbol filter if needed
                    if symbol_list and symbol_choice != "All":
                        filtered_news = []
                        for _, row in news_df.iterrows():
                            # Check primary symbol
                            if row["symbol"] in symbol_list:
                                filtered_news.append(True)
                                continue

                            # Check related symbols (as string, don't parse as JSON)
                            related_str = (
                                str(row["related_symbols"])
                                if pd.notnull(row["related_symbols"])
                                else ""
                            )
                            if any(symbol in related_str for symbol in symbol_list):
                                filtered_news.append(True)
                            else:
                                filtered_news.append(False)

                        news_df = news_df[filtered_news]

                    # Apply publisher filter
                    if selected_publisher != "All":
                        news_df = news_df[news_df["publisher"] == selected_publisher]

                    st.session_state.news_df = news_df

            # Display news articles
            if "news_df" not in st.session_state or st.session_state.news_df.empty:
                st.info("No news articles found. Try adjusting your filters.")
            else:
                # Display news cards
                st.markdown("### Latest Market News")

                # Track which symbols are shown for chart display
                shown_symbols = set()

                for _, article in st.session_state.news_df.iterrows():
                    # Track primary symbol for charting
                    if pd.notnull(article["symbol"]):
                        shown_symbols.add(article["symbol"])

                    # Create card using custom HTML for better styling
                    card_html = f"""
                    <div style="margin-bottom: 15px; padding: 12px; border-radius: 5px; background-color: rgba(30, 30, 30, 0.7); border: 1px solid rgba(80, 80, 80, 0.5);">
                        <div style="display: flex; align-items: flex-start; gap: 12px;">
                            <div style="flex: 3;">
                                <h4 style="margin-top: 0; margin-bottom: 8px;">{article['title']}</h4>
                                <p style="margin-top: 0; margin-bottom: 8px; font-size: 0.8em; opacity: 0.7;">
                                    <strong>Source:</strong> {article['publisher']} | 
                                    <strong>Date:</strong> {article['published_date'].strftime('%Y-%m-%d %H:%M') if pd.notnull(article['published_date']) else 'N/A'}
                                </p>
                                <p style="margin-top: 0; margin-bottom: 8px; font-size: 0.9em;">
                                    {article['preview_text'][:150] + '...' if pd.notnull(article['preview_text']) else 'No preview available'}
                                </p>
                    """

                    # Add related symbols with color-coded changes
                    symbols_html = '<div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px;">'

                    # Add primary symbol first
                    primary_symbol = article["symbol"]
                    if pd.notnull(primary_symbol):
                        change = (
                            article["change"] if pd.notnull(article["change"]) else None
                        )
                        color = (
                            "#26A69A"
                            if change is not None and float(change) > 0
                            else (
                                "#EF5350"
                                if change is not None and float(change) < 0
                                else "#AAAAAA"
                            )
                        )
                        change_text = (
                            f"{float(change):.2f}%" if change is not None else ""
                        )

                        symbols_html += f"""
                        <div style="background-color: rgba(50, 50, 50, 0.8); border-radius: 4px; padding: 3px 8px; display: inline-flex; align-items: center;">
                            <span style="font-weight: bold; margin-right: 5px;">{primary_symbol}</span>
                            <span style="color: {color};">{change_text}</span>
                        </div>
                        """

                    # Add related symbols from the list
                    related_symbols = (
                        article["related_symbols_list"]
                        if hasattr(article, "related_symbols_list")
                        else []
                    )
                    for symbol in related_symbols:
                        if symbol != primary_symbol:
                            symbols_html += f"""
                            <div style="background-color: rgba(50, 50, 50, 0.8); border-radius: 4px; padding: 3px 8px; display: inline-flex; align-items: center;">
                                <span style="margin-right: 5px;">{symbol}</span>
                            </div>
                            """

                    symbols_html += "</div>"

                    # Complete the first column
                    card_html += symbols_html + "</div>"

                    # Add thumbnail if available in the second column
                    card_html += '<div style="flex: 1; text-align: right;">'
                    if pd.notnull(article["thumbnail"]) and article["thumbnail"]:
                        card_html += f'<img src="{article["thumbnail"]}" style="max-width: 100%; max-height: 120px; border-radius: 4px; object-fit: cover;" />'
                    card_html += "</div>"

                    # Close the flex container and add link to full article
                    card_html += """
                        </div>
                        <div style="margin-top: 8px; text-align: right;">
                    """

                    if pd.notnull(article["link"]):
                        card_html += f'<a href="{article["link"]}" target="_blank" style="color: #64B5F6; text-decoration: none; font-size: 0.9em;">Read full article →</a>'

                    card_html += """
                        </div>
                    </div>
                    """

                    # Display the card
                    st.markdown(card_html, unsafe_allow_html=True)

                # Store the symbols for the chart column to use
                if shown_symbols:
                    st.session_state.shown_symbols = list(shown_symbols)

        # Third column: Price charts
        with chart_col:
            st.markdown("### Price Charts")

            # Show charts for symbols found in news
            symbols_to_chart = []

            # First priority: Selected symbol from filter
            if symbol_list and symbol_choice != "All":
                symbols_to_chart = symbol_list[
                    :3
                ]  # Limit to first 3 symbols if multiple selected
            # Second priority: Symbols from news articles
            elif "shown_symbols" in st.session_state and st.session_state.shown_symbols:
                symbols_to_chart = st.session_state.shown_symbols[
                    :3
                ]  # Show first 3 symbols

            if symbols_to_chart:
                # Let user select which symbol to display if multiple options
                if len(symbols_to_chart) > 1:
                    selected_chart_symbol = st.selectbox(
                        "Select symbol to chart", symbols_to_chart
                    )
                else:
                    selected_chart_symbol = symbols_to_chart[0]

                # Load and display price chart for selected symbol
                with st.spinner(f"Loading price data for {selected_chart_symbol}..."):
                    price_df = load_historical_prices(session, selected_chart_symbol)

                    if not price_df.empty:
                        fig = create_price_chart(price_df, selected_chart_symbol)
                        st.plotly_chart(fig, use_container_width=True)

                        # Add key statistics below chart
                        latest_price = (
                            price_df["close"].iloc[-1] if len(price_df) > 0 else None
                        )
                        change_1d = (
                            price_df["close"].iloc[-1] - price_df["close"].iloc[-2]
                            if len(price_df) > 1
                            else None
                        )
                        pct_change_1d = (
                            (change_1d / price_df["close"].iloc[-2] * 100)
                            if change_1d is not None
                            else None
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Latest Price",
                                f"${latest_price:.2f}" if latest_price else "N/A",
                                f"{pct_change_1d:.2f}%" if pct_change_1d else None,
                                delta_color="normal",
                            )

                        with col2:
                            # Calculate distance from MA-200
                            if "MA200" in price_df.columns and not pd.isna(
                                price_df["MA200"].iloc[-1]
                            ):
                                dist_from_ma200 = (
                                    (latest_price / price_df["MA200"].iloc[-1]) - 1
                                ) * 100
                                st.metric(
                                    "Vs. MA-200",
                                    f"{dist_from_ma200:.2f}%",
                                    delta_color="normal",
                                )
                    else:
                        st.info(f"No price data available for {selected_chart_symbol}")
            else:
                st.info("Select a symbol or view news to see price charts")

    except Exception as e:
        st.error(f"Error in Sentiment Analysis: {e}")
        st.exception(e)
        logger.exception("Error in Sentiment Analysis")


if __name__ == "__main__":
    show_sentiment_analysis()

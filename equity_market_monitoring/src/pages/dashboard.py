import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import database connection
from src.database.mysql_connection import (
    get_mysql_connection,
    load_risk_profile_data,
    get_risk_distribution_metrics,
)
from src.database.models import SymbolFields, YfBar1d, EquityRiskProfile

# Import utility functions from new modules
from src.utils.chart_utils import (
    RISK_COLORS,
    apply_chart_theme,
    display_metric_with_tooltip,
    display_notification,
    format_delta_for_metric,
    create_risk_metrics_radar_chart,
    create_focus_risk_chart,
    create_asset_class_risk_chart,
    create_drawdown_time_chart,
)


# Load environment variables
load_dotenv()


def load_custom_css():
    """Load custom CSS styles from file"""
    css_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "assets", "custom_styles.css"
    )

    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        logger.warning(f"Custom CSS file not found at {css_file}")


def draw_market_overview_tab(risk_data):
    """
    Draw the Market Drawdown Overview tab

    Args:
        risk_data: DataFrame with risk profile data
    """
    st.header("Market Risk Classification Analysis")

    with st.expander("About This Dashboard", expanded=False):
        st.markdown(
            """
        This dashboard provides analysis of the risk classification of stocks and ETFs based on several key metrics:
        
        - **Volatility**: Measures the dispersion of returns over time (higher = more risk)
        - **Beta**: Measures sensitivity to market movements compared to the S&P 500 benchmark (higher = more risk)
        - **Max Drawdown**: Largest peak-to-trough decline (more negative = more risk)
        - **ADR (Average Daily Range)**: Average daily price fluctuation as a percentage (higher = more risk)
        
        The risk classification combines these metrics to categorize each asset as:
        - **Conservative**: Lower risk, typically with volatility < 15%, beta < 0.8, etc.
        - **Moderate**: Medium risk, balanced characteristics
        - **Aggressive**: Higher risk, typically with volatility > 21%, beta > 1.2, etc.
        """
        )

    # ETF and Stock sections side by side
    col1, col2 = st.columns(2)

    # ETF Section
    with col1:
        st.subheader("ETF Market Drawdown Analysis")

        if risk_data[risk_data["is_etf"]].empty:
            st.info("No ETF data available. Run the script with --include-etfs flag.")
        else:
            # Charts in 2x2 grid
            row1_col1, row1_col2 = st.columns(2)

            with row1_col1:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.plotly_chart(
                    create_focus_risk_chart(risk_data, is_etf=True),
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with row1_col2:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.plotly_chart(
                    create_asset_class_risk_chart(risk_data, is_etf=True),
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # Drawdown analysis
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(
                create_drawdown_time_chart(risk_data, is_etf=True),
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Risk metrics radar chart
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(
                create_risk_metrics_radar_chart(risk_data, is_etf=True),
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Stock Section
    with col2:
        st.subheader("Stock Market Drawdown Analysis")

        if risk_data[~risk_data["is_etf"]].empty:
            st.info(
                "No Stock data available. Run the script with --include-stocks flag."
            )
        else:
            # Charts in 2x2 grid
            row1_col1, row1_col2 = st.columns(2)

            with row1_col1:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.plotly_chart(
                    create_focus_risk_chart(risk_data, is_etf=False),
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with row1_col2:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.plotly_chart(
                    create_asset_class_risk_chart(risk_data, is_etf=False),
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # Drawdown analysis
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(
                create_drawdown_time_chart(risk_data, is_etf=False),
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Risk metrics radar chart
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(
                create_risk_metrics_radar_chart(risk_data, is_etf=False),
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)


def draw_detailed_analysis_tab(risk_data, session):
    """
    Draw the Detailed Analysis tab with advanced metrics and comparisons

    Args:
        risk_data: DataFrame with risk profile data
        session: Database session for additional queries
    """
    st.header("Detailed Market Risk Analysis")

    # Create sections for analysis type
    analysis_type = st.radio(
        "Analysis Type",
        ["Risk Category Comparison", "Symbol Analysis", "Time Series Analysis"],
        horizontal=True,
    )

    if analysis_type == "Risk Category Comparison":
        # Risk category comparison section
        st.subheader("Risk Category Metrics Comparison")

        # Select asset type for comparison
        asset_filter = st.radio(
            "Asset Type", ["Both", "ETFs Only", "Stocks Only"], horizontal=True
        )

        filtered_data = risk_data.copy()
        if asset_filter == "ETFs Only":
            filtered_data = risk_data[risk_data["is_etf"]]
        elif asset_filter == "Stocks Only":
            filtered_data = filtered_data[~filtered_data["is_etf"]]

        if filtered_data.empty:
            st.warning("No data available for the selected filters.")
            return

        # Group data by risk type and calculate average metrics
        metrics_cols = [
            "volatility_3yr_avg",
            "beta_3yr_avg",
            "max_drawdown_3yr_avg",
            "adr_3yr_avg",
            "average_score",
        ]

        risk_metrics = (
            filtered_data.groupby("risk_type")[metrics_cols].mean().reset_index()
        )

        # Create comparison charts
        col1, col2 = st.columns(2)

        with col1:
            # Volatility comparison
            fig = px.bar(
                risk_metrics,
                x="risk_type",
                y="volatility_3yr_avg",
                color="risk_type",
                color_discrete_map=RISK_COLORS,
                title="Average Volatility by Risk Category",
                labels={
                    "volatility_3yr_avg": "Average Volatility (%)",
                    "risk_type": "Risk Category",
                },
            )
            fig.update_layout(showlegend=False)
            fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            # Max drawdown comparison
            fig = px.bar(
                risk_metrics,
                x="risk_type",
                y="max_drawdown_3yr_avg",
                color="risk_type",
                color_discrete_map=RISK_COLORS,
                title="Average Maximum Drawdown by Risk Category",
                labels={
                    "max_drawdown_3yr_avg": "Average Max Drawdown (%)",
                    "risk_type": "Risk Category",
                },
            )
            fig.update_layout(showlegend=False)
            fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Beta comparison
            fig = px.bar(
                risk_metrics,
                x="risk_type",
                y="beta_3yr_avg",
                color="risk_type",
                color_discrete_map=RISK_COLORS,
                title="Average Beta by Risk Category",
                labels={"beta_3yr_avg": "Average Beta", "risk_type": "Risk Category"},
            )
            fig.update_layout(showlegend=False)
            fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            # ADR comparison
            fig = px.bar(
                risk_metrics,
                x="risk_type",
                y="adr_3yr_avg",
                color="risk_type",
                color_discrete_map=RISK_COLORS,
                title="Average Daily Range (ADR) by Risk Category",
                labels={
                    "adr_3yr_avg": "Average Daily Range (%)",
                    "risk_type": "Risk Category",
                },
            )
            fig.update_layout(showlegend=False)
            fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Risk score distribution
        st.subheader("Risk Score Distribution")
        # Create a histogram of risk scores
        fig = px.histogram(
            filtered_data,
            x="average_score",
            color="risk_type",
            color_discrete_map=RISK_COLORS,
            marginal="box",
            nbins=50,
            opacity=0.7,
            title="Distribution of Risk Scores",
            labels={"average_score": "Risk Score", "count": "Number of Assets"},
        )
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Symbol Analysis":
        # Individual symbol analysis section
        st.subheader("Individual Symbol Analysis")

        # Filter by asset type
        asset_type = st.radio("Asset Type", ["ETFs", "Stocks"], horizontal=True)
        is_etf = asset_type == "ETFs"

        # Filter data by asset type
        symbols_data = risk_data[risk_data["is_etf"] == is_etf]

        if symbols_data.empty:
            st.warning(f"No {asset_type} data available.")
            return

        # Create filters for symbol selection
        col1, col2, col3 = st.columns(3)

        with col1:
            # Filter by risk type
            risk_types = ["All"] + sorted(symbols_data["risk_type"].unique().tolist())
            selected_risk = st.selectbox("Risk Type", risk_types)

            if selected_risk != "All":
                symbols_data = symbols_data[symbols_data["risk_type"] == selected_risk]

        with col2:
            # Filter by sector/focus
            if "sector" in symbols_data.columns and not is_etf:
                sectors = ["All"] + sorted(symbols_data["sector"].unique().tolist())
                selected_sector = st.selectbox("Sector", sectors)

                if selected_sector != "All":
                    symbols_data = symbols_data[
                        symbols_data["sector"] == selected_sector
                    ]
            elif "focus" in symbols_data.columns and is_etf:
                focuses = ["All"] + sorted(symbols_data["focus"].unique().tolist())
                selected_focus = st.selectbox("Focus", focuses)

                if selected_focus != "All":
                    symbols_data = symbols_data[symbols_data["focus"] == selected_focus]

        with col3:
            # Select symbol
            if not symbols_data.empty:
                symbols = sorted(symbols_data["symbol"].unique().tolist())
                selected_symbol = st.selectbox("Select Symbol", symbols)

                # Get the selected symbol data
                symbol_data = symbols_data[
                    symbols_data["symbol"] == selected_symbol
                ].iloc[0]
            else:
                st.warning("No symbols match the selected filters.")
                return

        # Display symbol information and metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Symbol Information")
            info_data = {
                "Symbol": selected_symbol,
                "Name": symbol_data.get("company_name", "N/A"),
                "Risk Type": symbol_data.get("risk_type", "N/A"),
                "Asset Type": "ETF" if is_etf else "Stock",
                "Sector/Focus": symbol_data.get(
                    "sector" if not is_etf else "focus", "N/A"
                ),
                "Country": symbol_data.get("country", "N/A"),
                "Exchange": symbol_data.get("exchange", "N/A"),
            }

            # Display as a styled dataframe
            info_df = pd.DataFrame(
                list(info_data.items()), columns=["Attribute", "Value"]
            )
            st.dataframe(info_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("### Risk Metrics")
            metrics_data = {
                "Volatility (3yr Avg)": f"{symbol_data.get('volatility_3yr_avg', 0):.2%}",
                "Beta (3yr Avg)": f"{symbol_data.get('beta_3yr_avg', 0):.2f}",
                "Max Drawdown (3yr Avg)": f"{symbol_data.get('max_drawdown_3yr_avg', 0):.2%}",
                "Avg Daily Range": f"{symbol_data.get('adr_3yr_avg', 0):.2%}",
                "Risk Score": f"{symbol_data.get('average_score', 0):.1f}/10",
            }

            # Display as a styled dataframe
            metrics_df = pd.DataFrame(
                list(metrics_data.items()), columns=["Metric", "Value"]
            )
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        # Fetch historical price data for the selected symbol
        st.markdown("### Historical Price and Drawdown")

        try:
            with st.spinner(f"Loading price data for {selected_symbol}..."):
                from src.database.mysql_connection import load_price_data

                price_data = load_price_data(
                    session,
                    selected_symbol,
                    lookback_days=395,
                )

                if not price_data.empty:
                    # Create candlestick chart with moving averages
                    fig = go.Figure()

                    # Add candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=price_data["timestamp"],
                            open=price_data["open"],
                            high=price_data["high"],
                            low=price_data["low"],
                            close=price_data["close"],
                            name="Price",
                            increasing_line_color="#26A69A",  # Green for increasing candles
                            decreasing_line_color="#EF5350",  # Red for decreasing candles
                        )
                    )

                    # Add moving averages if available
                    if "MA21" in price_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=price_data["timestamp"],
                                y=price_data["MA21"],
                                mode="lines",
                                name="21-day MA",
                                line=dict(color="#1E88E5", width=1.5),
                            )
                        )

                    if "MA200" in price_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=price_data["timestamp"],
                                y=price_data["MA200"],
                                mode="lines",
                                name="200-day MA",
                                line=dict(color="#4CAF50", width=1.5),
                            )
                        )

                    # Update layout
                    fig.update_layout(
                        title=f"{selected_symbol} - Price History (1 Year)",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False,  # Hide rangeslider for cleaner look
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )

                    fig = apply_chart_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)

                    # Create drawdown chart if available
                    if "drawdown" in price_data.columns:
                        fig = go.Figure()

                        # Add drawdown area chart
                        fig.add_trace(
                            go.Scatter(
                                x=price_data["timestamp"],
                                y=price_data["drawdown"],
                                fill="tozeroy",
                                fillcolor="rgba(229, 57, 53, 0.3)",
                                line=dict(color="#E53935", width=1),
                                name="Drawdown",
                            )
                        )

                        # Add horizontal reference lines for risk levels - ONLY 2 lines now
                        fig.add_shape(
                            type="line",
                            x0=price_data["timestamp"].min(),
                            x1=price_data["timestamp"].max(),
                            y0=-0.15,
                            y1=-0.15,
                            line=dict(
                                color="#4CAF50",  # Green for conservative
                                width=2,
                                dash="dash",
                            ),
                            name="Conservative",
                        )

                        fig.add_shape(
                            type="line",
                            x0=price_data["timestamp"].min(),
                            x1=price_data["timestamp"].max(),
                            y0=-0.21,
                            y1=-0.21,
                            line=dict(
                                color="#FFC107",  # Yellow/amber for moderate
                                width=2,
                                dash="dash",
                            ),
                            name="Moderate",
                        )

                        # Add annotations for the lines - ONLY 2 annotations now
                        fig.add_annotation(
                            x=price_data["timestamp"].min(),
                            y=-0.15,
                            text="Conservative (-15%)",
                            showarrow=False,
                            yshift=10,
                            xshift=80,
                            font=dict(color="#4CAF50"),
                        )

                        fig.add_annotation(
                            x=price_data["timestamp"].min(),
                            y=-0.21,
                            text="Moderate (-21%)",
                            showarrow=False,
                            yshift=10,
                            xshift=70,
                            font=dict(color="#FFC107"),
                        )

                        # Mark the max drawdown point
                        max_dd_idx = price_data["drawdown"].idxmin()
                        max_dd = price_data.loc[max_dd_idx, "drawdown"]
                        max_dd_date = price_data.loc[max_dd_idx, "timestamp"]

                        fig.add_trace(
                            go.Scatter(
                                x=[max_dd_date],
                                y=[max_dd],
                                mode="markers+text",
                                marker=dict(color="#E53935", size=10),
                                text=["Max DD"],
                                textposition="top center",
                                name="Maximum Drawdown",
                            )
                        )

                        # Update layout
                        fig.update_layout(
                            title=f"{selected_symbol} - Drawdown Analysis (1 Year)",
                            xaxis_title="Date",
                            yaxis_title="Drawdown (%)",
                            yaxis=dict(tickformat=".0%"),
                        )

                        fig = apply_chart_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No price data available for {selected_symbol}")
        except Exception as e:
            st.error(f"Error loading price data: {e}")

        # For ETFs, show holdings data
        if is_etf:
            st.markdown("### ETF Holdings Analysis")

            try:
                with st.spinner(f"Loading holdings data for {selected_symbol}..."):
                    from src.database.mysql_connection import load_etf_holdings

                    holdings_data = load_etf_holdings(session, selected_symbol)

                    if not holdings_data.empty:
                        # Format data for display
                        display_data = holdings_data.copy()
                        display_data["weight"] = display_data["weight"].map(
                            lambda x: f"{x:.2f}%" if pd.notnull(x) else "-"
                        )
                        display_data["market_value"] = display_data["market_value"].map(
                            lambda x: f"${x:,.0f}" if pd.notnull(x) else "-"
                        )
                        display_data["price"] = display_data["price"].map(
                            lambda x: f"${x:.2f}" if pd.notnull(x) else "-"
                        )
                        display_data["drawdown"] = display_data["drawdown"].map(
                            lambda x: f"{x:.2%}" if pd.notnull(x) else "-"
                        )

                        # Keep numeric version for chart
                        holdings_data["weight_numeric"] = holdings_data["weight"]

                        # VERTICAL LAYOUT - All in a single column
                        # 1. First the table
                        st.markdown("#### All ETF Holdings")
                        st.dataframe(
                            display_data[
                                [
                                    "stock_symbol",
                                    "name",
                                    "sector",
                                    "weight",
                                    "market_value",
                                    "price",
                                    "drawdown",
                                    "risk_type",
                                ]
                            ],
                            hide_index=True,
                            column_config={
                                "stock_symbol": "Stock Symbol",
                                "name": "Name",
                                "sector": "Sector",
                                "weight": "Weight",
                                "market_value": "Market Value",
                                "price": "Price",
                                "drawdown": "Drawdown",
                                "risk_type": "Risk Type",
                            },
                            use_container_width=True,
                            height=400,  # Set a fixed height to make it scrollable
                        )

                        # 2. Then the chart with dropdown legend options
                        st.markdown("#### Top Holdings Distribution")

                        # Add dropdown for legend selection
                        legend_options = [
                            "Top 10",
                            "Top 20",
                            "By Sector",
                            "By Risk Type",
                        ]
                        selected_view = st.selectbox(
                            "Chart Display Option", legend_options, index=0
                        )

                        # Process data based on selection
                        if selected_view == "By Sector":
                            # Group by sector
                            sector_data = (
                                holdings_data.groupby("sector")["weight_numeric"]
                                .sum()
                                .reset_index()
                            )
                            sector_data = sector_data.sort_values(
                                by="weight_numeric", ascending=False
                            )

                            # Take top sectors and group the rest as "Other Sectors"
                            top_sectors = sector_data.head(8)
                            other_sectors_weight = (
                                sector_data.iloc[8:]["weight_numeric"].sum()
                                if len(sector_data) > 8
                                else 0
                            )

                            if other_sectors_weight > 0:
                                chart_labels = top_sectors["sector"].tolist() + [
                                    "Other Sectors"
                                ]
                                chart_values = top_sectors[
                                    "weight_numeric"
                                ].tolist() + [other_sectors_weight]
                            else:
                                chart_labels = top_sectors["sector"].tolist()
                                chart_values = top_sectors["weight_numeric"].tolist()

                            title = f"{selected_symbol} - Holdings by Sector"

                        elif selected_view == "By Risk Type":
                            # Group by risk type, handling None values
                            holdings_data["risk_type"] = holdings_data[
                                "risk_type"
                            ].fillna("Not Classified")
                            risk_data = (
                                holdings_data.groupby("risk_type")["weight_numeric"]
                                .sum()
                                .reset_index()
                            )
                            risk_data = risk_data.sort_values(
                                by="weight_numeric", ascending=False
                            )

                            chart_labels = risk_data["risk_type"].tolist()
                            chart_values = risk_data["weight_numeric"].tolist()
                            title = f"{selected_symbol} - Holdings by Risk Type"

                        elif selected_view == "Top 10":
                            # Get top 10 holdings
                            top_n = holdings_data.nlargest(10, "weight_numeric")
                            others_weight = (
                                holdings_data["weight_numeric"].sum()
                                - top_n["weight_numeric"].sum()
                            )

                            if len(holdings_data) > 10:
                                chart_labels = top_n["stock_symbol"].tolist() + [
                                    "Others"
                                ]
                                chart_values = top_n["weight_numeric"].tolist() + [
                                    others_weight
                                ]
                            else:
                                chart_labels = top_n["stock_symbol"].tolist()
                                chart_values = top_n["weight_numeric"].tolist()

                            title = f"{selected_symbol} - Top 10 Holdings"

                        else:  # Top 20
                            # Get top 20 holdings
                            top_n = holdings_data.nlargest(20, "weight_numeric")
                            others_weight = (
                                holdings_data["weight_numeric"].sum()
                                - top_n["weight_numeric"].sum()
                            )

                            if len(holdings_data) > 20:
                                chart_labels = top_n["stock_symbol"].tolist() + [
                                    "Others"
                                ]
                                chart_values = top_n["weight_numeric"].tolist() + [
                                    others_weight
                                ]
                            else:
                                chart_labels = top_n["stock_symbol"].tolist()
                                chart_values = top_n["weight_numeric"].tolist()

                            title = f"{selected_symbol} - Top 20 Holdings"

                        # Create donut chart
                        fig = go.Figure(
                            data=[
                                go.Pie(
                                    labels=chart_labels,
                                    values=chart_values,
                                    hole=0.4,
                                    textinfo="label+percent",
                                    textposition="outside",
                                    texttemplate="%{label}: %{percent:.1%}",
                                    marker=dict(line=dict(color="#000000", width=0.5)),
                                    insidetextorientation="radial",
                                )
                            ]
                        )

                        # Set colors based on view type
                        if selected_view == "By Risk Type":
                            # Use consistent risk colors
                            color_map = {
                                "Conservative": "#4CAF50",
                                "Moderate": "#FFC107",
                                "Aggressive": "#F44336",
                                "Not Classified": "#9E9E9E",
                            }
                            colors = [
                                color_map.get(risk, "#9E9E9E") for risk in chart_labels
                            ]
                            fig.update_traces(marker=dict(colors=colors))

                        fig.update_layout(
                            title=title,
                            # Move legend to dropdown position
                            showlegend=False,  # Hide default legend since we're using dropdown
                            margin=dict(t=50, b=20, l=20, r=20),
                            height=500,  # Taller chart since we're in vertical layout
                        )

                        fig = apply_chart_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(
                            f"No holdings data available for ETF {selected_symbol}"
                        )

            except Exception as e:
                st.error(f"Error loading ETF holdings: {e}")

    elif analysis_type == "Time Series Analysis":
        # Time series analysis of drawdowns and risk metrics
        st.subheader("Risk Metrics Time Series Analysis")

        # Select asset type for time series
        asset_filter = st.radio(
            "Asset Type", ["Both", "ETFs Only", "Stocks Only"], horizontal=True
        )

        filtered_data = risk_data.copy()
        if asset_filter == "ETFs Only":
            filtered_data = risk_data[risk_data["is_etf"]]
        elif asset_filter == "Stocks Only":
            filtered_data = filtered_data[~filtered_data["is_etf"]]

        if filtered_data.empty:
            st.warning("No data available for the selected filters.")
            return

        # Convert yearly columns to time series format for visualization
        years = ["year1", "year2", "year3"]
        year_labels = ["1 Year Ago", "2 Years Ago", "3 Years Ago"]
        year_map = dict(zip(years, year_labels))

        # Gather metrics to analyze
        metrics_to_analyze = st.multiselect(
            "Select Metrics to Analyze",
            ["Volatility", "Beta", "Maximum Drawdown", "ADR (Average Daily Range)"],
            default=["Volatility", "Maximum Drawdown"],
        )

        if not metrics_to_analyze:
            st.warning("Please select at least one metric to analyze.")
            return

        # Create time series dataframe
        time_series_data = []

        if "Volatility" in metrics_to_analyze:
            for year in years:
                for risk_type in filtered_data["risk_type"].unique():
                    subset = filtered_data[filtered_data["risk_type"] == risk_type]
                    if f"volatility_{year}" in subset.columns:
                        avg_value = subset[f"volatility_{year}"].mean()
                        if not pd.isna(avg_value):
                            time_series_data.append(
                                {
                                    "metric": "Volatility",
                                    "year": year_map[year],
                                    "risk_type": risk_type,
                                    "value": avg_value,
                                }
                            )

        if "Beta" in metrics_to_analyze:
            for year in years:
                for risk_type in filtered_data["risk_type"].unique():
                    subset = filtered_data[filtered_data["risk_type"] == risk_type]
                    if f"beta_{year}" in subset.columns:
                        avg_value = subset[f"beta_{year}"].mean()
                        if not pd.isna(avg_value):
                            time_series_data.append(
                                {
                                    "metric": "Beta",
                                    "year": year_map[year],
                                    "risk_type": risk_type,
                                    "value": avg_value,
                                }
                            )

        if "Maximum Drawdown" in metrics_to_analyze:
            for year in years:
                for risk_type in filtered_data["risk_type"].unique():
                    subset = filtered_data[filtered_data["risk_type"] == risk_type]
                    if f"max_drawdown_{year}" in subset.columns:
                        avg_value = subset[f"max_drawdown_{year}"].mean()
                        if not pd.isna(avg_value):
                            time_series_data.append(
                                {
                                    "metric": "Maximum Drawdown",
                                    "year": year_map[year],
                                    "risk_type": risk_type,
                                    "value": avg_value,
                                }
                            )

        if "ADR (Average Daily Range)" in metrics_to_analyze:
            for year in years:
                for risk_type in filtered_data["risk_type"].unique():
                    subset = filtered_data[filtered_data["risk_type"] == risk_type]
                    if f"adr_{year}" in subset.columns:
                        avg_value = subset[f"adr_{year}"].mean()
                        if not pd.isna(avg_value):
                            time_series_data.append(
                                {
                                    "metric": "ADR (Average Daily Range)",
                                    "year": year_map[year],
                                    "risk_type": risk_type,
                                    "value": avg_value,
                                }
                            )

        if not time_series_data:
            st.warning("No time series data available for the selected metrics.")
            return

        time_series_df = pd.DataFrame(time_series_data)

        # Create line charts for each metric
        for metric in metrics_to_analyze:
            metric_data = time_series_df[time_series_df["metric"] == metric]

            if not metric_data.empty:
                fig = px.line(
                    metric_data,
                    x="year",
                    y="value",
                    color="risk_type",
                    color_discrete_map=RISK_COLORS,
                    markers=True,
                    title=f"{metric} Trends by Risk Type",
                    labels={
                        "value": metric,
                        "year": "Period",
                        "risk_type": "Risk Category",
                    },
                )

                # Add formatting based on metric type
                if metric == "Maximum Drawdown":
                    fig.update_layout(yaxis_tickformat=".0%")
                elif metric in ["Volatility", "ADR (Average Daily Range)"]:
                    fig.update_layout(yaxis_tickformat=".1%")

                fig = apply_chart_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for {metric}.")


def show_dashboard():
    st.title("Stock and ETF Market Drawdown Analysis Dashboard")

    # Load custom CSS
    load_custom_css()

    # Connect to database
    engine, session = get_mysql_connection()

    if engine is None or session is None:
        st.error("Failed to connect to database. Please check logs for details.")
        return

    try:
        # Global filters for sidebar
        st.sidebar.header("Dashboard Filters")

        with st.sidebar.expander("Data Options", expanded=True):
            include_stocks = st.checkbox("Include Stocks", value=True)
            include_etfs = st.checkbox("Include ETFs", value=True)

            if not include_stocks and not include_etfs:
                st.sidebar.warning("Please select at least one asset type")
                include_stocks = True
                st.checkbox("Include Stocks", value=True)

        # Load risk profile data with spinner
        with st.spinner("Loading risk profile data..."):
            risk_data = load_risk_profile_data(
                session, include_stocks=include_stocks, include_etfs=include_etfs
            )

        if risk_data.empty:
            st.error(
                "No risk profile data found. Please ensure the database is properly initialized."
            )
            return

        # Display metrics in a card-like container
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        # Dashboard metrics
        total_symbols, etf_count, stock_count, risk_distribution = (
            get_risk_distribution_metrics(risk_data)
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            display_metric_with_tooltip(
                "Total Assets",
                f"{total_symbols:,}",
                tooltip="Total number of stocks and ETFs with risk profiles",
            )

        with col2:
            display_metric_with_tooltip(
                "ETFs", f"{etf_count:,}", tooltip="Number of ETFs with risk profiles"
            )

        with col3:
            display_metric_with_tooltip(
                "Stocks",
                f"{stock_count:,}",
                tooltip="Number of stocks with risk profiles",
            )

        with col4:
            if not risk_distribution.empty:
                most_common_risk = risk_distribution.idxmax()
                pct_most_common = risk_distribution.max()

                display_metric_with_tooltip(
                    "Most Common Risk",
                    f"{most_common_risk} ({pct_most_common:.1%})",
                    tooltip="Most common risk type across all assets",
                )
            else:
                display_metric_with_tooltip(
                    "Risk Profile", "No data", tooltip="Risk profile data not available"
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # Create tabs for different sections
        tab1, tab2 = st.tabs(["Market Risk Overview", "Detailed Analysis"])

        with tab1:
            draw_market_overview_tab(risk_data)

        with tab2:
            # Replace the placeholder with actual detailed analysis
            draw_detailed_analysis_tab(risk_data, session)

    finally:
        # Close database session
        if session:
            session.close()


if __name__ == "__main__":
    show_dashboard()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import database connection classes (if needed)
from src.database.mysql_connection import get_mysql_connection

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


def load_sectors(session):
    """Load distinct sectors from the database"""
    try:
        from sqlalchemy import text

        query = """
        SELECT DISTINCT sector 
        FROM symbol_fields 
        WHERE sector IS NOT NULL AND sector != ''
        ORDER BY sector
        """

        result = pd.read_sql(text(query), session.bind)
        return result["sector"].tolist()
    except Exception as e:
        logger.error(f"Error loading sectors: {e}")
        return []


def load_stocks_by_sector(session, sector):
    """Load stocks for a specific sector"""
    try:
        from sqlalchemy import text

        # Keep original query - it correctly selects the database columns
        query = """
        SELECT 
            symbol, 
            company_name, 
            price,
            industry,
            earnings_release_trading_date_fq,
            earnings_release_next_trading_date_fq
        FROM 
            symbol_fields
        WHERE 
            sector = :sector
        ORDER BY 
            symbol
        """

        df = pd.read_sql(text(query), session.bind, params={"sector": sector})
        logger.info(f"Loaded {len(df)} stocks for sector {sector}")
        return df
    except Exception as e:
        logger.error(f"Error loading stocks for sector {sector}: {e}")
        st.error(f"Error loading stocks: {e}")
        return pd.DataFrame()


def load_income_statement(session, symbol, period_type="Annual"):
    """Load income statement data for a symbol"""
    try:
        from sqlalchemy import text

        query = """
        SELECT 
            date,
            period_type,
            total_revenue,
            cost_of_revenue,
            gross_profit,
            operating_expense,
            operating_income,
            total_operating_income,
            total_expenses,
            net_non_operating_interest,
            other_income_expense,
            pretax_income,
            tax_provision,
            net_income
        FROM 
            income_statements
        WHERE 
            symbol = :symbol
            AND period_type LIKE :period_type
        ORDER BY 
            date DESC
        LIMIT 8
        """

        df = pd.read_sql(
            text(query),
            session.bind,
            params={
                "symbol": symbol,
                "period_type": f"{period_type}%",  # Use wildcard to match both "Annual" and "Quarterly"
            },
        )

        logger.info(f"Loaded {len(df)} income statement records for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Error loading income statement for {symbol}: {e}")
        return pd.DataFrame()


def load_balance_sheet(session, symbol, period_type="Annual"):
    """Load balance sheet data for a symbol"""
    try:
        from sqlalchemy import text

        query = """
        SELECT 
            date,
            period_type,
            total_assets,
            total_liabilities,
            total_equity,
            total_capitalization,
            common_stock_equity,
            capital_lease_obligations,
            net_tangible_assets,
            working_capital,
            invested_capital,
            tangible_book_value,
            total_debt,
            shares_issued
        FROM 
            balance_sheets
        WHERE 
            symbol = :symbol
            AND period_type LIKE :period_type
        ORDER BY 
            date DESC
        LIMIT 8
        """

        df = pd.read_sql(
            text(query),
            session.bind,
            params={
                "symbol": symbol,
                "period_type": f"{period_type}%",  # Use wildcard to match both "Annual" and "Quarterly"
            },
        )

        logger.info(f"Loaded {len(df)} balance sheet records for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Error loading balance sheet for {symbol}: {e}")
        return pd.DataFrame()


def load_cash_flow(session, symbol, period_type="Annual"):
    """Load cash flow data for a symbol"""
    try:
        from sqlalchemy import text

        query = """
        SELECT 
            date,
            period_type,
            operating_cash_flow,
            investing_cash_flow,
            financing_cash_flow,
            free_cash_flow,
            end_cash_position,
            income_tax_paid,
            interest_paid,
            capital_expenditure
        FROM 
            cash_flows
        WHERE 
            symbol = :symbol
            AND period_type LIKE :period_type
        ORDER BY 
            date DESC
        LIMIT 8
        """

        df = pd.read_sql(
            text(query),
            session.bind,
            params={
                "symbol": symbol,
                "period_type": f"{period_type}%",  # Use wildcard to match both "Annual" and "Quarterly"
            },
        )

        logger.info(f"Loaded {len(df)} cash flow records for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Error loading cash flow for {symbol}: {e}")
        return pd.DataFrame()


def create_bar_chart(
    df,
    x_column,
    y_column,
    title,
    color_increasing="#26A69A",
    color_decreasing="#EF5350",
):
    """Create a bar chart with positive/negative coloring"""
    if df.empty:
        return None

    # Convert dates if needed
    if pd.api.types.is_datetime64_any_dtype(df[x_column]):
        df = df.copy()
        df[x_column] = df[x_column].dt.strftime("%Y-%m-%d")

    # Create color array based on positive/negative values
    colors = [color_increasing if v >= 0 else color_decreasing for v in df[y_column]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[x_column],
            y=df[y_column],
            marker_color=colors,
            text=[f"${x:,.0f}" if pd.notnull(x) else "N/A" for x in df[y_column]],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        height=400,
    )

    return apply_chart_theme(fig)


def create_waterfall_chart(df, title):
    """Create a waterfall chart for financial breakdown"""
    if df.empty:
        return None

    # Get the most recent period
    latest_data = df.iloc[0]

    # Define the waterfall components
    measure = [
        "absolute",
        "relative",
        "relative",
        "total",
        "relative",
        "relative",
        "total",
    ]

    if "total_revenue" in latest_data and "cost_of_revenue" in latest_data:
        # Income statement waterfall
        names = [
            "Revenue",
            "COGS",
            "Gross Profit",
            "Operating Expenses",
            "Operating Income",
            "Net Income",
        ]
        values = [
            (
                latest_data["total_revenue"]
                if pd.notnull(latest_data["total_revenue"])
                else 0
            ),
            (
                -latest_data["cost_of_revenue"]
                if pd.notnull(latest_data["cost_of_revenue"])
                else 0
            ),
            0,  # Placeholder for gross profit total
            (
                -latest_data["operating_expense"]
                if pd.notnull(latest_data["operating_expense"])
                else 0
            ),
            0,  # Placeholder for operating income total
            (
                latest_data["net_income"] - latest_data["operating_income"]
                if pd.notnull(latest_data["net_income"])
                and pd.notnull(latest_data["operating_income"])
                else 0
            ),
        ]
    else:
        # Balance sheet waterfall
        names = ["Assets", "Liabilities", "Equity"]
        values = [
            (
                latest_data["total_assets"]
                if pd.notnull(latest_data["total_assets"])
                else 0
            ),
            (
                -latest_data["total_liabilities"]
                if pd.notnull(latest_data["total_liabilities"])
                else 0
            ),
            0,  # Placeholder for equity total (should equal assets - liabilities)
        ]
        measure = ["absolute", "relative", "total"]

    fig = go.Figure(
        go.Waterfall(
            name=title,
            orientation="v",
            measure=measure,
            x=names,
            textposition="outside",
            text=[f"${x:,.0f}" if pd.notnull(x) and x != 0 else "" for x in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#EF5350"}},  # Red
            increasing={"marker": {"color": "#26A69A"}},  # Green
            totals={"marker": {"color": "#FFC107"}},  # Yellow
        )
    )

    fig.update_layout(
        title=title,
        height=400,
    )

    return apply_chart_theme(fig)


def format_financial_value(value):
    """Format financial values for display"""
    if pd.isnull(value):
        return "N/A"

    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"


def show_financials():
    """Main function to display the Financial Analysis page"""
    st.title("Financial Analysis")

    # Get database session from session state
    try:
        if "db_session" not in st.session_state:
            st.error("Database session not found. Please restart the application.")
            return

        session = st.session_state["db_session"]

    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return

    try:
        # Create a layout with sidebar for selections and main area for financials
        sector_col, stock_col, main_col = st.columns([1, 1, 3])

        # First column: Sector selection
        with sector_col:
            st.markdown("### Sector Selection")

            # Get all available sectors
            sectors = load_sectors(session)

            if not sectors:
                st.warning("No sector data available")
                return

            # Initialize session state if needed
            if "selected_fin_sector" not in st.session_state:
                st.session_state.selected_fin_sector = sectors[0] if sectors else None

            # Create sector selection
            selected_sector = st.selectbox(
                "Select Sector",
                sectors,
                index=(
                    sectors.index(st.session_state.selected_fin_sector)
                    if st.session_state.selected_fin_sector in sectors
                    else 0
                ),
                key="sector_select",
            )

            if selected_sector != st.session_state.selected_fin_sector:
                st.session_state.selected_fin_sector = selected_sector
                if "selected_fin_stock" in st.session_state:
                    del st.session_state.selected_fin_stock
                st.rerun()

        # Second column: Stock selection
        with stock_col:
            st.markdown("### Company Selection")

            if st.session_state.selected_fin_sector:
                # Get stocks in the selected sector
                stocks_df = load_stocks_by_sector(
                    session, st.session_state.selected_fin_sector
                )

                if stocks_df.empty:
                    st.info(
                        f"No stocks found for sector {st.session_state.selected_fin_sector}"
                    )
                    return

                # Default selection if needed
                if (
                    "selected_fin_stock" not in st.session_state
                    or st.session_state.selected_fin_stock
                    not in stocks_df["symbol"].values
                ):
                    st.session_state.selected_fin_stock = stocks_df["symbol"].iloc[0]

                # Create stock selection
                stock_options = [
                    f"{row['symbol']} - {row['company_name']}"
                    for _, row in stocks_df.iterrows()
                ]
                selected_stock_option = st.selectbox(
                    "Select Company",
                    stock_options,
                    index=(
                        [
                            i
                            for i, s in enumerate(stock_options)
                            if s.startswith(st.session_state.selected_fin_stock + " -")
                        ][0]
                        if st.session_state.selected_fin_stock
                        else 0
                    ),
                    key="stock_select",
                )

                # Extract symbol from the selected option
                selected_stock = selected_stock_option.split(" - ")[0]

                if selected_stock != st.session_state.selected_fin_stock:
                    st.session_state.selected_fin_stock = selected_stock
                    st.rerun()

                # Show stock details
                selected_stock_info = stocks_df[
                    stocks_df["symbol"] == st.session_state.selected_fin_stock
                ].iloc[0]

                st.markdown("#### Company Details")
                st.markdown(f"**Symbol:** {selected_stock_info['symbol']}")
                st.markdown(f"**Name:** {selected_stock_info['company_name']}")
                st.markdown(f"**Industry:** {selected_stock_info['industry']}")
                st.markdown(f"**Current Price:** ${selected_stock_info['price']:.2f}")

                # Show earnings dates
                st.markdown("#### Earnings Dates")

                # Previous/Last earnings
                if (
                    "earnings_release_trading_date_fq" in selected_stock_info
                    and pd.notnull(
                        selected_stock_info["earnings_release_trading_date_fq"]
                    )
                ):
                    st.markdown(
                        f"**Last Earnings:** {pd.to_datetime(selected_stock_info['earnings_release_trading_date_fq']).strftime('%Y-%m-%d')}"
                    )
                else:
                    st.markdown("**Last Earnings:** N/A")

                # Next earnings
                if (
                    "earnings_release_next_trading_date_fq" in selected_stock_info
                    and pd.notnull(
                        selected_stock_info["earnings_release_next_trading_date_fq"]
                    )
                ):
                    st.markdown(
                        f"**Next Earnings:** {pd.to_datetime(selected_stock_info['earnings_release_next_trading_date_fq']).strftime('%Y-%m-%d')}"
                    )
                else:
                    st.markdown("**Next Earnings:** N/A")

                # Period type selection
                period_type = st.radio(
                    "Data Frequency",
                    ["Annual", "Quarterly"],
                    horizontal=True,
                    key="period_type",
                )
            else:
                st.info("Please select a sector first")

        # Main area: Financial data visualization
        with main_col:
            if (
                "selected_fin_stock" in st.session_state
                and st.session_state.selected_fin_stock
            ):
                st.markdown(
                    f"### {st.session_state.selected_fin_stock} Financial Analysis"
                )

                # Create tabs for different financial statements
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Income Statement", "Balance Sheet", "Cash Flow", "Key Metrics"]
                )

                # Load financial data
                income_data = load_income_statement(
                    session, st.session_state.selected_fin_stock, period_type
                )
                balance_data = load_balance_sheet(
                    session, st.session_state.selected_fin_stock, period_type
                )
                cashflow_data = load_cash_flow(
                    session, st.session_state.selected_fin_stock, period_type
                )

                # Income Statement tab
                with tab1:
                    if income_data.empty:
                        st.info(
                            f"No income statement data available for {st.session_state.selected_fin_stock}"
                        )
                    else:
                        # Create two rows of charts
                        st.markdown("#### Income Statement Analysis")

                        # First row: Revenue chart
                        revenue_fig = create_bar_chart(
                            income_data,
                            "date",
                            "total_revenue",
                            f"{st.session_state.selected_fin_stock} Total Revenue ({period_type})",
                        )
                        st.plotly_chart(revenue_fig, use_container_width=True)

                        # Second row: Net Income chart
                        income_fig = create_bar_chart(
                            income_data,
                            "date",
                            "net_income",
                            f"{st.session_state.selected_fin_stock} Net Income ({period_type})",
                        )
                        st.plotly_chart(income_fig, use_container_width=True)

                        # Third row: Waterfall chart for most recent period
                        waterfall_fig = create_waterfall_chart(
                            income_data,
                            f"{st.session_state.selected_fin_stock} Income Breakdown ({income_data['date'].iloc[0]})",
                        )
                        st.plotly_chart(waterfall_fig, use_container_width=True)

                        # Table view of the data
                        st.markdown("#### Income Statement Data")
                        display_cols = [
                            "date",
                            "period_type",
                            "total_revenue",
                            "cost_of_revenue",
                            "gross_profit",
                            "operating_income",
                            "net_income",
                        ]

                        # Format the table data
                        display_df = income_data[display_cols].copy()
                        for col in display_cols:
                            if col not in ["date", "period_type"]:
                                display_df[col] = display_df[col].apply(
                                    format_financial_value
                                )

                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                # Balance Sheet tab
                with tab2:
                    if balance_data.empty:
                        st.info(
                            f"No balance sheet data available for {st.session_state.selected_fin_stock}"
                        )
                    else:
                        # Create charts for balance sheet
                        st.markdown("#### Balance Sheet Analysis")

                        # First row: Assets chart
                        assets_fig = create_bar_chart(
                            balance_data,
                            "date",
                            "total_assets",
                            f"{st.session_state.selected_fin_stock} Total Assets ({period_type})",
                        )
                        st.plotly_chart(assets_fig, use_container_width=True)

                        # Create columns for side-by-side charts
                        col1, col2 = st.columns(2)

                        with col1:
                            # Liabilities chart
                            liab_fig = create_bar_chart(
                                balance_data,
                                "date",
                                "total_liabilities",
                                f"Total Liabilities ({period_type})",
                            )
                            st.plotly_chart(liab_fig, use_container_width=True)

                        with col2:
                            # Equity chart
                            equity_fig = create_bar_chart(
                                balance_data,
                                "date",
                                "total_equity",
                                f"Total Equity ({period_type})",
                            )
                            st.plotly_chart(equity_fig, use_container_width=True)

                        # Table view of the data
                        st.markdown("#### Balance Sheet Data")
                        display_cols = [
                            "date",
                            "period_type",
                            "total_assets",
                            "total_liabilities",
                            "total_equity",
                            "total_debt",
                            "working_capital",
                        ]

                        # Format the table data
                        display_df = balance_data[display_cols].copy()
                        for col in display_cols:
                            if col not in ["date", "period_type"]:
                                display_df[col] = display_df[col].apply(
                                    format_financial_value
                                )

                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                # Cash Flow tab
                with tab3:
                    if cashflow_data.empty:
                        st.info(
                            f"No cash flow data available for {st.session_state.selected_fin_stock}"
                        )
                    else:
                        # Create charts for cash flow
                        st.markdown("#### Cash Flow Analysis")

                        # First row: Operating Cash Flow
                        op_cf_fig = create_bar_chart(
                            cashflow_data,
                            "date",
                            "operating_cash_flow",
                            f"{st.session_state.selected_fin_stock} Operating Cash Flow ({period_type})",
                        )
                        st.plotly_chart(op_cf_fig, use_container_width=True)

                        # Create columns for side-by-side charts
                        col1, col2 = st.columns(2)

                        with col1:
                            # Free Cash Flow chart
                            fcf_fig = create_bar_chart(
                                cashflow_data,
                                "date",
                                "free_cash_flow",
                                f"Free Cash Flow ({period_type})",
                            )
                            st.plotly_chart(fcf_fig, use_container_width=True)

                        with col2:
                            # Capex chart
                            capex_fig = create_bar_chart(
                                cashflow_data,
                                "date",
                                "capital_expenditure",
                                f"Capital Expenditure ({period_type})",
                            )
                            st.plotly_chart(capex_fig, use_container_width=True)

                        # Table view of the data
                        st.markdown("#### Cash Flow Data")
                        display_cols = [
                            "date",
                            "period_type",
                            "operating_cash_flow",
                            "investing_cash_flow",
                            "financing_cash_flow",
                            "free_cash_flow",
                            "capital_expenditure",
                        ]

                        # Format the table data
                        display_df = cashflow_data[display_cols].copy()
                        for col in display_cols:
                            if col not in ["date", "period_type"]:
                                display_df[col] = display_df[col].apply(
                                    format_financial_value
                                )

                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                # Key Metrics tab
                with tab4:
                    st.markdown("#### Key Financial Metrics")

                    if not income_data.empty and not balance_data.empty:
                        # Calculate key metrics from most recent period
                        latest_income = income_data.iloc[0]
                        latest_balance = balance_data.iloc[0]
                        latest_cashflow = (
                            cashflow_data.iloc[0] if not cashflow_data.empty else None
                        )

                        # Create metrics
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # Profitability metrics
                            st.markdown("##### Profitability")

                            if (
                                pd.notnull(latest_income["gross_profit"])
                                and pd.notnull(latest_income["total_revenue"])
                                and latest_income["total_revenue"] != 0
                            ):
                                gross_margin = (
                                    latest_income["gross_profit"]
                                    / latest_income["total_revenue"]
                                ) * 100
                                st.metric("Gross Margin", f"{gross_margin:.2f}%")
                            else:
                                st.metric("Gross Margin", "N/A")

                            if (
                                pd.notnull(latest_income["operating_income"])
                                and pd.notnull(latest_income["total_revenue"])
                                and latest_income["total_revenue"] != 0
                            ):
                                operating_margin = (
                                    latest_income["operating_income"]
                                    / latest_income["total_revenue"]
                                ) * 100
                                st.metric(
                                    "Operating Margin", f"{operating_margin:.2f}%"
                                )
                            else:
                                st.metric("Operating Margin", "N/A")

                            if (
                                pd.notnull(latest_income["net_income"])
                                and pd.notnull(latest_income["total_revenue"])
                                and latest_income["total_revenue"] != 0
                            ):
                                net_margin = (
                                    latest_income["net_income"]
                                    / latest_income["total_revenue"]
                                ) * 100
                                st.metric("Net Margin", f"{net_margin:.2f}%")
                            else:
                                st.metric("Net Margin", "N/A")

                        with col2:
                            # Efficiency metrics
                            st.markdown("##### Efficiency")

                            if (
                                pd.notnull(latest_income["total_revenue"])
                                and pd.notnull(latest_balance["total_assets"])
                                and latest_balance["total_assets"] != 0
                            ):
                                asset_turnover = (
                                    latest_income["total_revenue"]
                                    / latest_balance["total_assets"]
                                )
                                st.metric("Asset Turnover", f"{asset_turnover:.2f}x")
                            else:
                                st.metric("Asset Turnover", "N/A")

                            if (
                                latest_cashflow is not None
                                and pd.notnull(latest_cashflow["free_cash_flow"])
                                and pd.notnull(latest_income["net_income"])
                                and latest_income["net_income"] != 0
                            ):
                                fcf_to_net_income = (
                                    latest_cashflow["free_cash_flow"]
                                    / latest_income["net_income"]
                                )
                                st.metric(
                                    "FCF to Net Income", f"{fcf_to_net_income:.2f}x"
                                )
                            else:
                                st.metric("FCF to Net Income", "N/A")

                            if (
                                pd.notnull(latest_income["operating_income"])
                                and pd.notnull(latest_balance["total_assets"])
                                and latest_balance["total_assets"] != 0
                            ):
                                roa = (
                                    latest_income["net_income"]
                                    / latest_balance["total_assets"]
                                ) * 100
                                st.metric("Return on Assets", f"{roa:.2f}%")
                            else:
                                st.metric("Return on Assets", "N/A")

                        with col3:
                            # Leverage metrics
                            st.markdown("##### Leverage")

                            if (
                                pd.notnull(latest_balance["total_liabilities"])
                                and pd.notnull(latest_balance["total_equity"])
                                and latest_balance["total_equity"] != 0
                            ):
                                debt_to_equity = (
                                    latest_balance["total_liabilities"]
                                    / latest_balance["total_equity"]
                                )
                                st.metric("Debt to Equity", f"{debt_to_equity:.2f}x")
                            else:
                                st.metric("Debt to Equity", "N/A")

                            if (
                                pd.notnull(latest_balance["total_assets"])
                                and pd.notnull(latest_balance["total_equity"])
                                and latest_balance["total_equity"] != 0
                            ):
                                equity_multiplier = (
                                    latest_balance["total_assets"]
                                    / latest_balance["total_equity"]
                                )
                                st.metric(
                                    "Equity Multiplier", f"{equity_multiplier:.2f}x"
                                )
                            else:
                                st.metric("Equity Multiplier", "N/A")

                            if (
                                latest_cashflow is not None
                                and pd.notnull(latest_cashflow["operating_cash_flow"])
                                and pd.notnull(latest_balance["total_debt"])
                                and latest_balance["total_debt"] != 0
                            ):
                                ocf_to_debt = (
                                    latest_cashflow["operating_cash_flow"]
                                    / latest_balance["total_debt"]
                                )
                                st.metric("OCF to Debt", f"{ocf_to_debt:.2f}x")
                            else:
                                st.metric("OCF to Debt", "N/A")

                        # Historical metrics over time
                        st.markdown("#### Historical Performance")

                        # Create trend charts for key metrics
                        if not income_data.empty:
                            # Calculate margin values over time
                            trend_data = income_data.copy()
                            trend_data["gross_margin"] = (
                                trend_data["gross_profit"] / trend_data["total_revenue"]
                            ) * 100
                            trend_data["operating_margin"] = (
                                trend_data["operating_income"]
                                / trend_data["total_revenue"]
                            ) * 100
                            trend_data["net_margin"] = (
                                trend_data["net_income"] / trend_data["total_revenue"]
                            ) * 100

                            # Create a line chart
                            fig = go.Figure()
                            fig.add_trace(
                                go.Scatter(
                                    x=trend_data["date"],
                                    y=trend_data["gross_margin"],
                                    mode="lines+markers",
                                    name="Gross Margin",
                                    line=dict(color="#26A69A", width=2),
                                )
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=trend_data["date"],
                                    y=trend_data["operating_margin"],
                                    mode="lines+markers",
                                    name="Operating Margin",
                                    line=dict(color="#FFC107", width=2),
                                )
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=trend_data["date"],
                                    y=trend_data["net_margin"],
                                    mode="lines+markers",
                                    name="Net Margin",
                                    line=dict(color="#EF5350", width=2),
                                )
                            )

                            fig.update_layout(
                                title=f"{st.session_state.selected_fin_stock} Margin Analysis ({period_type})",
                                xaxis_title="Date",
                                yaxis_title="Margin (%)",
                                height=400,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                ),
                            )

                            fig = apply_chart_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Insufficient data to calculate key metrics")

            else:
                st.info("Please select a company to view financial data")

    except Exception as e:
        st.error(f"Error in Financial Analysis: {e}")
        logger.exception("Error in Financial Analysis")


if __name__ == "__main__":
    show_financials()

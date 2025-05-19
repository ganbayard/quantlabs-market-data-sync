import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def load_eligible_stocks(session):
    """Load stocks eligible for the QuantLabs ETF based on risk metrics"""
    try:
        from sqlalchemy import text

        query = """
        SELECT 
            symbol,
            price,
            risk_type,
            average_score,
            volatility_3yr_avg,
            beta_3yr_avg,
            max_drawdown_3yr_avg,
            volatility_year3,
            volatility_year2,
            volatility_year1
        FROM 
            equity_risk_profile
        WHERE 
            max_drawdown_3yr_avg BETWEEN -0.21 AND -0.12
            AND price > 0
            AND beta_3yr_avg IS NOT NULL
            AND volatility_3yr_avg IS NOT NULL
        ORDER BY 
            symbol
        """

        df = pd.read_sql(text(query), session.bind)
        logger.info(f"Loaded {len(df)} eligible stocks for ETF")
        return df
    except Exception as e:
        logger.error(f"Error loading eligible stocks: {e}")
        st.error(f"Error loading eligible stocks: {e}")
        return pd.DataFrame()


def calculate_etf_weights(eligible_stocks):
    """Calculate optimal weights for the ETF"""
    if eligible_stocks.empty:
        return pd.DataFrame()

    # Create a copy to avoid modifying original
    df = eligible_stocks.copy()

    # Use the existing scores if available, otherwise calculate them
    if "volatility_score" not in df.columns:
        df["volatility_score"] = 1 / df["volatility_3yr_avg"]

    if "beta_score" not in df.columns:
        # For beta, we prefer values close to 1, so calculate distance from 1
        df["beta_distance"] = (df["beta_3yr_avg"] - 1).abs()
        df["beta_score"] = 1 / (1 + df["beta_distance"])

    if "drawdown_score" not in df.columns:
        # Convert drawdown to positive score (less negative is better)
        df["drawdown_score"] = 1 / df["max_drawdown_3yr_avg"].abs()

    # Combine scores (equal weight to all factors)
    df["composite_score"] = (
        df["volatility_score"] + df["beta_score"] + df["drawdown_score"]
    ) / 3

    # Initial weights based on composite score
    total_score = df["composite_score"].sum()
    df["initial_weight"] = df["composite_score"] / total_score

    # Apply cap of 2.1% per stock
    MAX_WEIGHT = 0.021  # 2.1%
    df["capped_weight"] = np.minimum(df["initial_weight"], MAX_WEIGHT)

    # Redistribute excess weight
    total_capped = df["capped_weight"].sum()
    excess = 1 - total_capped

    # If there's excess weight to distribute
    if excess > 0.001:  # Check if excess is significant
        # Identify stocks below cap
        below_cap = df[df["initial_weight"] < MAX_WEIGHT].copy()

        if not below_cap.empty:
            # Redistribute proportionally to stocks below cap
            below_cap_score_sum = below_cap["composite_score"].sum()

            for idx, row in df.iterrows():
                if row["initial_weight"] < MAX_WEIGHT:
                    # Add proportional share of excess
                    df.at[idx, "capped_weight"] += excess * (
                        row["composite_score"] / below_cap_score_sum
                    )

    # Final normalization to ensure sum is exactly 1.0
    df["weight"] = df["capped_weight"] / df["capped_weight"].sum()

    # Calculate number of shares based on a hypothetical $10M ETF
    ETF_SIZE = 10000000  # $10M ETF
    df["allocation"] = df["weight"] * ETF_SIZE
    df["shares"] = (df["allocation"] / df["price"]).round()

    # Recalculate actual weights based on rounded shares
    df["actual_value"] = df["shares"] * df["price"]
    total_value = df["actual_value"].sum()
    df["final_weight"] = df["actual_value"] / total_value

    # Sort by weight descending
    result = df.sort_values(by="final_weight", ascending=False).reset_index(drop=True)

    # Select only needed columns
    result = result[
        [
            "symbol",
            "price",
            "risk_type",
            "beta_3yr_avg",
            "volatility_3yr_avg",
            "max_drawdown_3yr_avg",
            "final_weight",
            "shares",
            "actual_value",
        ]
    ]

    return result


def create_etf_composition_chart(etf_composition):
    """Create visual charts for ETF composition"""
    if etf_composition.empty:
        return None

    # Create figure with subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("ETF Weight Distribution", "Top 10 Holdings"),
    )

    # Weight distribution pie chart
    fig.add_trace(
        go.Pie(
            labels=etf_composition["symbol"],
            values=etf_composition["final_weight"],
            textinfo="label+percent",
            hoverinfo="label+percent+value",
            marker=dict(
                colors=px.colors.qualitative.Plotly,
                line=dict(color="rgba(255, 255, 255, 0.5)", width=1),
            ),
        ),
        row=1,
        col=1,
    )

    # Top 10 holdings bar chart
    top10 = etf_composition.head(10).copy()
    top10["weight_percent"] = top10["final_weight"] * 100  # Convert to percentage

    fig.add_trace(
        go.Bar(
            x=top10["symbol"],
            y=top10["weight_percent"],
            text=[f"{w:.2f}%" for w in top10["weight_percent"]],
            textposition="auto",
            marker_color=px.colors.qualitative.Plotly[:10],
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(title="QuantLabs ETF Composition", height=500, showlegend=False)

    fig.update_yaxes(title_text="Weight (%)", row=1, col=2)

    return apply_chart_theme(fig)


def create_risk_metrics_chart(etf_composition):
    """Create chart comparing risk metrics"""
    if etf_composition.empty:
        return None

    # Calculate average values weighted by allocation
    avg_beta = (etf_composition["beta_3yr_avg"] * etf_composition["final_weight"]).sum()
    avg_volatility = (
        etf_composition["volatility_3yr_avg"] * etf_composition["final_weight"]
    ).sum()
    avg_drawdown = (
        etf_composition["max_drawdown_3yr_avg"] * etf_composition["final_weight"]
    ).sum()

    # Create figure
    fig = go.Figure()

    # Add individual stocks as scatter points
    fig.add_trace(
        go.Scatter(
            x=etf_composition["beta_3yr_avg"],
            y=etf_composition["volatility_3yr_avg"],
            mode="markers",
            marker=dict(
                size=etf_composition["final_weight"]
                * 500,  # Size proportional to weight
                color=etf_composition["max_drawdown_3yr_avg"],
                colorscale="RdYlGn_r",
                colorbar=dict(title="Max Drawdown"),
                line=dict(width=1, color="rgba(255, 255, 255, 0.5)"),
            ),
            text=etf_composition["symbol"],
            hovertemplate="<b>%{text}</b><br>Beta: %{x:.2f}<br>Volatility: %{y:.2f}<br>Max Drawdown: %{marker.color:.2f}<extra></extra>",
        )
    )

    # Add ETF average as a star
    fig.add_trace(
        go.Scatter(
            x=[avg_beta],
            y=[avg_volatility],
            mode="markers",
            marker=dict(
                symbol="star",
                size=20,
                color="yellow",
                line=dict(width=1, color="black"),
            ),
            name="ETF Average",
            hovertemplate="<b>ETF Average</b><br>Beta: %{x:.2f}<br>Volatility: %{y:.2f}<br>Max Drawdown: "
            + f"{avg_drawdown:.2f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title="QuantLabs ETF Risk Profile",
        xaxis_title="Beta (3-Year Average)",
        yaxis_title="Volatility (3-Year Average)",
        height=500,
        hovermode="closest",
    )

    return apply_chart_theme(fig)


def show_quantlabs_etf():
    """Main function to display the QuantLabs ETF page"""
    st.title("QuantLabs ETF")

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
        # Show ETF description
        st.markdown(
            """
        ### QuantLabs ETF: Advanced Risk-Managed Portfolio
        
        The QuantLabs ETF is designed to provide exposure to equities with optimal risk-return characteristics.
        This ETF specifically targets stocks with controlled drawdowns (between -12% and -21%) while balancing
        volatility and beta exposure for a more resilient portfolio.
        
        **Key Features:**
        - Controlled maximum drawdown exposure
        - Volatility-optimized weightings
        - Beta-balanced portfolio construction
        - Maximum single-stock exposure of 2.1% for diversification
        """
        )

        # Load eligible stocks
        with st.spinner("Constructing ETF portfolio..."):
            eligible_stocks = load_eligible_stocks(session)

            if eligible_stocks.empty:
                st.error(
                    "No eligible stocks found for ETF construction. Please check your database."
                )
                return

            # Calculate ETF weights
            etf_composition = calculate_etf_weights(eligible_stocks)

        # Display ETF summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Number of Holdings", len(etf_composition))

        with col2:
            avg_beta = (
                etf_composition["beta_3yr_avg"] * etf_composition["final_weight"]
            ).sum()
            st.metric("Portfolio Beta", f"{avg_beta:.2f}")

        with col3:
            avg_volatility = (
                etf_composition["volatility_3yr_avg"] * etf_composition["final_weight"]
            ).sum()
            st.metric("Portfolio Volatility", f"{avg_volatility:.2f}")

        with col4:
            avg_drawdown = (
                etf_composition["max_drawdown_3yr_avg"]
                * etf_composition["final_weight"]
            ).sum()
            st.metric("Avg Max Drawdown", f"{avg_drawdown:.2f}")

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Holdings", "Visual Analysis", "Risk Profile"])

        with tab1:
            # Display the ETF composition in a sortable table
            st.markdown("### QuantLabs ETF Holdings")

            # Format data for display
            display_df = etf_composition.copy()
            display_df["weight_percent"] = (display_df["final_weight"] * 100).round(2)
            display_df["dollar_value"] = display_df["actual_value"].round(2)

            # Rename columns for clarity
            display_df = display_df.rename(
                columns={
                    "symbol": "Symbol",
                    "price": "Price ($)",
                    "risk_type": "Risk Profile",
                    "beta_3yr_avg": "Beta",
                    "volatility_3yr_avg": "Volatility",
                    "max_drawdown_3yr_avg": "Max Drawdown",
                    "weight_percent": "Weight (%)",
                    "shares": "Shares",
                    "dollar_value": "Allocation ($)",
                }
            )

            # Select columns to display
            display_cols = [
                "Symbol",
                "Price ($)",
                "Risk Profile",
                "Weight (%)",
                "Beta",
                "Volatility",
                "Max Drawdown",
                "Shares",
                "Allocation ($)",
            ]

            st.dataframe(display_df[display_cols], height=500)

            # Display total value
            total_value = display_df["Allocation ($)"].sum()
            st.markdown(f"**Total ETF Value:** ${total_value:,.2f}")

            # Add export button
            csv = display_df[display_cols].to_csv(index=False)
            st.download_button(
                label="Export Holdings to CSV",
                data=csv,
                file_name="quantlabs_etf_holdings.csv",
                mime="text/csv",
            )

        with tab2:
            # Show visual analysis
            st.markdown("### ETF Composition Analysis")

            # Create and show charts
            composition_chart = create_etf_composition_chart(etf_composition)
            if composition_chart:
                st.plotly_chart(composition_chart, use_container_width=True)

            # Risk distribution by sector
            st.markdown("### Risk Type Distribution")

            # Summarize by risk type
            risk_summary = (
                display_df.groupby("Risk Profile")
                .agg(
                    count=("Symbol", "count"),
                    total_weight=("Weight (%)", "sum"),
                    avg_beta=("Beta", "mean"),
                    avg_volatility=("Volatility", "mean"),
                )
                .reset_index()
            )

            # Create bar chart
            risk_fig = px.bar(
                risk_summary,
                x="Risk Profile",
                y="total_weight",
                color="Risk Profile",
                hover_data=["count", "avg_beta", "avg_volatility"],
                labels={
                    "total_weight": "Total Weight (%)",
                    "count": "Number of Stocks",
                },
            )
            risk_fig.update_layout(title="ETF Allocation by Risk Profile")
            st.plotly_chart(apply_chart_theme(risk_fig), use_container_width=True)

        with tab3:
            # Show risk profile
            st.markdown("### ETF Risk Profile")

            risk_chart = create_risk_metrics_chart(etf_composition)
            if risk_chart:
                st.plotly_chart(risk_chart, use_container_width=True)

            # Risk commentary
            st.markdown(
                """
            **Risk Analysis:**
            
            The chart above shows each holding's position in risk space:
            - **X-axis**: Beta (market sensitivity)
            - **Y-axis**: Volatility (price fluctuation)
            - **Color**: Maximum Drawdown (worst historical decline)
            - **Size**: Weight in portfolio
            
            The ETF's overall risk profile (yellow star) represents the weighted average of all holdings.
            """
            )

    except Exception as e:
        st.error(f"Error in QuantLabs ETF page: {e}")
        st.exception(e)
        logger.exception("Error in QuantLabs ETF page")


if __name__ == "__main__":
    show_quantlabs_etf()

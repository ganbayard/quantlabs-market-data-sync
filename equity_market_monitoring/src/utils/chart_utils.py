import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

# Constants for the application
RISK_COLORS = {
    "Conservative": "#3fbf91",  # Keep the green for Conservative
    "Moderate": "#BB972C",  # Deep golden yellow for Moderate
    "Aggressive": "#B8352C",  # Keep the red for Aggressive
}


def apply_chart_theme(fig) -> go.Figure:
    """Apply consistent theming to Plotly charts"""
    # Use the global theme setting
    is_dark = st.session_state.get("theme", "dark") == "dark"

    background_color = "#262730" if is_dark else "#FFFFFF"
    text_color = "#FAFAFA" if is_dark else "#262730"
    grid_color = "rgba(255, 255, 255, 0.1)" if is_dark else "rgba(0, 0, 0, 0.1)"

    fig.update_layout(
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font_color=text_color,
        title_font_size=18,
        legend_title_font_size=14,
        legend_font_size=12,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            showline=True,
            linewidth=1,
            linecolor=grid_color,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            showline=True,
            linewidth=1,
            linecolor=grid_color,
        ),
    )

    return fig


def display_metric_with_tooltip(
    label: str,
    value: str,
    delta: str = None,
    tooltip: str = None,
    delta_color: str = "normal",
) -> None:
    """Display a metric with an optional tooltip"""
    col1, col2 = st.columns([10, 1])

    with col1:
        if delta:
            st.metric(label=label, value=value, delta=delta, delta_color=delta_color)
        else:
            st.metric(label=label, value=value)

    with col2:
        if tooltip:
            st.markdown(
                f"""
            <div class="tooltip">ℹ️
                <span class="tooltiptext">{tooltip}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )


def display_notification(message: str, type: str = "info") -> None:
    """Display a notification banner"""
    st.markdown(
        f'<div class="notification notification-{type}">{message}</div>',
        unsafe_allow_html=True,
    )


def format_delta_for_metric(
    current: float, previous: float, inverse: bool = False
) -> Tuple[str, str]:
    """
    Format the delta value and determine color direction for metrics

    Args:
        current: Current value
        previous: Previous value
        inverse: Whether lower values are better (True) or worse (False)

    Returns:
        Tuple of (formatted_delta, delta_color)
    """
    if pd.isna(current) or pd.isna(previous):
        return "", "normal"

    delta = current - previous
    delta_pct = delta / abs(previous) if previous != 0 else 0

    # Format based on the type of value
    if abs(current) < 10:  # Small values, likely percentages
        formatted_delta = f"{delta:.2%}"
    else:  # Larger values
        formatted_delta = f"{delta:.2f}"

    # Determine color direction
    if delta == 0:
        delta_color = "normal"
    elif inverse:
        # For metrics where lower is better (like drawdown, volatility)
        delta_color = "normal" if delta < 0 else "inverse"
    else:
        # For metrics where higher is better (like returns)
        delta_color = "normal" if delta > 0 else "inverse"

    return formatted_delta, delta_color


def create_risk_metrics_radar_chart(df: pd.DataFrame, is_etf: bool = True) -> go.Figure:
    """
    Create a radar chart showing risk metrics by risk type

    Args:
        df: DataFrame with risk profile data
        is_etf: Whether to filter for ETFs or stocks

    Returns:
        Plotly figure
    """
    # Filter data
    filtered_df = df[df["is_etf"] == is_etf]

    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No {'ETF' if is_etf else 'Stock'} data available")
        return fig

    # Calculate average metrics by risk type
    metrics = [
        "volatility_3yr_avg",
        "beta_3yr_avg",
        "max_drawdown_3yr_avg",
        "adr_3yr_avg",
    ]
    metric_labels = ["Volatility", "Beta", "Max Drawdown", "ADR"]

    # Group by risk type and calculate means
    radar_data = filtered_df.groupby("risk_type")[metrics].mean().reset_index()

    # Create radar chart
    fig = go.Figure()

    for risk_type in radar_data["risk_type"].unique():
        risk_row = radar_data[radar_data["risk_type"] == risk_type].iloc[0]

        # Convert max_drawdown to positive for better visualization
        values = [
            risk_row["volatility_3yr_avg"],
            risk_row["beta_3yr_avg"],
            abs(risk_row["max_drawdown_3yr_avg"]),  # Convert to positive
            risk_row["adr_3yr_avg"],
        ]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metric_labels,
                fill="toself",
                name=risk_type,
                line_color=RISK_COLORS[risk_type],
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, showticklabels=True, showline=True, tickfont_size=10
            ),
            angularaxis=dict(showticklabels=True, showline=True, tickfont_size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=f"{'ETF' if is_etf else 'Stock'} Risk Metrics by Risk Type",
    )

    # Apply themed styling
    fig = apply_chart_theme(fig)

    return fig


def create_focus_risk_chart(df: pd.DataFrame, is_etf: bool = True) -> go.Figure:
    """
    Create a sunburst chart showing risk distribution by focus

    Args:
        df: DataFrame with risk profile data
        is_etf: Whether to filter for ETFs or stocks

    Returns:
        Plotly figure
    """
    # Filter data
    filtered_df = df[df["is_etf"] == is_etf]

    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No {'ETF' if is_etf else 'Stock'} data available")
        return fig

    # Group data by focus and risk_type
    focus_risk_df = (
        filtered_df.groupby(["focus", "risk_type"]).size().reset_index(name="count")
    )

    # Create sunburst chart
    fig = px.sunburst(
        focus_risk_df,
        path=["focus", "risk_type"],
        values="count",
        color="risk_type",
        color_discrete_map=RISK_COLORS,
        title=f"{'ETF' if is_etf else 'Stock'} Risk Distribution by Focus",
    )

    # Apply themed styling
    fig = apply_chart_theme(fig)

    return fig


def create_asset_class_risk_chart(df: pd.DataFrame, is_etf: bool = True) -> go.Figure:
    """
    Create a sunburst chart showing risk distribution by asset class

    Args:
        df: DataFrame with risk profile data
        is_etf: Whether to filter for ETFs or stocks

    Returns:
        Plotly figure
    """
    # Filter data
    filtered_df = df[df["is_etf"] == is_etf]

    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No {'ETF' if is_etf else 'Stock'} data available")
        return fig

    # Group data by asset_class and risk_type
    asset_risk_df = (
        filtered_df.groupby(["asset_class", "risk_type"])
        .size()
        .reset_index(name="count")
    )

    # Create sunburst chart
    fig = px.sunburst(
        asset_risk_df,
        path=["asset_class", "risk_type"],
        values="count",
        color="risk_type",
        color_discrete_map=RISK_COLORS,
        title=f"{'ETF' if is_etf else 'Stock'} Risk Distribution by Asset Class",
    )

    # Apply themed styling
    fig = apply_chart_theme(fig)

    return fig


def create_drawdown_time_chart(df: pd.DataFrame, is_etf: bool = True) -> go.Figure:
    """
    Create a bar chart showing drawdown over time for different risk categories

    Args:
        df: DataFrame with risk profile data
        is_etf: Whether to filter for ETFs or stocks

    Returns:
        Plotly figure
    """
    # Filter data
    filtered_df = df[df["is_etf"] == is_etf]

    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No {'ETF' if is_etf else 'Stock'} data available")
        return fig

    # Prepare data for visualization
    years = ["year1", "year2", "year3"]
    year_labels = ["3 Years Ago", "2 Years Ago", "Last Year"]

    # Calculate average drawdown by risk type and year
    agg_data = []
    for risk_type in filtered_df["risk_type"].unique():
        risk_df = filtered_df[filtered_df["risk_type"] == risk_type]
        for i, year in enumerate(years):
            avg_dd = risk_df[f"max_drawdown_{year}"].mean()
            if not pd.isna(avg_dd):
                agg_data.append(
                    {
                        "risk_type": risk_type,
                        "year": year_labels[i],
                        "avg_drawdown": avg_dd,
                    }
                )

    if not agg_data:
        fig = go.Figure()
        fig.update_layout(
            title=f"No drawdown data available for {'ETFs' if is_etf else 'Stocks'}"
        )
        return fig

    agg_df = pd.DataFrame(agg_data)

    # Create bar chart
    fig = px.bar(
        agg_df,
        x="year",
        y="avg_drawdown",
        color="risk_type",
        color_discrete_map=RISK_COLORS,
        barmode="group",
        title=f"{'ETF' if is_etf else 'Stock'} Average Max Drawdown by Risk Type",
        labels={"avg_drawdown": "Average Max Drawdown", "year": "Time Period"},
    )

    # Format y-axis as percentage
    fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
    fig.update_yaxes(tickformat=".0%")

    # Apply themed styling
    fig = apply_chart_theme(fig)

    return fig

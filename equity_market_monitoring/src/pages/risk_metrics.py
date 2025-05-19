import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.database.db_config import get_db_connection
from src.database.models import SymbolFields, EquityRiskProfile

def show_risk_metrics():
    st.title("Risk Metrics")
    
    # Connect to database
    db_config, engine, session = get_db_connection(st.session_state.get("db_type", "sqlite"))
    
    try:
        # Get risk profile data from database
        risk_profiles = session.query(EquityRiskProfile).all()
        
        # Create filters
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter by asset type
            asset_types = ["All", "Stocks", "ETFs"]
            selected_asset_type = st.selectbox("Asset Type", asset_types, index=0)
        
        with col2:
            # Filter by risk type
            risk_types = ["All", "Conservative", "Moderate", "Aggressive"]
            selected_risk_type = st.selectbox("Risk Type", risk_types, index=0)
        
        # Apply filters to risk profiles
        filtered_profiles = risk_profiles
        
        if selected_asset_type == "Stocks":
            filtered_profiles = [profile for profile in filtered_profiles if not profile.is_etf]
        elif selected_asset_type == "ETFs":
            filtered_profiles = [profile for profile in filtered_profiles if profile.is_etf]
        
        if selected_risk_type != "All":
            filtered_profiles = [profile for profile in filtered_profiles if profile.risk_type == selected_risk_type]
        
        # Create sample risk metrics data
        # In a real application, this would come from the filtered_profiles
        
        # Risk metrics data
        risk_data = {
            "Symbol": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "SPY", "QQQ", "IWM", "AAXJ"],
            "Name": ["Apple Inc.", "Microsoft Corp.", "Amazon.com Inc.", "Alphabet Inc.", "Meta Platforms Inc.", "Tesla Inc.", "SPDR S&P 500 ETF", "Invesco QQQ Trust", "iShares Russell 2000 ETF", "iShares MSCI All Country Asia ex Japan ETF"],
            "Type": ["Stock", "Stock", "Stock", "Stock", "Stock", "Stock", "ETF", "ETF", "ETF", "ETF"],
            "Risk Type": ["Moderate", "Conservative", "Moderate", "Conservative", "Aggressive", "Aggressive", "Conservative", "Moderate", "Aggressive", "Aggressive"],
            "Volatility": [18.5, 15.2, 25.3, 16.8, 28.7, 42.1, 12.5, 18.9, 22.3, 24.5],
            "Beta": [1.2, 0.9, 1.3, 1.1, 1.5, 1.8, 1.0, 1.2, 1.4, 1.3],
            "Max Drawdown": [-25.3, -18.7, -32.1, -20.5, -38.2, -45.6, -15.2, -22.8, -28.5, -30.2],
            "Avg Daily Range": [1.8, 1.5, 2.2, 1.7, 2.5, 3.2, 1.2, 1.7, 2.0, 2.1]
        }
        
        risk_df = pd.DataFrame(risk_data)
        
        # Apply filters to DataFrame
        if selected_asset_type == "Stocks":
            risk_df = risk_df[risk_df["Type"] == "Stock"]
        elif selected_asset_type == "ETFs":
            risk_df = risk_df[risk_df["Type"] == "ETF"]
        
        if selected_risk_type != "All":
            risk_df = risk_df[risk_df["Risk Type"] == selected_risk_type]
        
        # Create tabs for different analyses
        tab1, tab2 = st.tabs(["Risk Metrics Table", "Risk Visualization"])
        
        with tab1:
            st.subheader("Risk Metrics")
            
            # Display risk metrics table
            st.dataframe(
                risk_df.style.format({
                    "Volatility": "{:.1f}%",
                    "Beta": "{:.2f}",
                    "Max Drawdown": "{:.1f}%",
                    "Avg Daily Range": "{:.1f}%"
                }).background_gradient(
                    subset=["Volatility", "Beta", "Max Drawdown", "Avg Daily Range"],
                    cmap="RdYlGn_r"
                ),
                use_container_width=True
            )
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            
            # Calculate statistics by risk type
            risk_stats = risk_df.groupby("Risk Type")[["Volatility", "Beta", "Max Drawdown", "Avg Daily Range"]].mean().reset_index()
            
            # Create metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Volatility", f"{risk_df['Volatility'].mean():.1f}%")
            
            with col2:
                st.metric("Average Beta", f"{risk_df['Beta'].mean():.2f}")
            
            with col3:
                st.metric("Average Max Drawdown", f"{risk_df['Max Drawdown'].mean():.1f}%")
            
            # Display statistics by risk type
            st.subheader("Statistics by Risk Type")
            
            # Format the statistics table
            st.dataframe(
                risk_stats.style.format({
                    "Volatility": "{:.1f}%",
                    "Beta": "{:.2f}",
                    "Max Drawdown": "{:.1f}%",
                    "Avg Daily Range": "{:.1f}%"
                }),
                use_container_width=True
            )
        
        with tab2:
            st.subheader("Risk Visualization")
            
            # Create scatter plot of volatility vs. max drawdown
            fig = px.scatter(
                risk_df,
                x="Volatility",
                y="Max Drawdown",
                color="Risk Type",
                size="Beta",
                hover_name="Symbol",
                hover_data=["Name", "Type", "Avg Daily Range"],
                title="Volatility vs. Max Drawdown",
                color_discrete_map={
                    "Conservative": "green",
                    "Moderate": "purple",
                    "Aggressive": "red"
                }
            )
            
            fig.update_layout(
                xaxis_title="Volatility (%)",
                yaxis_title="Max Drawdown (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create radar chart for risk comparison
            st.subheader("Risk Type Comparison")
            
            # Calculate average metrics by risk type
            risk_type_avg = risk_df.groupby("Risk Type")[["Volatility", "Beta", "Max Drawdown", "Avg Daily Range"]].mean()
            
            # Normalize the data for radar chart
            risk_type_normalized = risk_type_avg.copy()
            for col in risk_type_normalized.columns:
                if col == "Max Drawdown":
                    # Invert max drawdown since it's negative
                    risk_type_normalized[col] = (risk_type_normalized[col] - risk_type_normalized[col].min()) / (risk_type_normalized[col].max() - risk_type_normalized[col].min())
                else:
                    risk_type_normalized[col] = (risk_type_normalized[col] - risk_type_normalized[col].min()) / (risk_type_normalized[col].max() - risk_type_normalized[col].min())
            
            # Create radar chart
            fig = go.Figure()
            
            categories = ["Volatility", "Beta", "Max Drawdown", "Avg Daily Range"]
            
            for risk_type in risk_type_normalized.index:
                values = risk_type_normalized.loc[risk_type].values.tolist()
                values.append(values[0])  # Close the loop
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],  # Close the loop
                    fill='toself',
                    name=risk_type,
                    line_color='red' if risk_type == 'Aggressive' else ('green' if risk_type == 'Conservative' else 'purple')
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    finally:
        # Close database session
        session.close()

if __name__ == "__main__":
    show_risk_metrics()

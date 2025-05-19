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
from src.database.models import SymbolFields, IShareETF, IShareETFHolding

def show_holdings_analysis():
    st.title("Holdings Analysis")
    
    # Connect to database
    db_config, engine, session = get_db_connection(st.session_state.get("db_type", "sqlite"))
    
    try:
        # Get ETF data from database
        etfs = session.query(SymbolFields).filter(SymbolFields.is_etf == True).all()
        
        # Create filters
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter by ETF
            etf_symbols = [etf.symbol for etf in etfs]
            selected_etf = st.selectbox("Select ETF", etf_symbols, index=0 if etf_symbols else None)
        
        if selected_etf:
            # Create sample holdings data for the selected ETF
            # In a real application, this would come from the database
            
            # Top Holdings
            holdings_data = {
                "Stock Symbol": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK.B", "JPM", "V"],
                "Name": ["Apple Inc.", "Microsoft Corp.", "Amazon.com Inc.", "Alphabet Inc.", "Meta Platforms Inc.", "Tesla Inc.", "NVIDIA Corp.", "Berkshire Hathaway Inc.", "JPMorgan Chase & Co.", "Visa Inc."],
                "Sector": ["Information Technology", "Information Technology", "Consumer Discretionary", "Communication", "Communication", "Consumer Discretionary", "Information Technology", "Financials", "Financials", "Financials"],
                "Weight": [6.21, 5.73, 3.40, 3.38, 2.11, 1.77, 1.36, 1.35, 1.20, 1.15],
                "Market Value": [228471987, 210932441, 125030831, 124398765, 77654321, 65123456, 50123456, 49876543, 44123456, 42345678],
                "Price": [154.64, 333.59, 134.69, 138.63, 303.81, 174.52, 5.40, 373.38, 154.46, 271.58]
            }
            
            holdings_df = pd.DataFrame(holdings_data)
            
            # Create tabs for different analyses
            tab1, tab2 = st.tabs(["Holdings Analysis", "Top Holdings"])
            
            with tab1:
                st.subheader(f"{selected_etf} Holdings Analysis")
                
                # Create pie chart for sector allocation
                sector_weights = holdings_df.groupby("Sector")["Weight"].sum().reset_index()
                
                fig = px.pie(
                    sector_weights,
                    values="Weight",
                    names="Sector",
                    title=f"{selected_etf} Holdings by Sector",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                fig.update_traces(textposition="inside", textinfo="percent+label")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create bar chart for top 10 holdings
                fig = px.bar(
                    holdings_df,
                    x="Stock Symbol",
                    y="Weight",
                    title=f"{selected_etf} Top 10 Holdings",
                    color="Sector",
                    hover_data=["Name", "Market Value", "Price"]
                )
                
                fig.update_layout(
                    xaxis_title="Stock",
                    yaxis_title="Weight (%)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader(f"{selected_etf} Top Holdings - AAXJ")
                
                # Display holdings table
                st.dataframe(
                    holdings_df.style.format({
                        "Weight": "{:.2f}%",
                        "Market Value": "${:,.2f}",
                        "Price": "${:.2f}"
                    }),
                    use_container_width=True
                )
                
                # Create additional visualizations
                st.subheader("About Holdings")
                
                st.info(
                    f"This table shows the top 10 holdings for {selected_etf}. "
                    f"The total weight of these holdings is {holdings_df['Weight'].sum():.2f}% of the ETF. "
                    f"The total market value is ${holdings_df['Market Value'].sum():,.2f}."
                )
        else:
            st.warning("No ETFs found in the database.")
    
    finally:
        # Close database session
        session.close()

if __name__ == "__main__":
    show_holdings_analysis()

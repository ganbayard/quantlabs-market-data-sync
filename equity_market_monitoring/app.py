import streamlit as st
import pandas as pd
import os
import sys
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler("streamlit.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Invest Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import MySQL connection
from src.database.mysql_connection import get_mysql_connection

# Import pages
from src.pages.dashboard import show_dashboard
from src.pages.financials import show_financials
from src.pages.holdings_analysis import show_holdings_analysis
from src.pages.risk_metrics import show_risk_metrics
from src.pages.market_breadth import show_market_breadth
from src.pages.sentiment_analysis import show_sentiment_analysis
from src.pages.quantlabs_etf import show_quantlabs_etf  # New import

# Import theme utilities
from src.utils.theme_utils import get_theme_colors

# Initialize theme in session state if not already set
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # Default theme


# Custom CSS
def load_css():
    # Get theme colors
    colors = get_theme_colors()

    # Base CSS file
    css_file = os.path.join(
        os.path.dirname(__file__), "src", "assets", "custom_styles.css"
    )

    # Theme-specific CSS adjustments
    theme_css = f"""
    <style>
    /* Theme-specific overrides */
    .stApp {{
        background-color: {colors["bg_color"]};
        color: {colors["text_color"]};
    }}
    
    .stSidebar {{
        background-color: {colors["secondary_bg"]};
    }}
    
    .chart-card {{
        background-color: {colors["card_bg"]};
        border: 1px solid {colors["card_border"]};
    }}
    </style>
    """

    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Apply theme-specific CSS
    st.markdown(theme_css, unsafe_allow_html=True)


def apply_theme_selection():
    """Handle theme selection with dropdown"""
    # Theme selection dropdown
    theme_options = ["System", "Light", "Dark"]

    # Determine current index based on session state
    current_theme = st.session_state.get("theme", "dark")
    default_index = 2  # Default to Dark

    if current_theme == "light":
        default_index = 1
    elif current_theme == "system":
        default_index = 0

    selected_theme = st.sidebar.selectbox(
        "Select Theme", theme_options, index=default_index, key="theme_selection"
    )

    # Apply selected theme if changed
    if (
        (selected_theme == "Dark" and current_theme != "dark")
        or (selected_theme == "Light" and current_theme != "light")
        or (selected_theme == "System" and current_theme != "system")
    ):

        # Map selection to theme value
        if selected_theme == "Dark":
            st.session_state.theme = "dark"
        elif selected_theme == "Light":
            st.session_state.theme = "light"
        else:  # System
            st.session_state.theme = "system"
            # For system theme, we'll default to dark for now
            # In a full implementation, you'd detect OS preference
            st.session_state.theme = "dark"

        st.rerun()


def main():
    # App title - MUST BE FIRST SIDEBAR ELEMENT
    st.sidebar.title("Invest Analytics")

    # Environment indicator
    env = os.environ.get("MIGRATION_ENV", "dev").upper()
    st.sidebar.info(f"Connected to {env} database")

    # Theme selection dropdown
    apply_theme_selection()

    # Load CSS after theme is determined
    load_css()

    # Create MySQL connection at the app level
    engine, session = get_mysql_connection()

    if engine is None or session is None:
        st.error("Failed to connect to database. Please check logs for details.")
        return

    try:
        # Store session in session state
        st.session_state["db_session"] = session

        # Navigation
        st.sidebar.header("Navigation")

        # Create navigation using radio buttons
        page = st.sidebar.radio(
            "Select a page",
            [
                "Dashboard",
                "Market Breadth",
                "Financial Analysis",
                "Sentiment Analysis",
                "QuantLabs ETF",  # New item replacing Holdings Analysis and Risk Metrics
            ],
        )

        # Display the selected page
        if page == "Dashboard":
            show_dashboard()
        elif page == "Market Breadth":
            show_market_breadth()
        elif page == "Financial Analysis":
            show_financials()
        elif page == "Sentiment Analysis":
            show_sentiment_analysis()
        elif page == "QuantLabs ETF":  # New condition
            show_quantlabs_etf()

        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

    finally:
        # Close session when app is done
        if "db_session" in st.session_state and st.session_state["db_session"]:
            st.session_state["db_session"].close()
            logger.info("Database session closed")


if __name__ == "__main__":
    main()

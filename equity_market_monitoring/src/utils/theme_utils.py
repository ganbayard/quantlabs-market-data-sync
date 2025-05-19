import streamlit as st


def get_theme_colors():
    """Get the current theme colors based on session state"""
    is_dark = st.session_state.get("theme", "dark") == "dark"

    if is_dark:
        return {
            "bg_color": "#0E1117",
            "secondary_bg": "#262730",
            "text_color": "#FAFAFA",
            "primary_color": "#FF4B4B",
            "chart_bg": "#262730",
            "grid_color": "rgba(255, 255, 255, 0.1)",
            "card_bg": "rgba(38, 39, 48, 0.3)",
            "card_border": "rgba(255, 255, 255, 0.1)",
        }
    else:
        return {
            "bg_color": "#FFFFFF",
            "secondary_bg": "#F0F2F6",
            "text_color": "#262730",
            "primary_color": "#FF4B4B",
            "chart_bg": "#FFFFFF",
            "grid_color": "rgba(0, 0, 0, 0.1)",
            "card_bg": "rgba(240, 242, 246, 0.7)",
            "card_border": "rgba(0, 0, 0, 0.1)",
        }

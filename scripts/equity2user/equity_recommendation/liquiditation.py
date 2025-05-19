import logging

# Configure logging
logger = logging.getLogger(__name__)

def calculate_buy_short_points(price, volatility, adr):
    """
    Calculate simple buy and short points based on price, volatility and ADR
    
    Args:
        price (float): Current stock price
        volatility (float): Stock volatility measure
        adr (float): Average Daily Range
        
    Returns:
        tuple: (buy_point, short_point)
    """
    buy_point = price * (1 - (volatility * 0.5))
    short_point = price * (1 + (adr * 0.01 * 0.5))
    return buy_point, short_point
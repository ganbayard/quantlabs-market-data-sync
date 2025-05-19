import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def determine_overbuy_oversold(mfi, trend_intensity_value, persistence_ratio):
    """
    Calculate a numeric value representing overbought/oversold status on a scale from -100 to 100
    
    - Negative values: Oversold (more negative = stronger oversold)
    - Positive values: Overbought (more positive = stronger overbought)
    - Zero area: Neutral
    
    Args:
        mfi (float): Money Flow Index value (0-100)
        trend_intensity_value (float): Trend intensity indicator
        persistence_ratio (float): Persistence ratio value
        
    Returns:
        float: Overbought/oversold value (-100 to 100)
    """
    # Base calculation from MFI (convert 0-100 scale to -100 to 100 scale)
    base_value = (mfi - 50) * 2
    
    # Adjustments based on trend intensity and persistence
    # For oversold conditions (negative base_value)
    if base_value < 0:
        # Strong trend + high persistence makes it more strongly oversold (buy signal)
        if trend_intensity_value > 1.05 and persistence_ratio > 1.5:
            # Amplify the signal for strong buy conditions
            adjustment = -25  # Push further negative for stronger buy signal
        else:
            adjustment = 0
    # For overbought conditions (positive base_value)
    elif base_value > 0:
        # Weak trend + low persistence makes it more strongly overbought (sell signal)
        if trend_intensity_value < 0.95 and persistence_ratio < 0.7:
            adjustment = 20  # Push further positive for stronger sell signal
        else:
            adjustment = 0
    else:
        adjustment = 0
    
    # Apply adjustment and ensure within bounds (-100 to 100)
    final_value = max(-100, min(100, base_value + adjustment))
    
    return final_value
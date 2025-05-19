import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Constants
ADR_PERIOD = 20  # Days to use for ADR calculation
MFI_PERIOD = 14  # Period for Money Flow Index
TREND_PERIOD = 65  # Period for trend intensity calculation

# Thread-local storage
thread_local = threading.local()

# Session factory - to be set by the main module
session_factory = None

def set_session_factory(factory):
    """Set the session factory to be used by this module"""
    global session_factory
    session_factory = factory

def get_thread_session():
    """Get or create a thread-local database session"""
    if not hasattr(thread_local, "session"):
        if session_factory is None:
            raise RuntimeError("Session factory not set. Call set_session_factory first.")
        thread_local.session = session_factory()
    return thread_local.session

def money_flow_index(high, low, close, volume, length=14): 
    """Calculate Money Flow Index (MFI)"""
    tp = (high + low + close) / 3
    raw_money_flow = tp * volume
    pos_flow = raw_money_flow.where(tp > tp.shift(1), 0)
    neg_flow = raw_money_flow.where(tp < tp.shift(1), 0)
    pos_mf_sum = pos_flow.rolling(window=length).sum()
    negat_mf_sum = neg_flow.rolling(window=length).sum()
    money_flow_ratio = pos_mf_sum / negat_mf_sum
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi

def trend_intensity(close_series):
    """Calculate Trend Intensity based on moving averages"""
    avgc7 = close_series.rolling(7).mean()
    avgc65 = close_series.rolling(65).mean()
    return avgc7/avgc65

def calculate_rsi(close_series, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_price_data(symbol, days=60):
    """Fetch real price data from YfBar1d table"""
    from models.market_data import YfBar1d
    
    session = get_thread_session()
    cutoff_date = date.today() - timedelta(days=days)
    
    try:
        # Query price data from YfBar1d table
        bars = session.query(YfBar1d).filter(
            YfBar1d.symbol == symbol,
            YfBar1d.timestamp >= cutoff_date
        ).order_by(YfBar1d.timestamp.asc()).all()
        
        if not bars:
            logger.warning(f"No price data found for {symbol}")
            return None
            
        # Convert to pandas DataFrame for calculations
        data = pd.DataFrame([
            {
                'date': bar.timestamp.date(),
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars
        ])
        
        return data
    
    except Exception as e:
        logger.error(f"Error fetching price data for {symbol}: {str(e)}")
        return None

def calculate_metrics(symbol, days=30):
    """Calculate all metrics for a symbol from real price data"""
    price_data = get_price_data(symbol, days=max(days, 90))  # Get more data for better calculations
    
    if price_data is None or len(price_data) < 20:  # Need minimum data for calculations
        logger.warning(f"Insufficient price data for {symbol}")
        return None
    
    try:
        # Calculate ADR (Average Daily Range)
        price_data['daily_range'] = (price_data['high'] - price_data['low']) / price_data['close'] * 100
        adr = price_data['daily_range'].tail(ADR_PERIOD).mean()
        
        # Calculate volatility (20-day)
        price_data['return'] = price_data['close'].pct_change()
        volatility = price_data['return'].tail(20).std() * (252 ** 0.5)  # Annualized
        
        # Calculate MFI (Money Flow Index)
        mfi_values = money_flow_index(
            price_data['high'], 
            price_data['low'], 
            price_data['close'], 
            price_data['volume'],
            length=MFI_PERIOD
        )
        current_mfi = mfi_values.iloc[-1] if not mfi_values.empty else 50
        
        # Calculate Trend Intensity
        trend_intensity_values = trend_intensity(price_data['close'])
        # Handle NaN values properly
        if trend_intensity_values.empty:
            current_trend = 1.0
        else:
            current_trend = trend_intensity_values.iloc[-1]
            if np.isnan(current_trend):
                current_trend = 1.0
        
        # Calculate Persistence Ratio
        try:
            # Calculate 14-day cumulative return
            cum_return_14d = price_data['return'].rolling(window=14).sum()
            
            # Calculate persistence ratio (absolute cumulative return / sum of absolute daily returns)
            abs_cum_return = cum_return_14d.abs()
            sum_abs_returns = price_data['return'].abs().rolling(window=14).sum()
            
            # Avoid division by zero
            persistence_ratio = np.where(
                sum_abs_returns > 0,
                abs_cum_return / sum_abs_returns,
                1.0  # Default value when denominator is zero
            )
            
            current_persistence = persistence_ratio[-1] if len(persistence_ratio) > 0 else 1.0
        except Exception as e:
            logger.error(f"Error calculating persistence ratio for {symbol}: {e}")
            current_persistence = 1.0  # Default value
        
        # Volume spike detection
        try:
            # Calculate average volume
            avg_volume = price_data['volume'].tail(20).mean()
            recent_volumes = price_data['volume'].tail(3)
            
            # Count how many recent days had volume spikes
            spike_days = sum(vol > avg_volume * 1.5 for vol in recent_volumes)
            
            # Format volume spike indicators
            if spike_days == 0:
                volume_spike = ''
            else:
                volume_spike = 'O ' * spike_days
        except Exception as e:
            logger.error(f"Error calculating volume spike for {symbol}: {e}")
            volume_spike = ''
        
        # Get current price
        current_price = price_data['close'].iloc[-1] if not price_data.empty else 0
        
        # Create metrics dictionary
        metrics = {
            'Symbol': symbol,
            'Price': current_price,
            'ADR': adr,
            'Volatility': volatility,
            'Trend Intensity': current_trend * 50,  # Scale to match expected range
            'Persistence Ratio': current_persistence,
            'MFI': current_mfi,
            'Volume': volume_spike,
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
        return None

def classify_stock(stock_data):
    """
    Classify a stock as 'Aggressive', 'Moderate', or 'Conservative' based on various metrics
    """
    adr_value = stock_data.get('ADR', 0)
    persistence_ratio = stock_data.get('Persistence Ratio', 0)
    volatility = stock_data.get('Volatility', 0)
    trend_intensity_value = stock_data.get('Trend Intensity', 0)
    
    # ADR Classification
    if adr_value >= 5:
        adr_class = 'Aggressive'
    elif 3 <= adr_value < 5:
        adr_class = 'Moderate'
    else:
        adr_class = 'Conservative'

    # Persistence Ratio Classification
    if persistence_ratio > 2.0:
        persistence_class = 'Aggressive'
    elif 0.8 <= persistence_ratio <= 2.0:
        persistence_class = 'Moderate'
    else:
        persistence_class = 'Conservative'

    # Volatility Classification
    if volatility > 0.15:
        volatility_class = 'Aggressive'
    elif 0.06 < volatility <= 0.15:
        volatility_class = 'Moderate'
    else:
        volatility_class = 'Conservative'
    
    # Trend intensity Classification
    if trend_intensity_value > 70:
        trend_class = 'Aggressive'
    elif 40 <= trend_intensity_value <= 70:
        trend_class = 'Moderate'
    else:
        trend_class = 'Conservative'
    
    classifications = [adr_class, persistence_class, volatility_class, trend_class]
 
    if classifications.count('Aggressive') >= 2:
        return 'Aggressive'
    elif classifications.count('Moderate') >= 2:
        return 'Moderate'
    else:
        return 'Conservative'

def get_stock_metrics(symbols, days=30):
    """
    Calculate real metrics for symbols using YfBar1d data
    """
    logger.info(f"Calculating metrics for {len(symbols)} symbols...")
    
    stock_data = {}
    calculated = 0
    failed = 0
    
    for symbol in symbols:
        metrics = calculate_metrics(symbol, days)
        if metrics:
            stock_data[symbol] = metrics
            calculated += 1
        else:
            failed += 1
        
        # Log progress
        if (calculated + failed) % 20 == 0 or (calculated + failed) == len(symbols):
            logger.info(f"Metrics progress: {calculated + failed}/{len(symbols)} - Success: {calculated}, Failed: {failed}")
    
    logger.info(f"Finished calculating metrics for {calculated} symbols")
    return stock_data, get_sector_data(symbols)

def get_sector_data(symbols):
    """
    Create a simple sector dictionary from symbol prefixes
    In a real implementation, this would use a proper sector database
    """
    sector_dict = {}
    
    # Simple mapping of first letter to sectors for demonstration
    sectors = {
        "A": "Technology",
        "B": "Financials",
        "C": "Healthcare",
        # Rest of the sector mappings...
        "Z": "Technology"
    }
    
    # Assign sectors based on first letter
    for symbol in symbols:
        first_letter = symbol[0].upper() if symbol else "A"
        sector_dict[symbol] = sectors.get(first_letter, "Unknown")
    
    return sector_dict
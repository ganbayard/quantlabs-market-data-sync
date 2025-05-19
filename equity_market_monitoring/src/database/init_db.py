import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.database.db_config import get_db_connection
from src.database.models import Base, SymbolFields, YfBar1d, EquityRiskProfile

def initialize_database(db_type="sqlite"):
    """Initialize the database with tables and sample data"""
    db_config, engine, session = get_db_connection(db_type)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Check if we already have data
    symbol_count = session.query(SymbolFields).count()
    
    if symbol_count == 0:
        # Insert sample data
        insert_sample_data(session)
    
    session.close()
    return True

def insert_sample_data(session):
    """Insert sample data into the database"""
    # Sample symbols
    symbols = [
        {"symbol": "AAPL", "company_name": "Apple Inc.", "is_etf": False, "sector": "Technology", "industry": "Consumer Electronics"},
        {"symbol": "MSFT", "company_name": "Microsoft Corporation", "is_etf": False, "sector": "Technology", "industry": "Software"},
        {"symbol": "AMZN", "company_name": "Amazon.com Inc.", "is_etf": False, "sector": "Consumer Cyclical", "industry": "Internet Retail"},
        {"symbol": "GOOGL", "company_name": "Alphabet Inc.", "is_etf": False, "sector": "Communication Services", "industry": "Internet Content & Information"},
        {"symbol": "META", "company_name": "Meta Platforms Inc.", "is_etf": False, "sector": "Communication Services", "industry": "Internet Content & Information"},
        {"symbol": "TSLA", "company_name": "Tesla Inc.", "is_etf": False, "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
        {"symbol": "SPY", "company_name": "SPDR S&P 500 ETF Trust", "is_etf": True, "asset_class": "Equity", "focus": "Large Cap"},
        {"symbol": "QQQ", "company_name": "Invesco QQQ Trust", "is_etf": True, "asset_class": "Equity", "focus": "Technology"},
        {"symbol": "IWM", "company_name": "iShares Russell 2000 ETF", "is_etf": True, "asset_class": "Equity", "focus": "Small Cap"},
        {"symbol": "AAXJ", "company_name": "iShares MSCI All Country Asia ex Japan ETF", "is_etf": True, "asset_class": "Equity", "focus": "International"}
    ]
    
    # Insert symbols
    for symbol_data in symbols:
        symbol = SymbolFields(
            symbol=symbol_data["symbol"],
            company_name=symbol_data["company_name"],
            is_etf=symbol_data["is_etf"],
            price=random.uniform(50, 500),
            change=random.uniform(-5, 5),
            volume=random.randint(1000000, 10000000),
            market="NASDAQ" if random.random() > 0.3 else "NYSE",
            country="USA",
            exchange="NASDAQ" if random.random() > 0.3 else "NYSE"
        )
        
        if not symbol_data["is_etf"]:
            symbol.sector = symbol_data["sector"]
            symbol.industry = symbol_data["industry"]
            symbol.market_cap = random.uniform(100000000000, 3000000000000)
        else:
            symbol.asset_class = symbol_data["asset_class"]
            symbol.focus = symbol_data["focus"]
            symbol.expense_ratio = random.uniform(0.03, 0.5)
            
        session.add(symbol)
    
    session.commit()
    
    # Generate price history for each symbol
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    for symbol_data in symbols:
        symbol = symbol_data["symbol"]
        
        # Generate daily price data
        current_date = start_date
        price = random.uniform(50, 500)
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Only weekdays
                daily_change = random.uniform(-0.03, 0.03)
                open_price = price
                close_price = price * (1 + daily_change)
                high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.02))
                low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.02))
                volume = random.randint(1000000, 10000000)
                
                # Don't include id field, let SQLAlchemy handle autoincrement
                bar = YfBar1d(
                    symbol=symbol,
                    timestamp=current_date,
                    open=float(open_price),  # Ensure these are Python floats
                    high=float(high_price),
                    low=float(low_price),
                    close=float(close_price),
                    volume=int(volume)  # Ensure this is a Python int
                )
                
                session.add(bar)
                price = close_price
            
            current_date += timedelta(days=1)
        
        # Commit after each symbol to avoid large transactions
        session.commit()
    
    # Add risk profiles
    risk_types = ["Conservative", "Moderate", "Aggressive"]
    
    for symbol_data in symbols:
        symbol = symbol_data["symbol"]
        is_etf = symbol_data["is_etf"]
        
        risk_profile = EquityRiskProfile(
            symbol=symbol,
            is_etf=is_etf,
            price=float(random.uniform(50, 500)),
            risk_type=random.choice(risk_types),
            average_score=float(random.uniform(0, 100)),
            volatility_3yr_avg=float(random.uniform(10, 30)),
            beta_3yr_avg=float(random.uniform(0.5, 1.5)),
            max_drawdown_3yr_avg=float(random.uniform(-30, -5)),
            adr_3yr_avg=float(random.uniform(0.5, 2.5)),
            classified_at=datetime.now()
        )
        
        session.add(risk_profile)
    
    session.commit()
    
    return True

if __name__ == "__main__":
    initialize_database()

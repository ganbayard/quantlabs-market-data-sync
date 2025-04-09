import os
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Import all models
from .market_data import (
    Base, 
    YfBar1d, 
    SymbolFields, 
    MmtvDailyBar,
    IShareETF, 
    IShareETFHolding, 
    IncomeStatement,  
    BalanceSheet,     
    CashFlow,         
    NewsArticle,      
    Equity2User,
    EquityTechnicalIndicator,
    Company,         
    Executive        
)

# Load environment variables
load_dotenv()

# Choose connection based on environment
if os.getenv('MIGRATION_ENV') == 'prod':
    # Production database connection
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '3306')
    db_name = os.getenv('DB_NAME')
else:
    # Development database connection (Docker)
    db_user = os.getenv('LOCAL_DB_USER')
    db_password = os.getenv('LOCAL_DB_PASSWORD')
    db_host = os.getenv('LOCAL_DB_HOST', 'localhost')
    db_port = os.getenv('LOCAL_DB_PORT', '3306')
    db_name = os.getenv('LOCAL_DB_NAME')

# Create engine from environment variables
connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(
    connection_string, 
    pool_size=30,           # Increase from 25 to 30
    max_overflow=20,        # Increase from 10 to 20
    pool_recycle=3600,
    pool_pre_ping=True      # Add this for connection health checks
)
Session = sessionmaker(bind=engine)

# Export all models
__all__ = [
    'Base',
    'YfBar1d',
    'SymbolFields',
    'MmtvDailyBar',
    'IShareETF',
    'IShareETFHolding',
    'IncomeStatement',    
    'BalanceSheet',       
    'CashFlow',           
    'NewsArticle',        
    'Equity2User',
    'EquityTechnicalIndicator',
    'Company',            
    'Executive',          
    'engine',
    'Session'
]


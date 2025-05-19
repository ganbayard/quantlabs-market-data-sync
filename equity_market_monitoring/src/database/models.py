import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
from sqlalchemy.dialects import mysql
import sqlite3

class Base(DeclarativeBase):
    pass

# Daily Bar data
class YfBar1d(Base):
    __tablename__ = 'yf_daily_bar'
    id        = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol    = sa.Column(sa.String(255), nullable=False)  
    timestamp = sa.Column(sa.DateTime, nullable=False)
    # Using DECIMAL for compatibility
    open      = sa.Column(sa.Float(precision=10), nullable=False)
    high      = sa.Column(sa.Float(precision=10), nullable=False)
    low       = sa.Column(sa.Float(precision=10), nullable=False)
    close     = sa.Column(sa.Float(precision=10), nullable=False)
    volume    = sa.Column(sa.BigInteger, nullable=False)
    __table_args__ = (
        sa.UniqueConstraint('symbol', 'timestamp', name='yf_daily_bar_symbol_timestamp_unique'),
        sa.Index('yf_daily_bar_symbol_index', 'symbol'),
        sa.Index('yf_daily_bar_timestamp_index', 'timestamp'),
    )

# Symbol Fields
class SymbolFields(Base):
    __tablename__ = 'symbol_fields'

    id           = sa.Column(sa.Integer, primary_key=True)
    symbol       = sa.Column(sa.String(255), unique=True, nullable=False, index=True)
    is_etf       = sa.Column(sa.Boolean, default=False, nullable=False, index=True)
    company_name = sa.Column(sa.String(255), nullable=True) # Fund name for ETFs
    price        = sa.Column(sa.Float(precision=15), nullable=True)
    change       = sa.Column(sa.Float(precision=15), key='change', nullable=True)
    volume       = sa.Column(sa.BigInteger, nullable=True)
    market       = sa.Column(sa.String(255), nullable=True)
    country      = sa.Column(sa.String(255), nullable=True)
    exchange     = sa.Column(sa.String(255), nullable=True)
    updated_at   = sa.Column(sa.DateTime, default=datetime.now, onupdate=datetime.now)

    # Stock specific fields (nullable)
    market_cap   = sa.Column(sa.Float(precision=25), nullable=True)
    sector       = sa.Column(sa.String(255), nullable=True)
    industry     = sa.Column(sa.String(255), nullable=True)
    earnings_release_trading_date_fq      = sa.Column(sa.DateTime, nullable=True)
    earnings_release_next_trading_date_fq = sa.Column(sa.DateTime, nullable=True)
    indexes      = sa.Column(sa.Text, nullable=True)

    # ETF specific fields (nullable)
    relative_volume     = sa.Column(sa.Float, nullable=True)
    aum                 = sa.Column(sa.Float(precision=25), nullable=True)
    nav_total_return_3y = sa.Column(sa.Float, nullable=True)
    expense_ratio       = sa.Column(sa.Float(precision=10), nullable=True)
    asset_class         = sa.Column(sa.String(255), nullable=True)
    focus               = sa.Column(sa.String(255), nullable=True)

    daily_bars = relationship("YfBar1d", foreign_keys=[YfBar1d.symbol], primaryjoin="SymbolFields.symbol == YfBar1d.symbol", cascade="all, delete-orphan")

# Sector rotation data
class MmtvDailyBar(Base):
    __tablename__ = 'mmtv_daily_bars'

    date       = sa.Column(sa.DateTime, primary_key=True)
    field_name = sa.Column(sa.String(255), primary_key=True)  
    field_type = sa.Column(sa.String(255), primary_key=True)  
    open       = sa.Column(sa.Float(precision=8))
    high       = sa.Column(sa.Float(precision=8))
    low        = sa.Column(sa.Float(precision=8))
    close      = sa.Column(sa.Float(precision=8))

# ETF data
class IShareETF(Base):
    __tablename__ = 'ishare_etf'
    
    id              = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    fund_name       = sa.Column(sa.String(255), nullable=False)
    inception_date  = sa.Column(sa.Date)
    ticker          = sa.Column(sa.String(255), nullable=False, unique=True)
    cusip           = sa.Column(sa.String(255))
    isin            = sa.Column(sa.String(255))
    asset_class     = sa.Column(sa.String(255))
    subasset_class  = sa.Column(sa.String(255))
    country         = sa.Column(sa.String(255))
    region          = sa.Column(sa.String(255))
    product_url     = sa.Column(sa.String(255))
    product_id      = sa.Column(sa.String(255))
    net_assets      = sa.Column(sa.Float(precision=20))
    fund_type       = sa.Column(sa.String(255))
    provider        = sa.Column(sa.String(255))
    exchange        = sa.Column(sa.String(255))
    benchmark       = sa.Column(sa.String(255))
    created_at      = sa.Column(sa.DateTime, default=datetime.now)
    updated_at      = sa.Column(sa.DateTime, default=datetime.now, onupdate=datetime.now)

    # Define relationship with holdings
    holdings = relationship('IShareETFHolding', back_populates='ishare_etf', cascade="all, delete-orphan")

class IShareETFHolding(Base):
    __tablename__ = 'ishare_etf_holding'
    id               = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    ishare_etf_id    = sa.Column(sa.BigInteger, sa.ForeignKey('ishare_etf.id', ondelete='CASCADE'), nullable=False)
    ticker           = sa.Column(sa.String(255))
    name             = sa.Column(sa.String(255), nullable=False)
    sector           = sa.Column(sa.String(255), server_default='Other')
    asset_class      = sa.Column(sa.String(255), nullable=False)
    market_value     = sa.Column(sa.Float(precision=20), nullable=False)
    weight           = sa.Column(sa.Float(precision=10), nullable=False)
    notional_value   = sa.Column(sa.Float(precision=20), nullable=False)
    amount           = sa.Column(sa.Float(precision=20), nullable=False)
    price            = sa.Column(sa.Float(precision=20), nullable=False)
    location         = sa.Column(sa.String(255))
    exchange         = sa.Column(sa.String(255))
    currency         = sa.Column(sa.String(255))
    fx_rate          = sa.Column(sa.Float(precision=10), nullable=False)
    market_currency  = sa.Column(sa.String(255), server_default='USD')
    accrual_date     = sa.Column(sa.Date)
    fund_ticker      = sa.Column(sa.String(255), nullable=False)
    as_of_date       = sa.Column(sa.Date, nullable=False)
    created_at       = sa.Column(sa.DateTime, default=datetime.now)
    updated_at       = sa.Column(sa.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Define relationship with parent ETF
    ishare_etf       = relationship('IShareETF', back_populates='holdings')
    
    __table_args__ = (
        sa.UniqueConstraint('ishare_etf_id', 'ticker', name='uix_ishare_etf_holding_etf_ticker'),
        sa.Index('ishare_etf_holding_ishare_etf_id_foreign', 'ishare_etf_id'),
    )

# Financial details data
class IncomeStatement(Base):
    __tablename__ = 'income_statements'
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(255), nullable=False)
    date                    = sa.Column(sa.DateTime, nullable=False)
    period_type             = sa.Column(sa.String(10), nullable=False)
    
    # Revenue and profitability
    total_revenue           = sa.Column(sa.Float)
    cost_of_revenue         = sa.Column(sa.Float)
    gross_profit            = sa.Column(sa.Float)
    
    # Operating metrics
    operating_expense       = sa.Column(sa.Float)
    operating_income        = sa.Column(sa.Float)
    total_operating_income  = sa.Column(sa.Float)
    total_expenses          = sa.Column(sa.Float)
    
    # Non-operating items
    net_non_operating_interest = sa.Column(sa.Float)
    other_income_expense    = sa.Column(sa.Float)
    pretax_income           = sa.Column(sa.Float)
    tax_provision           = sa.Column(sa.Float)
    
    # Net income and EPS
    net_income              = sa.Column(sa.Float)
    normalized_income       = sa.Column(sa.Float)
    basic_shares            = sa.Column(sa.Float)
    diluted_shares          = sa.Column(sa.Float)
    basic_eps               = sa.Column(sa.Float)
    diluted_eps             = sa.Column(sa.Float)
    
    # Additional metrics
    ebit                    = sa.Column(sa.Float)
    ebitda                  = sa.Column(sa.Float)
    interest_income         = sa.Column(sa.Float)
    interest_expense        = sa.Column(sa.Float)
    net_interest_income     = sa.Column(sa.Float)
    
    # Timestamps
    last_updated            = sa.Column(sa.DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        sa.UniqueConstraint('symbol', 'date', 'period_type', name='uix_income_statement_key'),
        sa.Index('income_statements_symbol_index', 'symbol'),
    )

class BalanceSheet(Base):
    __tablename__ = 'balance_sheets'
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(255), nullable=False)
    date                    = sa.Column(sa.DateTime, nullable=False)
    period_type             = sa.Column(sa.String(10), nullable=False)
    
    # Main categories
    total_assets            = sa.Column(sa.Float)
    total_liabilities       = sa.Column(sa.Float)
    total_equity            = sa.Column(sa.Float)
    
    # Additional metrics
    total_capitalization    = sa.Column(sa.Float)
    common_stock_equity     = sa.Column(sa.Float)
    capital_lease_obligations = sa.Column(sa.Float)
    net_tangible_assets     = sa.Column(sa.Float)
    working_capital         = sa.Column(sa.Float)
    invested_capital        = sa.Column(sa.Float)
    tangible_book_value     = sa.Column(sa.Float)
    total_debt              = sa.Column(sa.Float)
    
    # Share information
    shares_issued           = sa.Column(sa.Float)
    ordinary_shares_number  = sa.Column(sa.Float)
    
    # Timestamps
    last_updated            = sa.Column(sa.DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        sa.UniqueConstraint('symbol', 'date', 'period_type', name='uix_balance_sheet_key'),
        sa.Index('balance_sheets_symbol_index', 'symbol'),
    )

class CashFlow(Base):
    __tablename__ = 'cash_flows'
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(255), nullable=False)
    date                    = sa.Column(sa.DateTime, nullable=False)
    period_type             = sa.Column(sa.String(10), nullable=False)
    
    # Main cash flow categories
    operating_cash_flow     = sa.Column(sa.Float)
    investing_cash_flow     = sa.Column(sa.Float)
    financing_cash_flow     = sa.Column(sa.Float)
    free_cash_flow          = sa.Column(sa.Float)
    end_cash_position       = sa.Column(sa.Float)
    
    # Detailed items
    income_tax_paid         = sa.Column(sa.Float)
    interest_paid           = sa.Column(sa.Float)
    capital_expenditure     = sa.Column(sa.Float)
    issuance_of_capital_stock = sa.Column(sa.Float)
    issuance_of_debt        = sa.Column(sa.Float)
    repayment_of_debt       = sa.Column(sa.Float)
    
    # Timestamps
    last_updated            = sa.Column(sa.DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        sa.UniqueConstraint('symbol', 'date', 'period_type', name='uix_cash_flow_key'),
        sa.Index('cash_flows_symbol_index', 'symbol'),
    )

# Risk profile data
class EquityRiskProfile(Base):
    __tablename__ = 'equity_risk_profile'
    
    id               = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol           = sa.Column(sa.String(255), nullable=False, index=True)
    is_etf           = sa.Column(sa.Boolean, default=False, nullable=False, index=True)
    price            = sa.Column(sa.Float(precision=15), nullable=True)
    risk_type        = sa.Column(sa.String(50), nullable=False, index=True)
    average_score    = sa.Column(sa.Float, nullable=True)
    
    # 3-year averages
    volatility_3yr_avg    = sa.Column(sa.Float, nullable=True)
    beta_3yr_avg          = sa.Column(sa.Float, nullable=True)
    max_drawdown_3yr_avg  = sa.Column(sa.Float, nullable=True)
    adr_3yr_avg           = sa.Column(sa.Float, nullable=True)
    
    # Year by year metrics - Year3 is most recent
    volatility_year3  = sa.Column(sa.Float, nullable=True)
    volatility_year2  = sa.Column(sa.Float, nullable=True)
    volatility_year1  = sa.Column(sa.Float, nullable=True)
    
    beta_year3        = sa.Column(sa.Float, nullable=True)
    beta_year2        = sa.Column(sa.Float, nullable=True)
    beta_year1        = sa.Column(sa.Float, nullable=True)
    
    max_drawdown_year3 = sa.Column(sa.Float, nullable=True)
    max_drawdown_year2 = sa.Column(sa.Float, nullable=True)
    max_drawdown_year1 = sa.Column(sa.Float, nullable=True)
    
    adr_year3         = sa.Column(sa.Float, nullable=True)
    adr_year2         = sa.Column(sa.Float, nullable=True)
    adr_year1         = sa.Column(sa.Float, nullable=True)
    
    # Individual metric scores
    volatility_score  = sa.Column(sa.Float, nullable=True)
    beta_score        = sa.Column(sa.Float, nullable=True)
    drawdown_score    = sa.Column(sa.Float, nullable=True)
    adr_score         = sa.Column(sa.Float, nullable=True)
    
    # Timestamps
    classified_at    = sa.Column(sa.DateTime, nullable=True)
    created_at       = sa.Column(sa.DateTime, default=datetime.now)
    updated_at       = sa.Column(sa.DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        sa.UniqueConstraint('symbol', 'classified_at', name='uix_equity_risk_profile_symbol_date'),
    )

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
from sqlalchemy.dialects import mysql

class Base(DeclarativeBase):
    pass

############################################################################################ Daily Bar data

class YfBar1d(Base):
    __tablename__ = 'yf_daily_bar'
    id        = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol    = sa.Column(sa.String(255), nullable=False)  
    timestamp = sa.Column(sa.DateTime, nullable=False)
    # Using DECIMAL to match production DDL
    open      = sa.Column(mysql.DECIMAL(precision=10, scale=4), nullable=False)
    high      = sa.Column(mysql.DECIMAL(precision=10, scale=4), nullable=False)
    low       = sa.Column(mysql.DECIMAL(precision=10, scale=4), nullable=False)
    close     = sa.Column(mysql.DECIMAL(precision=10, scale=4), nullable=False)
    volume    = sa.Column(sa.BigInteger, nullable=False)
    __table_args__ = (
        sa.UniqueConstraint('symbol', 'timestamp', name='yf_daily_bar_symbol_timestamp_unique'),
        sa.Index('yf_daily_bar_symbol_index', 'symbol'),
        sa.Index('yf_daily_bar_timestamp_index', 'timestamp'),
        sa.ForeignKeyConstraint(['symbol'], ['symbol_fields.symbol'], name='yf_daily_bar_symbol_foreign', ondelete='CASCADE'),
    )
    def __repr__(self):
        return f"YfBar1d(symbol='{self.symbol}', timestamp={self.timestamp})"
    def get_tuple(self):
        return (
            self.timestamp,
            self.symbol,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume
        )

############################################################################################ Daily Changes data


class SymbolFields(Base):
    __tablename__ = 'symbol_fields'

    id           = sa.Column(sa.Integer, primary_key=True)
    symbol       = sa.Column(sa.String(255), unique=True, nullable=False, index=True)
    is_etf       = sa.Column(sa.Boolean, default=False, nullable=False, index=True)
    company_name = sa.Column(sa.String(255), nullable=True) # Fund name for ETFs
    price        = sa.Column(mysql.DECIMAL(precision=15, scale=4), nullable=True)
    change       = sa.Column(mysql.DECIMAL(precision=15, scale=4), key='change', nullable=True)
    volume       = sa.Column(sa.BigInteger, nullable=True)
    market       = sa.Column(sa.String(255), nullable=True)
    country      = sa.Column(sa.String(255), nullable=True)
    exchange     = sa.Column(sa.String(255), nullable=True)
    updated_at   = sa.Column(mysql.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now())

    # Stock specific fields (nullable)
    market_cap   = sa.Column(mysql.DECIMAL(precision=25, scale=4), nullable=True)
    sector       = sa.Column(sa.String(255), nullable=True)
    industry     = sa.Column(sa.String(255), nullable=True)
    earnings_release_trading_date_fq      = sa.Column(sa.DateTime, nullable=True)
    earnings_release_next_trading_date_fq = sa.Column(sa.DateTime, nullable=True)
    indexes      = sa.Column(sa.Text, nullable=True)

    # ETF specific fields (nullable)
    relative_volume     = sa.Column(sa.Float, nullable=True) # TradingView often uses float for relative values
    aum                 = sa.Column(mysql.DECIMAL(precision=25, scale=4), nullable=True) # Kept in model, but API name is different/unknown
    nav_total_return_3y = sa.Column(sa.Float, nullable=True) # Kept in model, but API name is different/unknown
    expense_ratio       = sa.Column(mysql.DECIMAL(precision=10, scale=4), nullable=True)
    asset_class         = sa.Column(sa.String(255), nullable=True)
    focus               = sa.Column(sa.String(255), nullable=True)

    daily_bars = relationship("YfBar1d", cascade="all, delete-orphan", passive_deletes=True)

    def __repr__(self):
        asset_type = "ETF" if self.is_etf else "Stock"
        return f"SymbolFields(symbol='{self.symbol}', type='{asset_type}', name='{self.company_name}')"
    

############################################################################################ Sector rotation data

class MmtvDailyBar(Base):
    __tablename__ = 'mmtv_daily_bars'

    date       = sa.Column(sa.DateTime, primary_key=True)
    field_name = sa.Column(sa.String(255), primary_key=True)  
    field_type = sa.Column(sa.String(255), primary_key=True)  
    open       = sa.Column(mysql.DOUBLE(precision=8, scale=2))  # Using DOUBLE to match double(8,2) in database
    high       = sa.Column(mysql.DOUBLE(precision=8, scale=2))  # Using DOUBLE to match double(8,2) in database
    low        = sa.Column(mysql.DOUBLE(precision=8, scale=2))  # Using DOUBLE to match double(8,2) in database
    close      = sa.Column(mysql.DOUBLE(precision=8, scale=2))  # Using DOUBLE to match double(8,2) in database

    def __repr__(self):
        return f"MmtvDailyBar(date='{self.date}', field_name='{self.field_name}', field_type='{self.field_type}', open={self.open}, high={self.high}, low={self.low}, close={self.close})"


############################################################################################ Etf data

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
    net_assets      = sa.Column(mysql.DECIMAL(precision=20, scale=2))  # Match production precision
    fund_type       = sa.Column(sa.String(255))
    provider        = sa.Column(sa.String(255))
    exchange        = sa.Column(sa.String(255))
    benchmark       = sa.Column(sa.String(255))
    created_at      = sa.Column(mysql.TIMESTAMP)  # Using TIMESTAMP to match production
    updated_at      = sa.Column(mysql.TIMESTAMP)  # Using TIMESTAMP to match production

    # Define relationship with holdings
    holdings = relationship('IShareETFHolding', back_populates='ishare_etf', cascade="all, delete-orphan")

    def __repr__(self):
        return f"IShareETF(ticker='{self.ticker}', fund_name='{self.fund_name}')"
    

class IShareETFHolding(Base):
    __tablename__ = 'ishare_etf_holding'
    id               = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    ishare_etf_id    = sa.Column(sa.BigInteger, sa.ForeignKey('ishare_etf.id', ondelete='CASCADE'), nullable=False)
    ticker           = sa.Column(sa.String(255))
    name             = sa.Column(sa.String(255), nullable=False)
    sector           = sa.Column(sa.String(255), server_default='Other')
    asset_class      = sa.Column(sa.String(255), nullable=False)
    market_value     = sa.Column(mysql.DECIMAL(precision=20, scale=2), nullable=False)  # Match production precision
    weight           = sa.Column(mysql.DECIMAL(precision=10, scale=4), nullable=False)  # Match production precision
    notional_value   = sa.Column(mysql.DECIMAL(precision=20, scale=2), nullable=False)  # Match production precision
    amount           = sa.Column(mysql.DECIMAL(precision=20, scale=2), nullable=False)  # Match production precision
    price            = sa.Column(mysql.DECIMAL(precision=20, scale=2), nullable=False)  # Match production precision
    location         = sa.Column(sa.String(255))
    exchange         = sa.Column(sa.String(255))
    currency         = sa.Column(sa.String(255))
    fx_rate          = sa.Column(mysql.DECIMAL(precision=10, scale=4), nullable=False)  # Match production precision
    market_currency  = sa.Column(sa.String(255), server_default='USD')
    accrual_date     = sa.Column(sa.Date)
    fund_ticker      = sa.Column(sa.String(255), nullable=False)
    as_of_date       = sa.Column(sa.Date, nullable=False)
    created_at       = sa.Column(mysql.TIMESTAMP)  # Using TIMESTAMP to match production
    updated_at       = sa.Column(mysql.TIMESTAMP)  # Using TIMESTAMP to match production
    
    # Define relationship with parent ETF
    ishare_etf       = relationship('IShareETF', back_populates='holdings')
    
    __table_args__ = (
        sa.UniqueConstraint('ishare_etf_id', 'ticker', name='uix_ishare_etf_holding_etf_ticker'),
        sa.Index('ishare_etf_holding_ishare_etf_id_foreign', 'ishare_etf_id'),
    )
    
    def __repr__(self) -> str:
        return f"IShareETFHolding(ticker='{self.ticker}', ishare_etf_id={self.ishare_etf_id})"


############################################################################################ Financial details data
## income_statement, balance_sheet, cash_flow

class IncomeStatement(Base):
    __tablename__ = 'income_statements'
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(255), nullable=False)
    date                    = sa.Column(sa.DateTime, nullable=False)
    period_type             = sa.Column(sa.String(10), nullable=False)
    
    # Revenue and profitability - using mysql.DOUBLE to match 'double' in production
    total_revenue           = sa.Column(mysql.DOUBLE)
    cost_of_revenue         = sa.Column(mysql.DOUBLE)
    gross_profit            = sa.Column(mysql.DOUBLE)
    
    # Operating metrics
    operating_expense       = sa.Column(mysql.DOUBLE)
    operating_income        = sa.Column(mysql.DOUBLE)
    total_operating_income  = sa.Column(mysql.DOUBLE)
    total_expenses          = sa.Column(mysql.DOUBLE)
    
    # Non-operating items
    net_non_operating_interest = sa.Column(mysql.DOUBLE)
    other_income_expense    = sa.Column(mysql.DOUBLE)
    pretax_income           = sa.Column(mysql.DOUBLE)
    tax_provision           = sa.Column(mysql.DOUBLE)
    
    # Net income and EPS
    net_income              = sa.Column(mysql.DOUBLE)
    normalized_income       = sa.Column(mysql.DOUBLE)
    basic_shares            = sa.Column(mysql.DOUBLE)
    diluted_shares          = sa.Column(mysql.DOUBLE)
    basic_eps               = sa.Column(mysql.DOUBLE)
    diluted_eps             = sa.Column(mysql.DOUBLE)
    
    # Additional metrics
    ebit                    = sa.Column(mysql.DOUBLE)
    ebitda                  = sa.Column(mysql.DOUBLE)
    interest_income         = sa.Column(mysql.DOUBLE)
    interest_expense        = sa.Column(mysql.DOUBLE)
    net_interest_income     = sa.Column(mysql.DOUBLE)
    
    # Timestamps - use TIMESTAMP to match production
    last_updated            = sa.Column(mysql.TIMESTAMP, server_onupdate=sa.func.current_timestamp())
    
    _table_args_ = (
        sa.UniqueConstraint('symbol', 'date', 'period_type', name='uix_income_statement_key'),
        sa.Index('income_statements_symbol_index', 'symbol'),
    )
    
    def __repr__(self):
        return f"IncomeStatement(symbol='{self.symbol}', date={self.date})"


class BalanceSheet(Base):
    __tablename__ = 'balance_sheets'
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(255), nullable=False)
    date                    = sa.Column(sa.DateTime, nullable=False)
    period_type             = sa.Column(sa.String(10), nullable=False)
    
    # Main categories - using mysql.DOUBLE to match 'double' in production
    total_assets            = sa.Column(mysql.DOUBLE)
    total_liabilities       = sa.Column(mysql.DOUBLE)
    total_equity            = sa.Column(mysql.DOUBLE)
    
    # Additional metrics
    total_capitalization    = sa.Column(mysql.DOUBLE)
    common_stock_equity     = sa.Column(mysql.DOUBLE)
    capital_lease_obligations = sa.Column(mysql.DOUBLE)
    net_tangible_assets     = sa.Column(mysql.DOUBLE)
    working_capital         = sa.Column(mysql.DOUBLE)
    invested_capital        = sa.Column(mysql.DOUBLE)
    tangible_book_value     = sa.Column(mysql.DOUBLE)
    total_debt              = sa.Column(mysql.DOUBLE)
    
    # Share information
    shares_issued           = sa.Column(mysql.DOUBLE)
    ordinary_shares_number  = sa.Column(mysql.DOUBLE)
    
    # Timestamps - use TIMESTAMP to match production
    last_updated            = sa.Column(mysql.TIMESTAMP, server_onupdate=sa.func.current_timestamp())
    
    _table_args_ = (
        sa.UniqueConstraint('symbol', 'date', 'period_type', name='uix_balance_sheet_key'),
        sa.Index('balance_sheets_symbol_index', 'symbol'),
    )
    
    def __repr__(self):
        return f"BalanceSheet(symbol='{self.symbol}', date={self.date})"


class CashFlow(Base):
    __tablename__ = 'cash_flows'
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(255), nullable=False)
    date                    = sa.Column(sa.DateTime, nullable=False)
    period_type             = sa.Column(sa.String(10), nullable=False)
    
    # Main cash flow categories - using mysql.DOUBLE to match 'double' in production
    operating_cash_flow     = sa.Column(mysql.DOUBLE)
    investing_cash_flow     = sa.Column(mysql.DOUBLE)
    financing_cash_flow     = sa.Column(mysql.DOUBLE)
    free_cash_flow          = sa.Column(mysql.DOUBLE)
    end_cash_position       = sa.Column(mysql.DOUBLE)
    
    # Detailed items
    income_tax_paid         = sa.Column(mysql.DOUBLE)
    interest_paid           = sa.Column(mysql.DOUBLE)
    capital_expenditure     = sa.Column(mysql.DOUBLE)
    issuance_of_capital_stock = sa.Column(mysql.DOUBLE)
    issuance_of_debt        = sa.Column(mysql.DOUBLE)
    repayment_of_debt       = sa.Column(mysql.DOUBLE)
    
    # Timestamps - use TIMESTAMP to match production
    last_updated            = sa.Column(mysql.TIMESTAMP, server_onupdate=sa.func.current_timestamp())
    
    _table_args_ = (
        sa.UniqueConstraint('symbol', 'date', 'period_type', name='uix_cash_flow_key'),
        sa.Index('cash_flows_symbol_index', 'symbol'),
    )
    
    def __repr__(self):
        return f"CashFlow(symbol='{self.symbol}', date={self.date})"


############################################################################################ equity_user_histories data
class Equity2User(Base):
    __tablename__ = 'equity_user_histories'
    
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(255), nullable=False)
    risk_type               = sa.Column(sa.String(255), nullable=False)
    sector                  = sa.Column(sa.String(255))
    volume_spike            = sa.Column(sa.String(255), nullable=False)
    RSI                     = sa.Column(mysql.DOUBLE, nullable=False) 
    ADR                     = sa.Column(mysql.DOUBLE, nullable=False)
    long_term_persistance   = sa.Column(mysql.DOUBLE, nullable=False)
    long_term_divergence    = sa.Column(mysql.DOUBLE, nullable=False)
    earnings_date_score     = sa.Column(mysql.DOUBLE, nullable=False)
    income_statement_score  = sa.Column(mysql.DOUBLE, nullable=False)
    cashflow_statement_score = sa.Column(mysql.DOUBLE, nullable=False)
    balance_sheet_score     = sa.Column(mysql.DOUBLE, nullable=False)
    rate_scoring            = sa.Column(mysql.DOUBLE, nullable=False)
    buy_point               = sa.Column(mysql.DOUBLE, nullable=False)
    short_point             = sa.Column(mysql.DOUBLE, nullable=False)
    recommended_date        = sa.Column(mysql.TIMESTAMP, nullable=False) 
    is_active               = sa.Column(sa.Boolean, nullable=False)
    status                  = sa.Column(sa.String(20), nullable=False)
    overbuy_oversold        = sa.Column(mysql.DOUBLE, nullable=False)
    created_at              = sa.Column(mysql.TIMESTAMP) 
    updated_at              = sa.Column(mysql.TIMESTAMP) 

    def __repr__(self):
        return f"Equity2User(symbol='{self.symbol}', risk_type='{self.risk_type}', status='{self.status}')"


class EquityTechnicalIndicator(Base):
    __tablename__ = 'equity_technical_indicators_history'
    
    id               = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol           = sa.Column(sa.String(255), nullable=False)
    date             = sa.Column(mysql.TIMESTAMP, nullable=False)  # Changed to TIMESTAMP to match production
    mfi              = sa.Column(mysql.DOUBLE, nullable=False)  # Using DOUBLE to match 'double' in production
    trend_intensity  = sa.Column(mysql.DOUBLE, nullable=False)
    persistent_ratio = sa.Column(mysql.DOUBLE, nullable=False)
    created_at       = sa.Column(mysql.TIMESTAMP)  # Changed to TIMESTAMP to match production
    updated_at       = sa.Column(mysql.TIMESTAMP)  # Changed to TIMESTAMP to match production

    def __repr__(self):
        return f"EquityTechnicalIndicator(symbol='{self.symbol}', date={self.date})"


############################################################################################ Company data

class Company(Base):
    __tablename__ = 'companies'
    
    symbol          = sa.Column(sa.String(10), primary_key=True)
    company_name    = sa.Column(sa.String(255), nullable=False)
    description     = sa.Column(sa.Text, nullable=False)
    sector          = sa.Column(sa.String(100), nullable=False)
    industry        = sa.Column(sa.String(100), nullable=False)
    employees       = sa.Column(sa.Integer, nullable=False)
    website         = sa.Column(sa.String(255), nullable=False)
    created_at      = sa.Column(mysql.TIMESTAMP)  # Changed to TIMESTAMP to match production
    updated_at      = sa.Column(mysql.TIMESTAMP)  # Changed to TIMESTAMP to match production
    
    # Relationships
    executives      = relationship("Executive", back_populates="company", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"Company(symbol='{self.symbol}', company_name='{self.company_name}')"


class Executive(Base):
    __tablename__ = 'executives'
    
    id              = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    company_symbol  = sa.Column(sa.String(10), sa.ForeignKey("companies.symbol"))
    name            = sa.Column(sa.String(255), nullable=False)
    title           = sa.Column(sa.String(255), nullable=False)
    year_born       = sa.Column(sa.Integer, nullable=False)
    compensation    = sa.Column(mysql.DOUBLE(precision=8, scale=2), nullable=False)  # Changed to DOUBLE(8,2) to match production
    created_at      = sa.Column(mysql.TIMESTAMP)  # Changed to TIMESTAMP to match production
    updated_at      = sa.Column(mysql.TIMESTAMP)  # Changed to TIMESTAMP to match production
    
    # Relationships
    company         = relationship("Company", back_populates="executives")
    
    __table_args__ = (
        sa.UniqueConstraint('company_symbol', 'name', name='uix_executive_company_name'),
    )

    def __repr__(self):
        return f"Executive(name='{self.name}', company_symbol='{self.company_symbol}', title='{self.title}')"


############################################################################################ News Articles data

class NewsArticle(Base):
    __tablename__ = 'news_articles'
    
    id               = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol           = sa.Column(sa.String(10), nullable=False)
    title            = sa.Column(sa.String(500), nullable=False)
    publisher        = sa.Column(sa.String(100), nullable=False)
    link             = sa.Column(sa.String(1000), nullable=False)
    published_date   = sa.Column(sa.DateTime, nullable=False)
    type             = sa.Column(sa.String(50), nullable=False)
    related_symbols  = sa.Column(sa.String(200), nullable=False)
    preview_text     = sa.Column(sa.Text, nullable=False)
    created_at       = sa.Column(mysql.TIMESTAMP, nullable=True)
    updated_at       = sa.Column(mysql.TIMESTAMP, nullable=True)
    thumbnail        = sa.Column(sa.String(255), nullable=True)
    
    __table_args__ = (
        sa.UniqueConstraint('symbol', 'title', 'published_date', name='uix_news_1'),
        sa.Index('ix_news_articles_published_date', 'published_date'),
        sa.Index('ix_news_articles_symbol', 'symbol'),
    )
    
    def __repr__(self):
        return f"NewsArticle(symbol='{self.symbol}', title='{self.title[:30]}...', date='{self.published_date}')"
    


class EtfStockRsBenchmark(Base):
    __tablename__ = 'etf_stock_rs_benchmark'
    
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    stock_symbol            = sa.Column(sa.String(255), nullable=False, index=True)
    etf_symbol              = sa.Column(sa.String(255), nullable=False, index=True)
    ishare_etf_id           = sa.Column(sa.BigInteger, sa.ForeignKey('ishare_etf.id', ondelete='SET NULL'), nullable=True)
    market_value            = sa.Column(mysql.DECIMAL(precision=20, scale=2), nullable=True)
    weight                  = sa.Column(mysql.DECIMAL(precision=10, scale=4), nullable=True)
    
    # Relative strength metrics
    rs_current              = sa.Column(mysql.DOUBLE, nullable=True)
    rs_mean_100d            = sa.Column(mysql.DOUBLE, nullable=True, index=True)
    rs_mean_50d             = sa.Column(mysql.DOUBLE, nullable=True)
    rs_mean_20d             = sa.Column(mysql.DOUBLE, nullable=True)
    rs_mean_5d              = sa.Column(mysql.DOUBLE, nullable=True)
    
    # Percentage of positive RS values
    rs_pos_pct_100d         = sa.Column(mysql.DOUBLE, nullable=True)
    rs_pos_pct_50d          = sa.Column(mysql.DOUBLE, nullable=True)
    rs_pos_pct_20d          = sa.Column(mysql.DOUBLE, nullable=True)
    rs_pos_pct_5d           = sa.Column(mysql.DOUBLE, nullable=True)
    
    # Lists stored as JSON arrays in TEXT fields
    rs_values_100d          = sa.Column(sa.Text, nullable=True)  # JSON array of values
    rs_values_50d           = sa.Column(sa.Text, nullable=True)  # JSON array of values
    rs_values_20d           = sa.Column(sa.Text, nullable=True)  # JSON array of values
    rs_values_5d            = sa.Column(sa.Text, nullable=True)  # JSON array of values
    
    benchmark_symbol        = sa.Column(sa.String(10), nullable=False, server_default='SPY')
    created_at              = sa.Column(mysql.TIMESTAMP, server_default=sa.func.now())
    updated_at              = sa.Column(mysql.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now())
    
    # Relationship to ishare_etf table
    ishare_etf              = relationship('IShareETF')
    
    def __repr__(self):
        return f"EtfStockRsBenchmark(stock='{self.stock_symbol}', etf='{self.etf_symbol}', rs_mean_100d={self.rs_mean_100d})"
    

class EquityRiskProfile(Base):
    __tablename__ = 'equity_risk_profile'
    
    id               = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol           = sa.Column(sa.String(255), nullable=False, index=True)
    is_etf           = sa.Column(sa.Boolean, default=False, nullable=False, index=True)
    price            = sa.Column(mysql.DECIMAL(precision=15, scale=4), nullable=True)
    risk_type        = sa.Column(sa.String(50), nullable=False, index=True)
    average_score    = sa.Column(mysql.DOUBLE, nullable=True)
    
    # 3-year averages
    volatility_3yr_avg    = sa.Column(mysql.DOUBLE, nullable=True)
    beta_3yr_avg          = sa.Column(mysql.DOUBLE, nullable=True)
    max_drawdown_3yr_avg  = sa.Column(mysql.DOUBLE, nullable=True)
    adr_3yr_avg           = sa.Column(mysql.DOUBLE, nullable=True)
    
    # Year by year metrics - Year3 is most recent
    volatility_year3  = sa.Column(mysql.DOUBLE, nullable=True)
    volatility_year2  = sa.Column(mysql.DOUBLE, nullable=True)
    volatility_year1  = sa.Column(mysql.DOUBLE, nullable=True)
    
    beta_year3        = sa.Column(mysql.DOUBLE, nullable=True)
    beta_year2        = sa.Column(mysql.DOUBLE, nullable=True)
    beta_year1        = sa.Column(mysql.DOUBLE, nullable=True)
    
    max_drawdown_year3 = sa.Column(mysql.DOUBLE, nullable=True)
    max_drawdown_year2 = sa.Column(mysql.DOUBLE, nullable=True)
    max_drawdown_year1 = sa.Column(mysql.DOUBLE, nullable=True)
    
    adr_year3         = sa.Column(mysql.DOUBLE, nullable=True)
    adr_year2         = sa.Column(mysql.DOUBLE, nullable=True)
    adr_year1         = sa.Column(mysql.DOUBLE, nullable=True)
    
    # Individual metric scores
    volatility_score  = sa.Column(mysql.DOUBLE, nullable=True)
    beta_score        = sa.Column(mysql.DOUBLE, nullable=True)
    drawdown_score    = sa.Column(mysql.DOUBLE, nullable=True)
    adr_score         = sa.Column(mysql.DOUBLE, nullable=True)
    
    # Timestamps
    classified_at    = sa.Column(sa.DateTime, nullable=True)
    created_at       = sa.Column(mysql.TIMESTAMP, server_default=sa.func.now())
    updated_at       = sa.Column(mysql.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now())
    
    __table_args__ = (
        sa.UniqueConstraint('symbol', 'classified_at', name='uix_equity_risk_profile_symbol_date'),
    )
    
    def __repr__(self):
        asset_type = "ETF" if self.is_etf else "Stock"
        return f"EquityRiskProfile(symbol='{self.symbol}', type='{asset_type}', risk_type='{self.risk_type}', avg_score={self.average_score})"
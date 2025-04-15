import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime

class Base(DeclarativeBase):
    pass

############################################################################################ Daily Bar data

class YfBar1d(Base):
    __tablename__ = 'yf_daily_bar'
    id        = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol    = sa.Column(sa.String(255))  
    timestamp = sa.Column(sa.DateTime)
    open      = sa.Column(sa.Numeric(10, 4))
    high      = sa.Column(sa.Numeric(10, 4))
    low       = sa.Column(sa.Numeric(10, 4))
    close     = sa.Column(sa.Numeric(10, 4))
    volume    = sa.Column(sa.BigInteger)
    __table_args__ = (sa.UniqueConstraint('symbol', 'timestamp', name='yf_daily_bar_symbol_timestamp_unique'),)
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
    symbol       = sa.Column(sa.String(255), unique=True, nullable=False)  
    company_name = sa.Column(sa.String(255))  
    price        = sa.Column(sa.Numeric(10, 4))
    change       = sa.Column(sa.Numeric(10, 4))
    volume       = sa.Column(sa.BigInteger)
    market_cap   = sa.Column(sa.Numeric(20, 2))
    market       = sa.Column(sa.String(100))  
    sector       = sa.Column(sa.String(100))  
    industry     = sa.Column(sa.String(100))  
    earnings_release_trading_date_fq      = sa.Column(sa.String(50))  
    earnings_release_next_trading_date_fq = sa.Column(sa.String(50))  
    indexes      = sa.Column(sa.Text)
    country      = sa.Column(sa.String(50))  
    exchange     = sa.Column(sa.String(50))  
    updated_at   = sa.Column(sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now())

    def __repr__(self):
        return f"SectorUpdate(symbol='{self.symbol}', sector='{self.sector}')"
    

############################################################################################ Sector rotation data

class MmtvDailyBar(Base):
    __tablename__ = 'mmtv_daily_bars'

    date       = sa.Column(sa.DateTime, primary_key=True)
    field_name = sa.Column(sa.String(255), primary_key=True)  
    field_type = sa.Column(sa.String(255), primary_key=True)  
    open       = sa.Column(sa.Numeric(8, 2))
    high       = sa.Column(sa.Numeric(8, 2))
    low        = sa.Column(sa.Numeric(8, 2))
    close      = sa.Column(sa.Numeric(8, 2))

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
    net_assets      = sa.Column(sa.Numeric(20, 2))
    fund_type       = sa.Column(sa.String(255))
    provider        = sa.Column(sa.String(255))
    exchange        = sa.Column(sa.String(255))
    benchmark       = sa.Column(sa.String(255))
    created_at      = sa.Column(sa.DateTime)
    updated_at      = sa.Column(sa.DateTime)

    def __repr__(self):
        return f"IShareETF(ticker='{self.ticker}', fund_name='{self.fund_name}')"
    

class IShareETFHolding(Base):
    __tablename__ = 'ishare_etf_holding'
    id               = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    ishare_etf_id    = sa.Column(sa.BigInteger, sa.ForeignKey('ishare_etf.id'), nullable=False)
    ticker           = sa.Column(sa.String(255))
    name             = sa.Column(sa.String(255), nullable=False)
    sector           = sa.Column(sa.String(255))
    asset_class      = sa.Column(sa.String(255), nullable=False)
    market_value     = sa.Column(sa.Numeric(20, 2), nullable=False)
    weight           = sa.Column(sa.Numeric(10, 4), nullable=False)
    notional_value   = sa.Column(sa.Numeric(20, 2), nullable=False)
    amount           = sa.Column(sa.Numeric(20, 2), nullable=False)
    price            = sa.Column(sa.Numeric(20, 2), nullable=False)
    location         = sa.Column(sa.String(255))
    exchange         = sa.Column(sa.String(255))
    currency         = sa.Column(sa.String(255))
    fx_rate          = sa.Column(sa.Numeric(10, 4), nullable=False)
    market_currency  = sa.Column(sa.String(255))
    accrual_date     = sa.Column(sa.Date)
    fund_ticker      = sa.Column(sa.String(255), nullable=False)
    as_of_date       = sa.Column(sa.Date, nullable=False)
    created_at       = sa.Column(sa.DateTime)
    updated_at       = sa.Column(sa.DateTime)
    ishare_etf       = relationship('IShareETF', back_populates='holdings')
    
    def __repr__(self) -> str:
        return f"IShareETFHolding(ticker='{self.ticker}', ishare_etf_id={self.ishare_etf_id})"

IShareETF.holdings = relationship('IShareETFHolding', order_by=IShareETFHolding.id, back_populates='ishare_etf')


############################################################################################ Financial details data
## income_statement, balance_sheet, cash_flow, news_articles

class IncomeStatement(Base):
    __tablename__ = 'income_statements'
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(255), index=True, nullable=False)
    date                    = sa.Column(sa.DateTime, nullable=False)
    period_type             = sa.Column(sa.String(10), nullable=False)
    
    # Revenue and profitability
    total_revenue           = sa.Column(sa.Numeric(precision=None))
    cost_of_revenue         = sa.Column(sa.Numeric(precision=None))
    gross_profit            = sa.Column(sa.Numeric(precision=None))
    
    # Operating metrics
    operating_expense       = sa.Column(sa.Numeric(precision=None))
    operating_income        = sa.Column(sa.Numeric(precision=None))
    total_operating_income  = sa.Column(sa.Numeric(precision=None))
    total_expenses          = sa.Column(sa.Numeric(precision=None))
    
    # Non-operating items
    net_non_operating_interest = sa.Column(sa.Numeric(precision=None))
    other_income_expense    = sa.Column(sa.Numeric(precision=None))
    pretax_income           = sa.Column(sa.Numeric(precision=None))
    tax_provision           = sa.Column(sa.Numeric(precision=None))
    
    # Net income and EPS
    net_income              = sa.Column(sa.Numeric(precision=None))
    normalized_income       = sa.Column(sa.Numeric(precision=None))
    basic_shares            = sa.Column(sa.Numeric(precision=None))
    diluted_shares          = sa.Column(sa.Numeric(precision=None))
    basic_eps               = sa.Column(sa.Numeric(precision=None))
    diluted_eps             = sa.Column(sa.Numeric(precision=None))
    
    # Additional metrics
    ebit                    = sa.Column(sa.Numeric(precision=None))
    ebitda                  = sa.Column(sa.Numeric(precision=None))
    interest_income         = sa.Column(sa.Numeric(precision=None))
    interest_expense        = sa.Column(sa.Numeric(precision=None))
    net_interest_income     = sa.Column(sa.Numeric(precision=None))
    
    # Timestamps
    last_updated            = sa.Column(sa.DateTime, onupdate=sa.func.now())
    
    def __repr__(self):
        return f"IncomeStatement(symbol='{self.symbol}', date={self.date})"


class BalanceSheet(Base):
    __tablename__ = 'balance_sheets'
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(255), index=True, nullable=False)
    date                    = sa.Column(sa.DateTime, nullable=False)
    period_type             = sa.Column(sa.String(10), nullable=False)
    
    # Main categories
    total_assets            = sa.Column(sa.Numeric(precision=None))
    total_liabilities       = sa.Column(sa.Numeric(precision=None))
    total_equity            = sa.Column(sa.Numeric(precision=None))
    
    # Additional metrics
    total_capitalization    = sa.Column(sa.Numeric(precision=None))
    common_stock_equity     = sa.Column(sa.Numeric(precision=None))
    capital_lease_obligations = sa.Column(sa.Numeric(precision=None))
    net_tangible_assets     = sa.Column(sa.Numeric(precision=None))
    working_capital         = sa.Column(sa.Numeric(precision=None))
    invested_capital        = sa.Column(sa.Numeric(precision=None))
    tangible_book_value     = sa.Column(sa.Numeric(precision=None))
    total_debt              = sa.Column(sa.Numeric(precision=None))
    
    # Share information
    shares_issued           = sa.Column(sa.Numeric(precision=None))
    ordinary_shares_number  = sa.Column(sa.Numeric(precision=None))
    
    # Timestamps
    last_updated            = sa.Column(sa.DateTime, onupdate=sa.func.now())
    
    def __repr__(self):
        return f"BalanceSheet(symbol='{self.symbol}', date={self.date})"


class CashFlow(Base):
    __tablename__ = 'cash_flows'
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(255), index=True, nullable=False)
    date                    = sa.Column(sa.DateTime, nullable=False)
    period_type             = sa.Column(sa.String(10), nullable=False)
    
    # Main cash flow categories
    operating_cash_flow     = sa.Column(sa.Numeric(precision=None))
    investing_cash_flow     = sa.Column(sa.Numeric(precision=None))
    financing_cash_flow     = sa.Column(sa.Numeric(precision=None))
    free_cash_flow          = sa.Column(sa.Numeric(precision=None))
    end_cash_position       = sa.Column(sa.Numeric(precision=None))
    
    # Detailed items
    income_tax_paid         = sa.Column(sa.Numeric(precision=None))
    interest_paid           = sa.Column(sa.Numeric(precision=None))
    capital_expenditure     = sa.Column(sa.Numeric(precision=None))
    issuance_of_capital_stock = sa.Column(sa.Numeric(precision=None))
    issuance_of_debt        = sa.Column(sa.Numeric(precision=None))
    repayment_of_debt       = sa.Column(sa.Numeric(precision=None))
    
    # Timestamps
    last_updated            = sa.Column(sa.DateTime, onupdate=sa.func.now())
    
    def __repr__(self):
        return f"CashFlow(symbol='{self.symbol}', date={self.date})"


class NewsArticle(Base):
    __tablename__ = 'news_articles'
    
    id               = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol           = sa.Column(sa.String(255), index=True)
    title            = sa.Column(sa.String(500))
    publisher        = sa.Column(sa.String(100))
    link             = sa.Column(sa.String(1000))
    published_date   = sa.Column(sa.DateTime, index=True)
    type             = sa.Column(sa.String(50))
    related_symbols  = sa.Column(sa.String(200))
    preview_text     = sa.Column(sa.Text)
    thumbnail        = sa.Column(sa.String(1000))
    created_at       = sa.Column(sa.DateTime, default=datetime.now)
    
    __table_args__ = (
        sa.UniqueConstraint('symbol', 'title', 'published_date', name='uix_news_1'),
    )
    
    def __repr__(self):
        return f"NewsArticles(symbol='{self.symbol}', title='{self.title[:30]}...', date='{self.published_date}')"


############################################################################################ equity_2_users data
class Equity2User(Base):
    __tablename__ = 'equity_user_histories'
    
    id                      = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(255), nullable=False)
    risk_type               = sa.Column(sa.String(255), nullable=False)
    sector                  = sa.Column(sa.String(255))
    volume_spike            = sa.Column(sa.String(255), nullable=False)
    RSI                     = sa.Column(sa.Numeric(precision=None), nullable=False)
    ADR                     = sa.Column(sa.Numeric(precision=None), nullable=False)
    long_term_persistance   = sa.Column(sa.Numeric(precision=None), nullable=False)
    long_term_divergence    = sa.Column(sa.Numeric(precision=None), nullable=False)
    earnings_date_score     = sa.Column(sa.Numeric(precision=None), nullable=False)
    income_statement_score  = sa.Column(sa.Numeric(precision=None), nullable=False)
    cashflow_statement_score = sa.Column(sa.Numeric(precision=None), nullable=False)
    balance_sheet_score     = sa.Column(sa.Numeric(precision=None), nullable=False)
    rate_scoring            = sa.Column(sa.Numeric(precision=None), nullable=False)
    buy_point               = sa.Column(sa.Numeric(precision=None), nullable=False)
    short_point             = sa.Column(sa.Numeric(precision=None), nullable=False)
    recommended_date        = sa.Column(sa.DateTime, nullable=False)
    is_active               = sa.Column(sa.Boolean, nullable=False)
    status                  = sa.Column(sa.String(20), nullable=False)
    overbuy_oversold        = sa.Column(sa.Numeric(precision=None), nullable=False)
    created_at              = sa.Column(sa.DateTime)
    updated_at              = sa.Column(sa.DateTime)

    def __repr__(self):
        return f"Equity2User(symbol='{self.symbol}', risk_type='{self.risk_type}', status='{self.status}')"


class EquityTechnicalIndicator(Base):
    __tablename__ = 'equity_technical_indicators_history'
    
    id               = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    symbol           = sa.Column(sa.String(255), nullable=False)
    date             = sa.Column(sa.DateTime, nullable=False)
    mfi              = sa.Column(sa.Numeric(precision=None), nullable=False)
    trend_intensity  = sa.Column(sa.Numeric(precision=None), nullable=False)
    persistent_ratio = sa.Column(sa.Numeric(precision=None), nullable=False)
    created_at       = sa.Column(sa.DateTime)
    updated_at       = sa.Column(sa.DateTime)

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
    created_at      = sa.Column(sa.DateTime)
    updated_at      = sa.Column(sa.DateTime)
    
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
    compensation    = sa.Column(sa.Numeric(8, 2), nullable=False)
    created_at      = sa.Column(sa.DateTime)
    updated_at      = sa.Column(sa.DateTime)
    
    # Relationships
    company         = relationship("Company", back_populates="executives")
    
    def __repr__(self):
        return f"Executive(name='{self.name}', company_symbol='{self.company_symbol}', title='{self.title}')"



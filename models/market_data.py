import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime

class Base(DeclarativeBase):
    pass

############################################################################################ Daily Bar data

class YfBar1d(Base):
    __tablename__ = 'yf_daily_bar'
    id        = sa.Column(sa.Integer, primary_key=True)
    symbol    = sa.Column(sa.String(20))  
    timestamp = sa.Column(sa.DateTime)
    open      = sa.Column(sa.Float)
    high      = sa.Column(sa.Float)
    low       = sa.Column(sa.Float)
    close     = sa.Column(sa.Float)
    volume    = sa.Column(sa.Float)
    __table_args__ = (sa.UniqueConstraint('symbol', 'timestamp'),)
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
    symbol       = sa.Column(sa.String(20), unique=True, nullable=False)  
    company_name = sa.Column(sa.String(255))  
    price        = sa.Column(sa.Float)
    change       = sa.Column(sa.Float)
    volume       = sa.Column(sa.Float)
    market_cap   = sa.Column(sa.Float)
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
    __tablename__ = 'mmtv_daily_bar'

    date       = sa.Column(sa.Date, primary_key=True)
    field_name = sa.Column(sa.String(100), primary_key=True)  
    field_type = sa.Column(sa.String(100), primary_key=True)  
    open       = sa.Column(sa.Float)
    high       = sa.Column(sa.Float)
    low        = sa.Column(sa.Float)
    close      = sa.Column(sa.Float)

    def __repr__(self):
        return f"MmtvDailyBar(date='{self.date}', field_name='{self.field_name}', field_type='{self.field_type}', open={self.open}, high={self.high}, low={self.low}, close={self.close})"


############################################################################################ Etf data

class IShareETF(Base):
    __tablename__ = 'ishare_etf'
    
    id              = sa.Column(sa.Integer, primary_key=True)
    fund_name       = sa.Column(sa.String(255), nullable=False)
    inception_date  = sa.Column(sa.Date)
    ticker          = sa.Column(sa.String(20), nullable=False, unique=True)
    cusip           = sa.Column(sa.String(9))
    isin            = sa.Column(sa.String(12))
    asset_class     = sa.Column(sa.String(50))
    subasset_class  = sa.Column(sa.String(50))
    country         = sa.Column(sa.String(50))
    region          = sa.Column(sa.String(50))
    product_url     = sa.Column(sa.String(255))
    product_id      = sa.Column(sa.String(50))
    net_assets      = sa.Column(sa.Float)
    fund_type       = sa.Column(sa.String(10))
    provider        = sa.Column(sa.String(50))
    exchange        = sa.Column(sa.String(50))
    benchmark       = sa.Column(sa.String(255))

    def __repr__(self):
        return f"IShareETF(ticker='{self.ticker}', fund_name='{self.fund_name}')"
    

class IShareETFHolding(Base):
    __tablename__ = 'ishare_etf_holding'
    id               = sa.Column(sa.Integer, primary_key=True)
    ishare_etf_id    = sa.Column(sa.Integer, sa.ForeignKey('ishare_etf.id'), nullable=False)
    ticker           = sa.Column(sa.String(20))
    name             = sa.Column(sa.String(255))
    sector           = sa.Column(sa.String(100))
    asset_class      = sa.Column(sa.String(100))
    market_value     = sa.Column(sa.Float)
    weight           = sa.Column(sa.Float)
    notional_value   = sa.Column(sa.Float)
    amount           = sa.Column(sa.Float)
    price            = sa.Column(sa.Float)
    location         = sa.Column(sa.String(100))
    exchange         = sa.Column(sa.String(100))
    currency         = sa.Column(sa.String(3))
    fx_rate          = sa.Column(sa.Float)
    market_currency  = sa.Column(sa.String(3))
    accrual_date     = sa.Column(sa.Date)
    fund_ticker      = sa.Column(sa.String(10))
    as_of_date       = sa.Column(sa.Date)
    ishare_etf       = relationship('IShareETF', back_populates='holdings')
    
    def __repr__(self) -> str:
        return f"IShareETFHolding(ticker='{self.ticker}', ishare_etf_id={self.ishare_etf_id})"

IShareETF.holdings = relationship('IShareETFHolding', order_by=IShareETFHolding.id , back_populates='ishare_etf')


############################################################################################ Financial details data
## income_statement, balance_sheet, cash_flow, news_articles

class IncomeStatement(Base):
    __tablename__ = 'income_statement'
    id                      = sa.Column(sa.Integer, primary_key=True)
    symbol                  = sa.Column(sa.String(10), index=True, nullable=False)
    date                    = sa.Column(sa.Date, nullable=False)
    period_type             = sa.Column(sa.String(10))  # 'Annual' or 'Quarterly'
    
    # Revenue and profitability
    revenue                 = sa.Column(sa.Float)
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
    eps                     = sa.Column(sa.Float)
    
    # Additional metrics
    ebit                    = sa.Column(sa.Float)
    ebitda                  = sa.Column(sa.Float)
    interest_income         = sa.Column(sa.Float)
    interest_expense        = sa.Column(sa.Float)
    net_interest_income     = sa.Column(sa.Float)
    
    # Timestamps
    created_at              = sa.Column(sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now())
    
    __table_args__ = (
        sa.UniqueConstraint("symbol", "date", "period_type", name="uix_income_1"),
    )
    
    def __repr__(self):
        return f"IncomeStatement(symbol='{self.symbol}', date={self.date})"


class BalanceSheet(Base):
    __tablename__ = 'balance_sheet'
    id                      = sa.Column(sa.Integer, primary_key=True)
    symbol                  = sa.Column(sa.String(10), index=True, nullable=False)
    date                    = sa.Column(sa.Date, nullable=False)
    period_type             = sa.Column(sa.String(10))
    
    # Main categories
    total_assets            = sa.Column(sa.Float)
    total_liabilities       = sa.Column(sa.Float)
    total_equity            = sa.Column(sa.Float)
    equity                  = sa.Column(sa.Float) 
    
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
    created_at              = sa.Column(sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now())
    
    __table_args__ = (
        sa.UniqueConstraint("symbol", "date", "period_type", name="uix_balance_1"),
    )
    
    def __repr__(self):
        return f"BalanceSheet(symbol='{self.symbol}', date={self.date})"


class CashFlow(Base):
    __tablename__ = 'cash_flow'
    id                      = sa.Column(sa.Integer, primary_key=True)
    symbol                  = sa.Column(sa.String(10), index=True, nullable=False)
    date                    = sa.Column(sa.Date, nullable=False)
    period_type             = sa.Column(sa.String(10))
    
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
    created_at              = sa.Column(sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now())
    
    __table_args__ = (
        sa.UniqueConstraint("symbol", "date", "period_type", name="uix_cash_1"),
    )
    
    def __repr__(self):
        return f"CashFlow(symbol='{self.symbol}', date={self.date})"


class NewsArticle(Base):
    __tablename__ = 'news_articles'
    
    id               = sa.Column(sa.Integer, primary_key=True)
    symbol           = sa.Column(sa.String(10), index=True)
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
    __tablename__ = 'equity_2_users'
    
    id                      = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    symbol                  = sa.Column(sa.String(10), index=True)
    risk_type               = sa.Column(sa.String(20))  # 'Conservative', 'Aggressive', 'Moderate'
    sector                  = sa.Column(sa.String(100))
    volume_spike            = sa.Column(sa.String(20))
    RSI                     = sa.Column(sa.Float)
    ADR                     = sa.Column(sa.Float)
    long_term_persistance   = sa.Column(sa.Float)
    long_term_divergence    = sa.Column(sa.Float)
    earnings_date_score     = sa.Column(sa.Float, default=0)
    income_statement_score  = sa.Column(sa.Float, default=0)
    cashflow_statement_score = sa.Column(sa.Float, default=0)
    balance_sheet_score     = sa.Column(sa.Float, default=0)
    rate_scoring            = sa.Column(sa.Float, default=0)
    buy_point               = sa.Column(sa.Float)
    short_point             = sa.Column(sa.Float)
    recommended_date        = sa.Column(sa.DateTime, default=datetime.now)
    is_active               = sa.Column(sa.Boolean, default=True)
    status                  = sa.Column(sa.String(20), default='')
    overbuy_oversold        = sa.Column(sa.Float)  # Scale from -100 to 100: negative = oversold, positive = overbought
    created_at              = sa.Column(sa.DateTime, default=datetime.now)
    updated_at              = sa.Column(sa.DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f"Equity2User(symbol='{self.symbol}', risk_type='{self.risk_type}', status='{self.status}')"


class EquityTechnicalIndicator(Base):
    __tablename__ = 'equity_technical_indicators'
    
    id               = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    symbol           = sa.Column(sa.String(10), index=True)
    date             = sa.Column(sa.Date, index=True)  # Date for this technical data point
    mfi              = sa.Column(sa.Float)  # Money Flow Index
    trend_intensity  = sa.Column(sa.Float)
    persistent_ratio = sa.Column(sa.Float)
    created_at       = sa.Column(sa.DateTime, default=datetime.now)
    updated_at       = sa.Column(sa.DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f"EquityTechnicalIndicator(symbol='{self.symbol}', date={self.date})"


############################################################################################ Company data

class Company(Base):
    __tablename__ = 'companies'
    
    symbol          = sa.Column(sa.String(10), primary_key=True)
    company_name    = sa.Column(sa.String(255))
    description     = sa.Column(sa.Text())
    sector          = sa.Column(sa.String(100))
    industry        = sa.Column(sa.String(100))
    employees       = sa.Column(sa.Integer)
    website         = sa.Column(sa.String(255))
    updated_at      = sa.Column(sa.String(50))
    
    # Relationships
    executives      = relationship("Executive", back_populates="company", cascade="all, delete-orphan")
    
    __table_args__ = (
        {"mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"},
    )
    
    def __repr__(self):
        return f"Company(symbol='{self.symbol}', company_name='{self.company_name}')"


class Executive(Base):
    __tablename__ = 'executives'
    
    id              = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    company_symbol  = sa.Column(sa.String(10), sa.ForeignKey("companies.symbol"))
    name            = sa.Column(sa.String(255))
    title           = sa.Column(sa.String(255))
    year_born       = sa.Column(sa.Integer)
    compensation    = sa.Column(sa.Float)
    
    # Relationships
    company         = relationship("Company", back_populates="executives")
    
    __table_args__ = (
        {"mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"},
    )
    
    def __repr__(self):
        return f"Executive(name='{self.name}', company_symbol='{self.company_symbol}', title='{self.title}')"



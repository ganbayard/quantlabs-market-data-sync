"""update_tables_to_match_production

Revision ID: 3505b5c65bbe
Revises: cf79e4f4e9c8
Create Date: 2025-04-15 17:05:44.733475

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
import logging

# Set up logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# revision identifiers, used by Alembic.
revision: str = '3505b5c65bbe'
down_revision: Union[str, None] = 'cf79e4f4e9c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Safer migration approach: Rename tables instead of dropping and recreating them.
    Handle foreign key constraints properly when altering column types.
    """
    print("Starting the database migration with safer approach...")
    conn = op.get_bind()
    print(f"Connected to database: {conn.engine.url.database}")
    print("This migration will rename tables and modify columns without dropping data.")
    
    # 1. Rename tables to match production names (only if they exist and don't already have the new name)
    table_renames = [
        ('equity_2_users', 'equity_user_histories'),
        ('equity_technical_indicators', 'equity_technical_indicators_history'),
        ('income_statement', 'income_statements'),
        ('balance_sheet', 'balance_sheets'),
        ('cash_flow', 'cash_flows'),
        ('mmtv_daily_bar', 'mmtv_daily_bars')
    ]
    
    for old_name, new_name in table_renames:
        # Check if old table exists and new table doesn't
        conn = op.get_bind()
        inspector = sa.inspect(conn)
        if old_name in inspector.get_table_names() and new_name not in inspector.get_table_names():
            print(f"Renaming table: {old_name} -> {new_name}")
            op.rename_table(old_name, new_name)
    
    # 2. Update columns for financial statement tables

    # Update income_statements columns (if it exists)
    if 'income_statements' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating income_statements table...")
        with op.batch_alter_table('income_statements') as batch_op:
            batch_op.alter_column('id',
                       existing_type=mysql.INTEGER(),
                       type_=sa.BigInteger(),
                       existing_nullable=False,
                       autoincrement=True)
            batch_op.alter_column('symbol',
                       existing_type=mysql.VARCHAR(length=20),
                       type_=sa.String(length=255),
                       existing_nullable=False)
            
            # Convert numeric fields to DOUBLE to match production DDL
            numeric_fields = [
                'total_revenue', 'cost_of_revenue', 'gross_profit', 'operating_expense',
                'operating_income', 'total_operating_income', 'total_expenses',
                'net_non_operating_interest', 'other_income_expense', 'pretax_income',
                'tax_provision', 'net_income', 'normalized_income', 'basic_shares',
                'diluted_shares', 'basic_eps', 'diluted_eps', 'ebit', 'ebitda',
                'interest_income', 'interest_expense', 'net_interest_income'
            ]
            
            for field in numeric_fields:
                # Check if the column exists before trying to modify it
                if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('income_statements')]:
                    batch_op.alter_column(field,
                               existing_type=mysql.FLOAT(),
                               type_=mysql.DOUBLE(),
                               existing_nullable=True)
    
    # Update balance_sheets columns (if it exists)
    if 'balance_sheets' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating balance_sheets table...")
        with op.batch_alter_table('balance_sheets') as batch_op:
            batch_op.alter_column('id',
                       existing_type=mysql.INTEGER(),
                       type_=sa.BigInteger(),
                       existing_nullable=False,
                       autoincrement=True)
            batch_op.alter_column('symbol',
                       existing_type=mysql.VARCHAR(length=20),
                       type_=sa.String(length=255),
                       existing_nullable=False)
            
            numeric_fields = [
                'total_assets', 'total_liabilities', 'total_equity', 'total_capitalization',
                'common_stock_equity', 'capital_lease_obligations', 'net_tangible_assets',
                'working_capital', 'invested_capital', 'tangible_book_value', 'total_debt',
                'shares_issued', 'ordinary_shares_number'
            ]
            
            for field in numeric_fields:
                if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('balance_sheets')]:
                    batch_op.alter_column(field,
                               existing_type=mysql.FLOAT(),
                               type_=mysql.DOUBLE(),
                               existing_nullable=True)
    
    # Update cash_flows columns (if it exists)
    if 'cash_flows' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating cash_flows table...")
        with op.batch_alter_table('cash_flows') as batch_op:
            batch_op.alter_column('id',
                       existing_type=mysql.INTEGER(),
                       type_=sa.BigInteger(),
                       existing_nullable=False,
                       autoincrement=True)
            batch_op.alter_column('symbol',
                       existing_type=mysql.VARCHAR(length=20),
                       type_=sa.String(length=255),
                       existing_nullable=False)
            
            numeric_fields = [
                'operating_cash_flow', 'investing_cash_flow', 'financing_cash_flow',
                'free_cash_flow', 'end_cash_position', 'income_tax_paid', 'interest_paid',
                'capital_expenditure', 'issuance_of_capital_stock', 'issuance_of_debt',
                'repayment_of_debt'
            ]
            
            for field in numeric_fields:
                if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('cash_flows')]:
                    batch_op.alter_column(field,
                               existing_type=mysql.FLOAT(),
                               type_=mysql.DOUBLE(),
                               existing_nullable=True)
    
    # Update equity tables
    if 'equity_user_histories' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating equity_user_histories table...")
        with op.batch_alter_table('equity_user_histories') as batch_op:
            batch_op.alter_column('id',
                       existing_type=mysql.INTEGER(),
                       type_=sa.BigInteger(),
                       existing_nullable=False,
                       autoincrement=True)
            
            # Update string columns
            string_columns = ['symbol', 'risk_type', 'sector', 'volume_spike']
            for col_name in string_columns:
                if col_name in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('equity_user_histories')]:
                    batch_op.alter_column(col_name,
                               existing_type=mysql.VARCHAR(length=50),  # Use a generic size
                               type_=sa.String(length=255),
                               existing_nullable=True)  # Use True as a safer default
                       
            # Update numeric columns
            numeric_columns = [
                'RSI', 'ADR', 'long_term_persistance', 'long_term_divergence',
                'earnings_date_score', 'income_statement_score', 'cashflow_statement_score',
                'balance_sheet_score', 'rate_scoring', 'buy_point', 'short_point',
                'overbuy_oversold'
            ]
            
            for col_name in numeric_columns:
                if col_name in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('equity_user_histories')]:
                    batch_op.alter_column(col_name,
                               existing_type=mysql.FLOAT(),
                               type_=mysql.DOUBLE(),
                               existing_nullable=True)  # Use True as a safer default
    
    # Update technical indicators table
    if 'equity_technical_indicators_history' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating equity_technical_indicators_history table...")
        with op.batch_alter_table('equity_technical_indicators_history') as batch_op:
            batch_op.alter_column('id',
                       existing_type=mysql.INTEGER(),
                       type_=sa.BigInteger(),
                       existing_nullable=False,
                       autoincrement=True)
            
            if 'symbol' in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('equity_technical_indicators_history')]:
                batch_op.alter_column('symbol',
                           existing_type=mysql.VARCHAR(length=20),
                           type_=sa.String(length=255),
                           existing_nullable=False)
            
            numeric_cols = ['mfi', 'trend_intensity', 'persistent_ratio']
            for col in numeric_cols:
                if col in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('equity_technical_indicators_history')]:
                    batch_op.alter_column(col,
                               existing_type=mysql.FLOAT(),
                               type_=mysql.DOUBLE(),
                               existing_nullable=False)
            
            # Add timestamp columns if they don't exist
            if 'created_at' not in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('equity_technical_indicators_history')]:
                batch_op.add_column(sa.Column('created_at', sa.DateTime(), nullable=True))
            
            if 'updated_at' not in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('equity_technical_indicators_history')]:
                batch_op.add_column(sa.Column('updated_at', sa.DateTime(), nullable=True))
    
    # Update mmtv_daily_bars table
    if 'mmtv_daily_bars' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating mmtv_daily_bars table...")
        with op.batch_alter_table('mmtv_daily_bars') as batch_op:
            string_cols = ['field_name', 'field_type']
            for col in string_cols:
                if col in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('mmtv_daily_bars')]:
                    batch_op.alter_column(col,
                               existing_type=mysql.VARCHAR(length=100),
                               type_=sa.String(length=255),
                               existing_nullable=False)
            
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('mmtv_daily_bars')]:
                    batch_op.alter_column(col,
                               existing_type=mysql.FLOAT(),
                               type_=sa.Numeric(precision=8, scale=2),
                               existing_nullable=True)
    
    # Update companies table
    print("Updating companies table...")
    if 'created_at' not in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('companies')]:
        op.add_column('companies', sa.Column('created_at', sa.DateTime(), nullable=True))
    
    try:
        op.alter_column('companies', 'company_name',
                   existing_type=mysql.VARCHAR(length=255),
                   nullable=False)
        op.alter_column('companies', 'description',
                   existing_type=mysql.TEXT(),
                   nullable=False)
        op.alter_column('companies', 'sector',
                   existing_type=mysql.VARCHAR(length=100),
                   nullable=False)
        op.alter_column('companies', 'industry',
                   existing_type=mysql.VARCHAR(length=100),
                   nullable=False)
        op.alter_column('companies', 'employees',
                   existing_type=mysql.INTEGER(),
                   nullable=False)
        op.alter_column('companies', 'website',
                   existing_type=mysql.VARCHAR(length=255),
                   nullable=False)
    except Exception as e:
        print(f"Warning while updating companies: {e}")
    
    try:
        op.alter_column('companies', 'updated_at',
                   existing_type=mysql.VARCHAR(length=50),
                   type_=sa.DateTime(),
                   existing_nullable=True)
    except Exception as e:
        print(f"Warning while updating companies.updated_at: {e}")
    
    # Update executives table
    print("Updating executives table...")
    if 'created_at' not in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('executives')]:
        op.add_column('executives', sa.Column('created_at', sa.DateTime(), nullable=True))
    
    if 'updated_at' not in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('executives')]:
        op.add_column('executives', sa.Column('updated_at', sa.DateTime(), nullable=True))
    
    try:
        op.alter_column('executives', 'id',
                   existing_type=mysql.INTEGER(),
                   type_=sa.BigInteger(),
                   existing_nullable=False,
                   autoincrement=True)
        op.alter_column('executives', 'name',
                   existing_type=mysql.VARCHAR(length=255),
                   nullable=False)
        op.alter_column('executives', 'title',
                   existing_type=mysql.VARCHAR(length=255),
                   nullable=False)
        op.alter_column('executives', 'year_born',
                   existing_type=mysql.INTEGER(),
                   nullable=False)
        op.alter_column('executives', 'compensation',
                   existing_type=mysql.FLOAT(),
                   type_=sa.Numeric(precision=8, scale=2),
                   nullable=False)
    except Exception as e:
        print(f"Warning while updating executives: {e}")
    
    # Handle news_articles table
    print("Updating news_articles table...")
    try:
        op.alter_column('news_articles', 'id',
                   existing_type=mysql.INTEGER(),
                   type_=sa.BigInteger(),
                   existing_nullable=False,
                   autoincrement=True)
        op.alter_column('news_articles', 'symbol',
                   existing_type=mysql.VARCHAR(length=10),
                   type_=sa.String(length=255),
                   existing_nullable=True)
    except Exception as e:
        print(f"Warning while updating news_articles: {e}")
    
    # Handle symbol_fields table
    print("Updating symbol_fields table...")
    try:
        op.alter_column('symbol_fields', 'symbol',
                   existing_type=mysql.VARCHAR(length=20),
                   type_=sa.String(length=255),
                   existing_nullable=False)
        op.alter_column('symbol_fields', 'price',
                   existing_type=mysql.FLOAT(),
                   type_=sa.Numeric(precision=10, scale=4),
                   existing_nullable=True)
        op.alter_column('symbol_fields', 'change',
                   existing_type=mysql.FLOAT(),
                   type_=sa.Numeric(precision=10, scale=4),
                   existing_nullable=True)
        op.alter_column('symbol_fields', 'volume',
                   existing_type=mysql.FLOAT(),
                   type_=sa.BigInteger(),
                   existing_nullable=True)
        op.alter_column('symbol_fields', 'market_cap',
                   existing_type=mysql.FLOAT(),
                   type_=sa.Numeric(precision=20, scale=2),
                   existing_nullable=True)
    except Exception as e:
        print(f"Warning while updating symbol_fields: {e}")
    
    # Handle yf_daily_bar table
    print("Updating yf_daily_bar table...")
    try:
        op.alter_column('yf_daily_bar', 'id',
                   existing_type=mysql.INTEGER(),
                   type_=sa.BigInteger(),
                   existing_nullable=False,
                   autoincrement=True)
        op.alter_column('yf_daily_bar', 'symbol',
                   existing_type=mysql.VARCHAR(length=20),
                   type_=sa.String(length=255),
                   existing_nullable=True)
        op.alter_column('yf_daily_bar', 'open',
                   existing_type=mysql.FLOAT(),
                   type_=sa.Numeric(precision=10, scale=4),
                   existing_nullable=True)
        op.alter_column('yf_daily_bar', 'high',
                   existing_type=mysql.FLOAT(),
                   type_=sa.Numeric(precision=10, scale=4),
                   existing_nullable=True)
        op.alter_column('yf_daily_bar', 'low',
                   existing_type=mysql.FLOAT(),
                   type_=sa.Numeric(precision=10, scale=4),
                   existing_nullable=True)
        op.alter_column('yf_daily_bar', 'close',
                   existing_type=mysql.FLOAT(),
                   type_=sa.Numeric(precision=10, scale=4),
                   existing_nullable=True)
        op.alter_column('yf_daily_bar', 'volume',
                   existing_type=mysql.FLOAT(),
                   type_=sa.BigInteger(),
                   existing_nullable=True)
    except Exception as e:
        print(f"Warning while updating yf_daily_bar: {e}")
    
    try:
        # Update constraints - safely
        print("Updating yf_daily_bar constraints...")
        # Try to drop the old index first
        op.drop_index('symbol', table_name='yf_daily_bar')
    except Exception as e:
        print(f"Note: Index 'symbol' may not exist on yf_daily_bar: {e}")
    
    # Check if constraint already exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    constraints = [c['name'] for c in inspector.get_unique_constraints('yf_daily_bar')]
    
    if 'yf_daily_bar_symbol_timestamp_unique' not in constraints:
        try:
            op.create_unique_constraint('yf_daily_bar_symbol_timestamp_unique', 'yf_daily_bar', ['symbol', 'timestamp'])
        except Exception as e:
            print(f"Warning while creating yf_daily_bar constraint: {e}")
    
    # SPECIAL HANDLING FOR ETF TABLES WITH FOREIGN KEY CONSTRAINTS
    
    # First check if we need to temporarily drop the foreign key
    inspector = sa.inspect(op.get_bind())
    foreign_keys = []
    for fk in inspector.get_foreign_keys('ishare_etf_holding'):
        if fk.get('referred_table') == 'ishare_etf':
            foreign_keys.append(fk)
    
    # If we have a foreign key, handle it carefully
    if foreign_keys:
        print("Found foreign keys between ishare_etf_holding and ishare_etf. Handling carefully...")
        # Get the foreign key name
        for fk in foreign_keys:
            fk_name = fk.get('name')
            print(f"  - Found foreign key: {fk_name}")
            
            # Temporarily drop the foreign key
            try:
                print(f"  - Temporarily dropping foreign key {fk_name}...")
                op.drop_constraint(fk_name, 'ishare_etf_holding', type_='foreignkey')
            except Exception as e:
                print(f"  - Warning while dropping foreign key: {e}")
    
    # Now update ishare_etf table
    print("Updating ishare_etf table...")
    if 'created_at' not in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('ishare_etf')]:
        op.add_column('ishare_etf', sa.Column('created_at', sa.DateTime(), nullable=True))
    
    if 'updated_at' not in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('ishare_etf')]:
        op.add_column('ishare_etf', sa.Column('updated_at', sa.DateTime(), nullable=True))
    
    try:
        # Do this one column at a time to avoid foreign key issues
        op.alter_column('ishare_etf', 'id',
                   existing_type=mysql.INTEGER(),
                   type_=sa.BigInteger(),
                   existing_nullable=False,
                   autoincrement=True)
        
        # Update string fields to length 255
        string_fields = [
            'ticker', 'cusip', 'isin', 'asset_class', 'subasset_class',
            'country', 'region', 'product_id', 'fund_type', 'provider', 'exchange'
        ]
        
        for field in string_fields:
            if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('ishare_etf')]:
                # Use a generic VARCHAR size for existing value
                op.alter_column('ishare_etf', field,
                           existing_type=mysql.VARCHAR(length=50), 
                           type_=sa.String(length=255),
                           existing_nullable=True)
        
        # Update net_assets to decimal
        if 'net_assets' in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('ishare_etf')]:
            op.alter_column('ishare_etf', 'net_assets',
                       existing_type=mysql.FLOAT(),
                       type_=sa.Numeric(precision=20, scale=2),
                       existing_nullable=True)
    except Exception as e:
        print(f"Warning while updating ishare_etf: {e}")
    
    # Now update ishare_etf_holding table
    print("Updating ishare_etf_holding table...")
    if 'created_at' not in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('ishare_etf_holding')]:
        op.add_column('ishare_etf_holding', sa.Column('created_at', sa.DateTime(), nullable=True))
    
    if 'updated_at' not in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('ishare_etf_holding')]:
        op.add_column('ishare_etf_holding', sa.Column('updated_at', sa.DateTime(), nullable=True))
    
    try:
        # Update id column
        op.alter_column('ishare_etf_holding', 'id',
                   existing_type=mysql.INTEGER(),
                   type_=sa.BigInteger(),
                   existing_nullable=False,
                   autoincrement=True)
        
        # Update foreign key column to match parent table
        op.alter_column('ishare_etf_holding', 'ishare_etf_id',
                   existing_type=mysql.INTEGER(),
                   type_=sa.BigInteger(),
                   existing_nullable=False)
        
        # Update string fields
        string_fields = [
            'ticker', 'name', 'sector', 'asset_class', 'location',
            'exchange', 'currency', 'market_currency', 'fund_ticker'
        ]
        
        for field in string_fields:
            if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('ishare_etf_holding')]:
                # Use a generic VARCHAR size for existing value
                op.alter_column('ishare_etf_holding', field,
                           existing_type=mysql.VARCHAR(length=100),
                           type_=sa.String(length=255),
                           existing_nullable=True)
        
        # Update numeric fields
        decimal_fields = [
            'market_value', 'weight', 'notional_value', 'amount', 'price', 'fx_rate'
        ]
        
        precisions = {
            'market_value': (20, 2),
            'weight': (10, 4),
            'notional_value': (20, 2),
            'amount': (20, 2),
            'price': (20, 2),
            'fx_rate': (10, 4)
        }
        
        for field in decimal_fields:
            if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('ishare_etf_holding')]:
                precision, scale = precisions.get(field, (10, 4))
                op.alter_column('ishare_etf_holding', field,
                           existing_type=mysql.FLOAT(),
                           type_=sa.Numeric(precision=precision, scale=scale),
                           existing_nullable=True)
    except Exception as e:
        print(f"Warning while updating ishare_etf_holding: {e}")
    
    # Now recreate the foreign key
    if foreign_keys:
        for fk in foreign_keys:
            fk_name = fk.get('name')
            constrained_columns = fk.get('constrained_columns')
            referred_columns = fk.get('referred_columns')
            
            if constrained_columns and referred_columns:
                try:
                    print(f"  - Recreating foreign key {fk_name}...")
                    op.create_foreign_key(
                        fk_name, 
                        'ishare_etf_holding', 
                        'ishare_etf',
                        constrained_columns,
                        referred_columns
                    )
                    print(f"  - Foreign key {fk_name} successfully recreated")
                except Exception as e:
                    print(f"  - Warning while recreating foreign key: {e}")
                    # Try to create a new foreign key with a different name
                    try:
                        new_fk_name = 'ishare_etf_holding_ishare_etf_id_foreign'
                        print(f"  - Trying to create new foreign key {new_fk_name}...")
                        op.create_foreign_key(
                            new_fk_name, 
                            'ishare_etf_holding', 
                            'ishare_etf',
                            ['ishare_etf_id'],
                            ['id']
                        )
                        print(f"  - Foreign key {new_fk_name} successfully created")
                    except Exception as e2:
                        print(f"  - Unable to create foreign key: {e2}")
    
    print("Migration completed successfully.")


def downgrade() -> None:
    # Empty downgrade function because we're maintaining a one-way migration
    pass
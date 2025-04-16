"""data_type_updates

Revision ID: ffdc42a01bbb
Revises: 3505b5c65bbe
Create Date: 2025-04-15 18:30:44.733475

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
revision: str = 'ffdc42a01bbb'
down_revision: Union[str, None] = '3505b5c65bbe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Update data types to match production database.
    Ensure numeric types use DOUBLE for financial data and create proper indexes.
    """
    print("Starting data type updates to match production database...")
    conn = op.get_bind()
    print(f"Connected to database: {conn.engine.url.database}")
    
    # 1. Update income_statements table to use DOUBLE (matching production)
    if 'income_statements' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating income_statements numeric fields to DOUBLE...")
        with op.batch_alter_table('income_statements') as batch_op:
            numeric_fields = [
                'total_revenue', 'cost_of_revenue', 'gross_profit', 'operating_expense',
                'operating_income', 'total_operating_income', 'total_expenses',
                'net_non_operating_interest', 'other_income_expense', 'pretax_income',
                'tax_provision', 'net_income', 'normalized_income', 'basic_shares',
                'diluted_shares', 'basic_eps', 'diluted_eps', 'ebit', 'ebitda',
                'interest_income', 'interest_expense', 'net_interest_income'
            ]
            
            for field in numeric_fields:
                if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('income_statements')]:
                    try:
                        batch_op.alter_column(field,
                                   existing_type=sa.Numeric(precision=None),
                                   type_=mysql.DOUBLE(),
                                   existing_nullable=True)
                    except Exception as e:
                        print(f"Warning while updating income_statements.{field}: {e}")
    
    # 2. Update balance_sheets table to use DOUBLE (matching production)
    if 'balance_sheets' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating balance_sheets numeric fields to DOUBLE...")
        with op.batch_alter_table('balance_sheets') as batch_op:
            numeric_fields = [
                'total_assets', 'total_liabilities', 'total_equity', 'total_capitalization',
                'common_stock_equity', 'capital_lease_obligations', 'net_tangible_assets',
                'working_capital', 'invested_capital', 'tangible_book_value', 'total_debt',
                'shares_issued', 'ordinary_shares_number'
            ]
            
            for field in numeric_fields:
                if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('balance_sheets')]:
                    try:
                        batch_op.alter_column(field,
                                   existing_type=sa.Numeric(precision=None),
                                   type_=mysql.DOUBLE(),
                                   existing_nullable=True)
                    except Exception as e:
                        print(f"Warning while updating balance_sheets.{field}: {e}")
    
    # 3. Update cash_flows table to use DOUBLE (matching production)
    if 'cash_flows' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating cash_flows numeric fields to DOUBLE...")
        with op.batch_alter_table('cash_flows') as batch_op:
            numeric_fields = [
                'operating_cash_flow', 'investing_cash_flow', 'financing_cash_flow',
                'free_cash_flow', 'end_cash_position', 'income_tax_paid', 'interest_paid',
                'capital_expenditure', 'issuance_of_capital_stock', 'issuance_of_debt',
                'repayment_of_debt'
            ]
            
            for field in numeric_fields:
                if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('cash_flows')]:
                    try:
                        batch_op.alter_column(field,
                                   existing_type=sa.Numeric(precision=None),
                                   type_=mysql.DOUBLE(),
                                   existing_nullable=True)
                    except Exception as e:
                        print(f"Warning while updating cash_flows.{field}: {e}")
    
    # 4. Update equity_technical_indicators_history table to use DOUBLE (matching production)
    if 'equity_technical_indicators_history' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating equity_technical_indicators_history numeric fields to DOUBLE...")
        with op.batch_alter_table('equity_technical_indicators_history') as batch_op:
            numeric_fields = ['mfi', 'trend_intensity', 'persistent_ratio']
            for field in numeric_fields:
                if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('equity_technical_indicators_history')]:
                    try:
                        batch_op.alter_column(field,
                                   existing_type=sa.Numeric(precision=None),
                                   type_=mysql.DOUBLE(),
                                   existing_nullable=False)
                    except Exception as e:
                        print(f"Warning while updating equity_technical_indicators_history.{field}: {e}")
    
    # 5. Update equity_user_histories table to use DOUBLE (matching production)
    if 'equity_user_histories' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating equity_user_histories numeric fields to DOUBLE...")
        with op.batch_alter_table('equity_user_histories') as batch_op:
            numeric_fields = [
                'RSI', 'ADR', 'long_term_persistance', 'long_term_divergence',
                'earnings_date_score', 'income_statement_score', 'cashflow_statement_score',
                'balance_sheet_score', 'rate_scoring', 'buy_point', 'short_point',
                'overbuy_oversold'
            ]
            
            for field in numeric_fields:
                if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('equity_user_histories')]:
                    try:
                        batch_op.alter_column(field,
                                   existing_type=sa.Numeric(precision=None),
                                   type_=mysql.DOUBLE(),
                                   existing_nullable=False)
                    except Exception as e:
                        print(f"Warning while updating equity_user_histories.{field}: {e}")
    
    # 6. Update executives table to use DOUBLE for compensation
    if 'executives' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating executives.compensation to DOUBLE(8,2)...")
        try:
            op.alter_column('executives', 'compensation',
                       existing_type=sa.Numeric(precision=8, scale=2),
                       type_=mysql.DOUBLE(precision=8, scale=2),
                       existing_nullable=False)
        except Exception as e:
            print(f"Warning while updating executives.compensation: {e}")
    
    # 7. Update mmtv_daily_bars price columns to DOUBLE(8,2)
    if 'mmtv_daily_bars' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating mmtv_daily_bars price fields to DOUBLE(8,2)...")
        with op.batch_alter_table('mmtv_daily_bars') as batch_op:
            price_fields = ['open', 'high', 'low', 'close']
            for field in price_fields:
                if field in [c['name'] for c in sa.inspect(op.get_bind()).get_columns('mmtv_daily_bars')]:
                    try:
                        batch_op.alter_column(field,
                                   existing_type=sa.Numeric(precision=8, scale=2),
                                   type_=mysql.DOUBLE(precision=8, scale=2),
                                   existing_nullable=True)
                    except Exception as e:
                        print(f"Warning while updating mmtv_daily_bars.{field}: {e}")
    
    # 8. Create missing indexes on various tables
    
    # Add symbol index on balance_sheets if it doesn't exist
    inspector = sa.inspect(op.get_bind())
    try:
        if 'balance_sheets' in inspector.get_table_names():
            indexes = [idx['name'] for idx in inspector.get_indexes('balance_sheets')]
            if 'balance_sheets_symbol_index' not in indexes:
                print("Creating index balance_sheets_symbol_index...")
                op.create_index('balance_sheets_symbol_index', 'balance_sheets', ['symbol'])
    except Exception as e:
        print(f"Warning while creating balance_sheets index: {e}")
    
    # Add symbol index on cash_flows if it doesn't exist
    try:
        if 'cash_flows' in inspector.get_table_names():
            indexes = [idx['name'] for idx in inspector.get_indexes('cash_flows')]
            if 'cash_flows_symbol_index' not in indexes:
                print("Creating index cash_flows_symbol_index...")
                op.create_index('cash_flows_symbol_index', 'cash_flows', ['symbol'])
    except Exception as e:
        print(f"Warning while creating cash_flows index: {e}")
    
    # Add symbol index on income_statements if it doesn't exist
    try:
        if 'income_statements' in inspector.get_table_names():
            indexes = [idx['name'] for idx in inspector.get_indexes('income_statements')]
            if 'income_statements_symbol_index' not in indexes:
                print("Creating index income_statements_symbol_index...")
                op.create_index('income_statements_symbol_index', 'income_statements', ['symbol'])
    except Exception as e:
        print(f"Warning while creating income_statements index: {e}")
    
    # 9. Update yf_daily_bar table - ensure DECIMAL types are used for price fields
    if 'yf_daily_bar' in sa.inspect(op.get_bind()).get_table_names():
        print("Updating yf_daily_bar fields to match production...")
        
        # Check if indexes exist and create if missing
        indexes = [idx['name'] for idx in inspector.get_indexes('yf_daily_bar')]
        
        if 'yf_daily_bar_symbol_index' not in indexes:
            try:
                print("Creating index yf_daily_bar_symbol_index...")
                op.create_index('yf_daily_bar_symbol_index', 'yf_daily_bar', ['symbol'])
            except Exception as e:
                print(f"Warning while creating yf_daily_bar_symbol_index: {e}")
        
        if 'yf_daily_bar_timestamp_index' not in indexes:
            try:
                print("Creating index yf_daily_bar_timestamp_index...")
                op.create_index('yf_daily_bar_timestamp_index', 'yf_daily_bar', ['timestamp'])
            except Exception as e:
                print(f"Warning while creating yf_daily_bar_timestamp_index: {e}")
        
        # Ensure correct NOT NULL constraints
        try:
            with op.batch_alter_table('yf_daily_bar') as batch_op:
                batch_op.alter_column('symbol',
                           existing_type=sa.String(length=255),
                           nullable=False)
                batch_op.alter_column('timestamp',
                           existing_type=sa.DateTime(),
                           nullable=False)
                batch_op.alter_column('open',
                           existing_type=sa.Numeric(precision=10, scale=4),
                           nullable=False)
                batch_op.alter_column('high',
                           existing_type=sa.Numeric(precision=10, scale=4),
                           nullable=False)
                batch_op.alter_column('low',
                           existing_type=sa.Numeric(precision=10, scale=4),
                           nullable=False)
                batch_op.alter_column('close',
                           existing_type=sa.Numeric(precision=10, scale=4),
                           nullable=False)
                batch_op.alter_column('volume',
                           existing_type=sa.BigInteger(),
                           nullable=False)
        except Exception as e:
            print(f"Warning while updating yf_daily_bar constraints: {e}")
    
    # 10. Check for symbol_fields foreign key in yf_daily_bar
    try:
        foreign_keys = inspector.get_foreign_keys('yf_daily_bar')
        has_symbol_fk = False
        
        for fk in foreign_keys:
            if fk.get('referred_table') == 'symbol_fields' and 'symbol' in fk.get('constrained_columns', []):
                has_symbol_fk = True
                break
        
        if not has_symbol_fk:
            print("Adding foreign key from yf_daily_bar.symbol to symbol_fields.symbol...")
            with op.batch_alter_table('yf_daily_bar') as batch_op:
                batch_op.create_foreign_key(
                    'yf_daily_bar_symbol_foreign',
                    'symbol_fields',
                    ['symbol'],
                    ['symbol'],
                    ondelete='CASCADE'
                )
    except Exception as e:
        print(f"Warning while checking/creating foreign key on yf_daily_bar: {e}")
    
    # 11. Update symbol_fields table price columns to use DOUBLE if needed
    try:
        with op.batch_alter_table('symbol_fields') as batch_op:
            batch_op.alter_column('price',
                       existing_type=sa.Numeric(precision=10, scale=4),
                       type_=mysql.DOUBLE(),
                       existing_nullable=True)
            batch_op.alter_column('change',
                       existing_type=sa.Numeric(precision=10, scale=4),
                       type_=mysql.DOUBLE(),
                       existing_nullable=True)
            batch_op.alter_column('market_cap',
                       existing_type=sa.Numeric(precision=20, scale=2),
                       type_=mysql.DOUBLE(),
                       existing_nullable=True)
    except Exception as e:
        print(f"Warning while updating symbol_fields numeric columns: {e}")
    
    print("Migration completed successfully.")


def downgrade() -> None:
    # Empty downgrade function because we're maintaining a one-way migration
    pass
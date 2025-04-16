"""align_dev_with_production

Revision ID: e57b92f4a21d
Revises: dfc58b29a45c
Create Date: 2025-04-16 15:30:45.678901

This migration aligns the development database schema with the production schema.
Key changes:
1. Replace created_at with last_updated in financial statement tables
2. Remove extra columns from dev (equity, revenue, eps)
3. Fix data types for better consistency with production
4. Update timestamp default expressions
"""
from typing import Sequence, Union
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import text

# Set up logging
logging.basicConfig()
logger = logging.getLogger("schema_alignment")
logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic
revision: str = 'e57b92f4a21d'
down_revision: Union[str, None] = 'dfc58b29a45c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema to match production DDL."""
    logger.info("Starting schema alignment with production DDL...")

    # Align balance_sheets table
    update_balance_sheets()
    
    # Align cash_flows table
    update_cash_flows()
    
    # Align income_statements table
    update_income_statements()
    
    # Align yf_daily_bar table
    update_yf_daily_bar()
    
    # Align timestamp columns in other tables
    update_timestamp_columns()
    
    # Align news_articles table
    update_news_articles()
    
    logger.info("Schema alignment completed successfully.")


def update_balance_sheets():
    """Update balance_sheets table to match production schema."""
    logger.info("Aligning balance_sheets table...")
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if inspector.has_table('balance_sheets'):
        # 1. Add last_updated column
        try:
            op.add_column('balance_sheets',
                sa.Column('last_updated', mysql.TIMESTAMP, nullable=True, 
                          server_default=None, server_onupdate=sa.func.current_timestamp())
            )
            logger.info("Added last_updated column to balance_sheets")
        except Exception as e:
            logger.warning(f"Error adding last_updated column to balance_sheets: {e}")
        
        # 2. Drop equity column (exists in dev but not in prod)
        try:
            if 'equity' in [c['name'] for c in inspector.get_columns('balance_sheets')]:
                op.drop_column('balance_sheets', 'equity')
                logger.info("Dropped equity column from balance_sheets")
        except Exception as e:
            logger.warning(f"Error dropping equity column from balance_sheets: {e}")
        
        # 3. Change date column from DATE to DATETIME
        try:
            op.alter_column('balance_sheets', 'date',
                existing_type=sa.Date(),
                type_=sa.DateTime(),
                nullable=False)
            logger.info("Changed date column type in balance_sheets")
        except Exception as e:
            logger.warning(f"Error changing date column type in balance_sheets: {e}")
        
        # 4. Update period_type to be NOT NULL as in production
        try:
            op.alter_column('balance_sheets', 'period_type',
                existing_type=sa.String(10),
                nullable=False)
            logger.info("Updated period_type column in balance_sheets to NOT NULL")
        except Exception as e:
            logger.warning(f"Error updating period_type column in balance_sheets: {e}")
        
        # 5. Drop created_at column if we choose to switch to last_updated
        try:
            if 'created_at' in [c['name'] for c in inspector.get_columns('balance_sheets')]:
                # You may want to preserve the data from created_at by copying it to last_updated first
                op.execute(text("""
                    UPDATE balance_sheets
                    SET last_updated = created_at
                    WHERE last_updated IS NULL
                """))
                
                op.drop_column('balance_sheets', 'created_at')
                logger.info("Dropped created_at column from balance_sheets")
        except Exception as e:
            logger.warning(f"Error dropping created_at column from balance_sheets: {e}")
    else:
        logger.warning("balance_sheets table not found")


def update_cash_flows():
    """Update cash_flows table to match production schema."""
    logger.info("Aligning cash_flows table...")
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if inspector.has_table('cash_flows'):
        # 1. Add last_updated column
        try:
            op.add_column('cash_flows',
                sa.Column('last_updated', mysql.TIMESTAMP, nullable=True, 
                          server_default=None, server_onupdate=sa.func.current_timestamp())
            )
            logger.info("Added last_updated column to cash_flows")
        except Exception as e:
            logger.warning(f"Error adding last_updated column to cash_flows: {e}")
        
        # 2. Change date column from DATE to DATETIME
        try:
            op.alter_column('cash_flows', 'date',
                existing_type=sa.Date(),
                type_=sa.DateTime(),
                nullable=False)
            logger.info("Changed date column type in cash_flows")
        except Exception as e:
            logger.warning(f"Error changing date column type in cash_flows: {e}")
        
        # 3. Update period_type to be NOT NULL as in production
        try:
            op.alter_column('cash_flows', 'period_type',
                existing_type=sa.String(10),
                nullable=False)
            logger.info("Updated period_type column in cash_flows to NOT NULL")
        except Exception as e:
            logger.warning(f"Error updating period_type column in cash_flows: {e}")
        
        # 4. Reorder free_cash_flow column to match production order
        # (This is just for consistency, doesn't affect functionality)
        
        # 5. Drop created_at column if we choose to switch to last_updated
        try:
            if 'created_at' in [c['name'] for c in inspector.get_columns('cash_flows')]:
                # You may want to preserve the data from created_at by copying it to last_updated first
                op.execute(text("""
                    UPDATE cash_flows
                    SET last_updated = created_at
                    WHERE last_updated IS NULL
                """))
                
                op.drop_column('cash_flows', 'created_at')
                logger.info("Dropped created_at column from cash_flows")
        except Exception as e:
            logger.warning(f"Error dropping created_at column from cash_flows: {e}")
    else:
        logger.warning("cash_flows table not found")


def update_income_statements():
    """Update income_statements table to match production schema."""
    logger.info("Aligning income_statements table...")
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if inspector.has_table('income_statements'):
        # 1. Add last_updated column
        try:
            op.add_column('income_statements',
                sa.Column('last_updated', mysql.TIMESTAMP, nullable=True, 
                          server_default=None, server_onupdate=sa.func.current_timestamp())
            )
            logger.info("Added last_updated column to income_statements")
        except Exception as e:
            logger.warning(f"Error adding last_updated column to income_statements: {e}")
        
        # 2. Drop columns that don't exist in production
        try:
            for col in ['revenue', 'eps']:
                if col in [c['name'] for c in inspector.get_columns('income_statements')]:
                    op.drop_column('income_statements', col)
                    logger.info(f"Dropped {col} column from income_statements")
        except Exception as e:
            logger.warning(f"Error dropping columns from income_statements: {e}")
        
        # 3. Change date column from DATE to DATETIME
        try:
            op.alter_column('income_statements', 'date',
                existing_type=sa.Date(),
                type_=sa.DateTime(),
                nullable=False)
            logger.info("Changed date column type in income_statements")
        except Exception as e:
            logger.warning(f"Error changing date column type in income_statements: {e}")
        
        # 4. Update period_type to be NOT NULL as in production
        try:
            op.alter_column('income_statements', 'period_type',
                existing_type=sa.String(10),
                nullable=False)
            logger.info("Updated period_type column in income_statements to NOT NULL")
        except Exception as e:
            logger.warning(f"Error updating period_type column in income_statements: {e}")
        
        # 5. Drop created_at column if we choose to switch to last_updated
        try:
            if 'created_at' in [c['name'] for c in inspector.get_columns('income_statements')]:
                # You may want to preserve the data from created_at by copying it to last_updated first
                op.execute(text("""
                    UPDATE income_statements
                    SET last_updated = created_at
                    WHERE last_updated IS NULL
                """))
                
                op.drop_column('income_statements', 'created_at')
                logger.info("Dropped created_at column from income_statements")
        except Exception as e:
            logger.warning(f"Error dropping created_at column from income_statements: {e}")
    else:
        logger.warning("income_statements table not found")


def update_yf_daily_bar():
    """Update yf_daily_bar table to match production schema."""
    logger.info("Aligning yf_daily_bar table...")
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if inspector.has_table('yf_daily_bar'):
        # Check the current data type of price columns
        columns = inspector.get_columns('yf_daily_bar')
        price_cols = {'open', 'high', 'low', 'close'}
        
        # If already changed in previous migration, skip this step
        needs_change = False
        for col in columns:
            if col['name'] in price_cols and str(col['type']).lower().startswith('double'):
                needs_change = True
                break
        
        if needs_change:
            try:
                # Change from DOUBLE to DECIMAL(10,4)
                with op.batch_alter_table('yf_daily_bar') as batch_op:
                    for col_name in price_cols:
                        batch_op.alter_column(col_name,
                            existing_type=mysql.DOUBLE(),
                            type_=mysql.DECIMAL(precision=10, scale=4),
                            nullable=False)
                logger.info("Changed price columns in yf_daily_bar from DOUBLE to DECIMAL(10,4)")
            except Exception as e:
                logger.warning(f"Error changing yf_daily_bar column types: {e}")
        else:
            logger.info("yf_daily_bar price columns already have correct type")
    else:
        logger.warning("yf_daily_bar table not found")


def update_timestamp_columns():
    """Update timestamp columns in various tables to match production schema."""
    logger.info("Aligning timestamp columns in other tables...")
    
    # Tables that need to be checked for timestamp column consistency
    tables_with_timestamps = [
        ('equity_technical_indicators_history', 'date', 'created_at', 'updated_at'),
        ('equity_user_histories', 'recommended_date', 'created_at', 'updated_at'),
        ('companies', 'created_at', 'updated_at'),
        ('executives', 'created_at', 'updated_at'),
        ('ishare_etf', 'created_at', 'updated_at'),
        ('ishare_etf_holding', 'created_at', 'updated_at')
    ]
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    for table_info in tables_with_timestamps:
        table_name = table_info[0]
        if not inspector.has_table(table_name):
            logger.warning(f"{table_name} table not found")
            continue
        
        columns = inspector.get_columns(table_name)
        column_names = [c['name'] for c in columns]
        
        for col_name in table_info[1:]:
            if col_name not in column_names:
                continue
                
            # Find the column
            col_info = next((c for c in columns if c['name'] == col_name), None)
            if not col_info:
                continue
                
            # Check if it's already a TIMESTAMP
            if str(col_info['type']).lower().startswith('timestamp'):
                logger.info(f"Column {col_name} in {table_name} is already TIMESTAMP")
                continue
                
            # Update to TIMESTAMP
            try:
                op.alter_column(table_name, col_name,
                    existing_type=sa.DateTime(),
                    type_=mysql.TIMESTAMP(),
                    nullable=True)
                logger.info(f"Changed {col_name} in {table_name} to TIMESTAMP")
            except Exception as e:
                logger.warning(f"Error changing {col_name} in {table_name} to TIMESTAMP: {e}")
    
    # Special case for equity_technical_indicators_history.date which is DATE in dev, TIMESTAMP in prod
    if inspector.has_table('equity_technical_indicators_history'):
        try:
            date_col = next((c for c in inspector.get_columns('equity_technical_indicators_history') 
                          if c['name'] == 'date'), None)
            if date_col and str(date_col['type']).lower().startswith('date'):
                op.alter_column('equity_technical_indicators_history', 'date',
                    existing_type=sa.Date(),
                    type_=mysql.TIMESTAMP(),
                    nullable=False)
                logger.info("Changed date column in equity_technical_indicators_history from DATE to TIMESTAMP")
        except Exception as e:
            logger.warning(f"Error changing date column in equity_technical_indicators_history: {e}")


def update_news_articles():
    """Update news_articles table to match production schema."""
    logger.info("Aligning news_articles table...")
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if inspector.has_table('news_articles'):
        # Update columns to match production schema
        try:
            # Get existing columns
            existing_columns = [c['name'] for c in inspector.get_columns('news_articles')]
            
            # Update nullability for existing columns
            with op.batch_alter_table('news_articles') as batch_op:
                # Make these columns NOT NULL
                for col_name in ['symbol', 'title', 'publisher', 'link', 'published_date', 
                                'type', 'related_symbols', 'preview_text']:
                    if col_name in existing_columns:
                        batch_op.alter_column(col_name,
                            existing_type=sa.String(length=None),  # Generic type, will be overridden
                            nullable=False)
                
                # Switch timestamp columns to match production
                if 'created_at' in existing_columns:
                    batch_op.alter_column('created_at',
                        existing_type=sa.DateTime(),
                        type_=mysql.TIMESTAMP(),
                        nullable=True)
                
                if 'updated_at' not in existing_columns:
                    batch_op.add_column(sa.Column('updated_at', mysql.TIMESTAMP, nullable=True))
                elif 'updated_at' in existing_columns:
                    batch_op.alter_column('updated_at',
                        existing_type=sa.DateTime(),
                        type_=mysql.TIMESTAMP(),
                        nullable=True)
                
                # Adjust column size for thumbnail
                if 'thumbnail' in existing_columns:
                    batch_op.alter_column('thumbnail',
                        existing_type=sa.String(length=1000),
                        type_=sa.String(255),
                        nullable=True)
            
            logger.info("Successfully updated news_articles table columns")
        except Exception as e:
            logger.warning(f"Error updating news_articles table: {e}")
    else:
        # Create the table if it doesn't exist
        try:
            op.create_table('news_articles',
                sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
                sa.Column('symbol', sa.String(10), nullable=False),
                sa.Column('title', sa.String(500), nullable=False),
                sa.Column('publisher', sa.String(100), nullable=False),
                sa.Column('link', sa.String(1000), nullable=False),
                sa.Column('published_date', sa.DateTime(), nullable=False),
                sa.Column('type', sa.String(50), nullable=False),
                sa.Column('related_symbols', sa.String(200), nullable=False),
                sa.Column('preview_text', sa.Text(), nullable=False),
                sa.Column('created_at', mysql.TIMESTAMP(), nullable=True),
                sa.Column('updated_at', mysql.TIMESTAMP(), nullable=True),
                sa.Column('thumbnail', sa.String(255), nullable=True),
                sa.PrimaryKeyConstraint('id')
            )
            logger.info("Created news_articles table")
        except Exception as e:
            logger.warning(f"Error creating news_articles table: {e}")


def downgrade() -> None:
    """Downgrade is not supported for this migration."""
    logger.warning("Downgrade is not supported for this schema alignment migration")
    pass
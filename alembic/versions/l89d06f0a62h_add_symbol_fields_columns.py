"""add_symbol_fields_columns

Revision ID: l89d06f0a62h
Revises: k78d95f9b50g
Create Date: 2025-04-18 09:30:45.123456

This migration adds new columns to the symbol_fields table
to support both stocks and ETFs in a unified model.
"""
from typing import Sequence, Union
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import text

# Set up logging
logging.basicConfig()
logger = logging.getLogger("symbol_fields_add_columns")
logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic
revision: str = 'l89d06f0a62h'
down_revision: Union[str, None] = 'k78d95f9b50g'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add new columns to symbol_fields table."""
    logger.info("Starting migration to add new columns to symbol_fields table...")
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if not inspector.has_table('symbol_fields'):
        logger.error("symbol_fields table not found - cannot continue")
        return
    
    # Get existing column names
    columns = inspector.get_columns('symbol_fields')
    existing_columns = [c['name'] for c in columns]
    
    # Add columns that don't exist yet
    with op.batch_alter_table('symbol_fields') as batch_op:
        # Add is_etf column and index
        if 'is_etf' not in existing_columns:
            batch_op.add_column(sa.Column('is_etf', sa.Boolean(), nullable=False, server_default='0'))
            logger.info("Added is_etf column with default value FALSE")
            
            # Add index on is_etf column
            batch_op.create_index('ix_symbol_fields_is_etf', ['is_etf'])
            logger.info("Added index on is_etf column")
        
        # Add index on symbol if it doesn't exist
        indexes = inspector.get_indexes('symbol_fields')
        index_names = [idx['name'] for idx in indexes if idx['name'] is not None]
        
        if 'ix_symbol_fields_symbol' not in index_names:
            batch_op.create_index('ix_symbol_fields_symbol', ['symbol'])
            logger.info("Added index on symbol column")
        
        # Make company_name nullable if it's not already
        if 'company_name' in existing_columns:
            for column in columns:
                if column['name'] == 'company_name' and not column['nullable']:
                    batch_op.alter_column('company_name', nullable=True)
                    logger.info("Modified company_name to be nullable")
                    break
        
        # Add ETF specific columns
        for column_name, column_type in [
            ('relative_volume', sa.Float()),
            ('aum', mysql.DECIMAL(precision=25, scale=4)),
            ('nav_total_return_3y', sa.Float()),
            ('expense_ratio', mysql.DECIMAL(precision=10, scale=4)),
            ('asset_class', sa.String(255)),
            ('focus', sa.String(255))
        ]:
            if column_name not in existing_columns:
                batch_op.add_column(sa.Column(column_name, column_type, nullable=True))
                logger.info(f"Added {column_name} column")
    
    logger.info("Successfully added new columns to symbol_fields table")


def downgrade() -> None:
    """Remove added columns from symbol_fields table."""
    logger.info("Starting downgrade to remove added columns from symbol_fields table...")
    
    # Check if table exists
    if not op.get_bind().dialect.has_table(op.get_bind(), 'symbol_fields'):
        logger.error("symbol_fields table not found - cannot continue")
        return
    
    with op.batch_alter_table('symbol_fields') as batch_op:
        # Remove indexes
        try:
            batch_op.drop_index('ix_symbol_fields_is_etf')
            logger.info("Removed index on is_etf column")
        except Exception as e:
            logger.warning(f"Could not remove index on is_etf column: {e}")
        
        # Remove columns
        for column_name in [
            'is_etf',
            'relative_volume',
            'aum',
            'nav_total_return_3y',
            'expense_ratio',
            'asset_class',
            'focus'
        ]:
            try:
                batch_op.drop_column(column_name)
                logger.info(f"Removed {column_name} column")
            except Exception as e:
                logger.warning(f"Could not remove {column_name} column: {e}")
    
    logger.info("Successfully removed added columns from symbol_fields table")
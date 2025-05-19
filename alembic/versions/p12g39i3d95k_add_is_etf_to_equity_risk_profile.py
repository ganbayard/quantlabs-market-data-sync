"""add_is_etf_to_equity_risk_profile

Revision ID: p12g39i3d95k
Revises: n01f28h2c84j
Create Date: 2025-04-18 17:45:25.987654

This migration adds the is_etf column to the equity_risk_profile table
to distinguish between stocks and ETFs.
"""
from typing import Sequence, Union
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import text

# Set up logging
logging.basicConfig()
logger = logging.getLogger("add_is_etf_to_equity_risk_profile")
logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic
revision: str = 'p12g39i3d95k'
down_revision: Union[str, None] = 'n01f28h2c84j'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add is_etf column to equity_risk_profile table."""
    logger.info("Starting migration to add is_etf column to equity_risk_profile table...")
    
    # Check if table exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if not inspector.has_table('equity_risk_profile'):
        logger.error("Table equity_risk_profile does not exist - cannot continue")
        return
    
    # Check if column already exists
    columns = inspector.get_columns('equity_risk_profile')
    column_names = [c['name'] for c in columns]
    
    if 'is_etf' in column_names:
        logger.warning("Column is_etf already exists in equity_risk_profile table, skipping addition")
        return
    
    # Add is_etf column
    try:
        with op.batch_alter_table('equity_risk_profile') as batch_op:
            batch_op.add_column(sa.Column('is_etf', sa.Boolean(), nullable=False, server_default='0'))
        logger.info("Added is_etf column to equity_risk_profile table")
        
        # Add index on is_etf column
        op.create_index('idx_equity_risk_profile_is_etf', 'equity_risk_profile', ['is_etf'])
        logger.info("Added index on is_etf column")
        
        # Try to set is_etf=True for symbols that are ETFs based on symbol_fields table
        # This is an optional step that will only work if the symbol_fields table exists
        # and has is_etf column
        if inspector.has_table('symbol_fields'):
            symbol_fields_columns = inspector.get_columns('symbol_fields')
            symbol_fields_column_names = [c['name'] for c in symbol_fields_columns]
            
            if 'is_etf' in symbol_fields_column_names:
                try:
                    # Update is_etf based on symbol_fields
                    op.execute(text("""
                        UPDATE equity_risk_profile erp
                        JOIN symbol_fields sf ON erp.symbol = sf.symbol
                        SET erp.is_etf = sf.is_etf
                        WHERE sf.is_etf = 1
                    """))
                    logger.info("Updated is_etf values based on symbol_fields table")
                except Exception as e:
                    logger.warning(f"Error updating is_etf values from symbol_fields: {e}")
                    # This is non-critical, so we continue
        
    except Exception as e:
        logger.error(f"Error adding is_etf column to equity_risk_profile table: {e}")
        raise
    
    logger.info("Successfully added is_etf column to equity_risk_profile table")


def downgrade() -> None:
    """Remove is_etf column from equity_risk_profile table."""
    logger.info("Starting downgrade to remove is_etf column from equity_risk_profile table...")
    
    # Check if table exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if not inspector.has_table('equity_risk_profile'):
        logger.warning("Table equity_risk_profile does not exist, skipping downgrade")
        return
    
    # Check if column exists
    columns = inspector.get_columns('equity_risk_profile')
    column_names = [c['name'] for c in columns]
    
    if 'is_etf' not in column_names:
        logger.warning("Column is_etf does not exist in equity_risk_profile table, skipping removal")
        return
    
    # Remove index first
    indexes = inspector.get_indexes('equity_risk_profile')
    index_names = [idx['name'] for idx in indexes if idx['name'] is not None]
    
    if 'idx_equity_risk_profile_is_etf' in index_names:
        try:
            op.drop_index('idx_equity_risk_profile_is_etf', table_name='equity_risk_profile')
            logger.info("Dropped index idx_equity_risk_profile_is_etf")
        except Exception as e:
            logger.warning(f"Error dropping index idx_equity_risk_profile_is_etf: {e}")
    
    # Remove column
    try:
        with op.batch_alter_table('equity_risk_profile') as batch_op:
            batch_op.drop_column('is_etf')
        logger.info("Removed is_etf column from equity_risk_profile table")
    except Exception as e:
        logger.error(f"Error removing is_etf column from equity_risk_profile table: {e}")
        raise
    
    logger.info("Successfully removed is_etf column from equity_risk_profile table")

    
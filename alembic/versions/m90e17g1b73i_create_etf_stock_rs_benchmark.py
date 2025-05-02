"""create_etf_stock_rs_benchmark

Revision ID: m90e17g1b73i
Revises: l89d06f0a62h
Create Date: 2025-04-18 14:45:30.987654

This migration creates the etf_stock_rs_benchmark table to store
relative strength metrics between stocks and ETFs.
"""
from typing import Sequence, Union
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import text

# Set up logging
logging.basicConfig()
logger = logging.getLogger("create_etf_stock_rs_benchmark")
logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic
revision: str = 'm90e17g1b73i'
down_revision: Union[str, None] = 'l89d06f0a62h'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create etf_stock_rs_benchmark table."""
    logger.info("Starting creation of etf_stock_rs_benchmark table...")
    
    # Check if table already exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if inspector.has_table('etf_stock_rs_benchmark'):
        logger.warning("Table etf_stock_rs_benchmark already exists, skipping creation")
        return
    
    # Create the table
    try:
        op.create_table('etf_stock_rs_benchmark',
            sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column('stock_symbol', sa.String(255), nullable=False),
            sa.Column('etf_symbol', sa.String(255), nullable=False),
            sa.Column('ishare_etf_id', sa.BigInteger(), nullable=True),
            sa.Column('market_value', mysql.DECIMAL(precision=20, scale=2), nullable=True),
            sa.Column('weight', mysql.DECIMAL(precision=10, scale=4), nullable=True),
            
            # Relative strength metrics
            sa.Column('rs_current', mysql.DOUBLE(), nullable=True),
            sa.Column('rs_mean_100d', mysql.DOUBLE(), nullable=True),
            sa.Column('rs_mean_50d', mysql.DOUBLE(), nullable=True),
            sa.Column('rs_mean_20d', mysql.DOUBLE(), nullable=True),
            sa.Column('rs_mean_5d', mysql.DOUBLE(), nullable=True),
            
            # Percentage of positive RS values
            sa.Column('rs_pos_pct_100d', mysql.DOUBLE(), nullable=True),
            sa.Column('rs_pos_pct_50d', mysql.DOUBLE(), nullable=True),
            sa.Column('rs_pos_pct_20d', mysql.DOUBLE(), nullable=True),
            sa.Column('rs_pos_pct_5d', mysql.DOUBLE(), nullable=True),
            
            # Lists stored as JSON arrays in TEXT fields
            sa.Column('rs_values_100d', sa.Text(), nullable=True),
            sa.Column('rs_values_50d', sa.Text(), nullable=True),
            sa.Column('rs_values_20d', sa.Text(), nullable=True),
            sa.Column('rs_values_5d', sa.Text(), nullable=True),
            
            sa.Column('benchmark_symbol', sa.String(10), nullable=False, server_default='SPY'),
            sa.Column('created_at', mysql.TIMESTAMP(), server_default=sa.func.now(), nullable=True),
            sa.Column('updated_at', mysql.TIMESTAMP(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
            
            sa.PrimaryKeyConstraint('id'),
            sa.ForeignKeyConstraint(['ishare_etf_id'], ['ishare_etf.id'], name='fk_etf_stock_rs_benchmark_ishare_etf', ondelete='SET NULL')
        )
        logger.info("Created etf_stock_rs_benchmark table")
        
        # Create indexes
        op.create_index('idx_stock_symbol', 'etf_stock_rs_benchmark', ['stock_symbol'])
        op.create_index('idx_etf_symbol', 'etf_stock_rs_benchmark', ['etf_symbol'])
        op.create_index('idx_rs_mean_100d', 'etf_stock_rs_benchmark', ['rs_mean_100d'])
        logger.info("Created indexes on etf_stock_rs_benchmark table")
        
    except Exception as e:
        logger.error(f"Error creating etf_stock_rs_benchmark table: {e}")
        raise
    
    logger.info("Successfully created etf_stock_rs_benchmark table")


def downgrade() -> None:
    """Drop etf_stock_rs_benchmark table."""
    logger.info("Starting downgrade to drop etf_stock_rs_benchmark table...")
    
    # Check if table exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if not inspector.has_table('etf_stock_rs_benchmark'):
        logger.warning("Table etf_stock_rs_benchmark does not exist, skipping drop")
        return
    
    # Drop the table
    try:
        op.drop_table('etf_stock_rs_benchmark')
        logger.info("Dropped etf_stock_rs_benchmark table")
    except Exception as e:
        logger.error(f"Error dropping etf_stock_rs_benchmark table: {e}")
        raise
    
    logger.info("Successfully dropped etf_stock_rs_benchmark table")
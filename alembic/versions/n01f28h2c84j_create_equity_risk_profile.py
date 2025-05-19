"""create_equity_risk_profile

Revision ID: n01f28h2c84j
Revises: m90e17g1b73i
Create Date: 2025-04-18 16:20:15.654321

This migration creates the equity_risk_profile table to store
volatility, beta, drawdown, and ADR metrics for equities.
"""
from typing import Sequence, Union
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import text

# Set up logging
logging.basicConfig()
logger = logging.getLogger("create_equity_risk_profile")
logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic
revision: str = 'n01f28h2c84j'
down_revision: Union[str, None] = 'm90e17g1b73i'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create equity_risk_profile table."""
    logger.info("Starting creation of equity_risk_profile table...")
    
    # Check if table already exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if inspector.has_table('equity_risk_profile'):
        logger.warning("Table equity_risk_profile already exists, skipping creation")
        return
    
    # Create the table
    try:
        op.create_table('equity_risk_profile',
            sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column('symbol', sa.String(255), nullable=False),
            sa.Column('price', mysql.DECIMAL(precision=15, scale=4), nullable=True),
            sa.Column('risk_type', sa.String(50), nullable=False),
            sa.Column('average_score', mysql.DOUBLE(), nullable=True),
            
            # 3-year averages
            sa.Column('volatility_3yr_avg', mysql.DOUBLE(), nullable=True),
            sa.Column('beta_3yr_avg', mysql.DOUBLE(), nullable=True),
            sa.Column('max_drawdown_3yr_avg', mysql.DOUBLE(), nullable=True),
            sa.Column('adr_3yr_avg', mysql.DOUBLE(), nullable=True),
            
            # Year by year metrics - Year3 is most recent
            sa.Column('volatility_year3', mysql.DOUBLE(), nullable=True),
            sa.Column('volatility_year2', mysql.DOUBLE(), nullable=True),
            sa.Column('volatility_year1', mysql.DOUBLE(), nullable=True),
            
            sa.Column('beta_year3', mysql.DOUBLE(), nullable=True),
            sa.Column('beta_year2', mysql.DOUBLE(), nullable=True),
            sa.Column('beta_year1', mysql.DOUBLE(), nullable=True),
            
            sa.Column('max_drawdown_year3', mysql.DOUBLE(), nullable=True),
            sa.Column('max_drawdown_year2', mysql.DOUBLE(), nullable=True),
            sa.Column('max_drawdown_year1', mysql.DOUBLE(), nullable=True),
            
            sa.Column('adr_year3', mysql.DOUBLE(), nullable=True),
            sa.Column('adr_year2', mysql.DOUBLE(), nullable=True),
            sa.Column('adr_year1', mysql.DOUBLE(), nullable=True),
            
            # Individual metric scores
            sa.Column('volatility_score', mysql.DOUBLE(), nullable=True),
            sa.Column('beta_score', mysql.DOUBLE(), nullable=True),
            sa.Column('drawdown_score', mysql.DOUBLE(), nullable=True),
            sa.Column('adr_score', mysql.DOUBLE(), nullable=True),
            
            # Timestamps
            sa.Column('classified_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', mysql.TIMESTAMP(), server_default=sa.func.now(), nullable=True),
            sa.Column('updated_at', mysql.TIMESTAMP(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
            
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('symbol', 'classified_at', name='uix_equity_risk_profile_symbol_date')
        )
        logger.info("Created equity_risk_profile table")
        
        # Create indexes
        op.create_index('idx_equity_risk_profile_symbol', 'equity_risk_profile', ['symbol'])
        op.create_index('idx_equity_risk_profile_risk_type', 'equity_risk_profile', ['risk_type'])
        logger.info("Created indexes on equity_risk_profile table")
        
    except Exception as e:
        logger.error(f"Error creating equity_risk_profile table: {e}")
        raise
    
    logger.info("Successfully created equity_risk_profile table")


def downgrade() -> None:
    """Drop equity_risk_profile table."""
    logger.info("Starting downgrade to drop equity_risk_profile table...")
    
    # Check if table exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if not inspector.has_table('equity_risk_profile'):
        logger.warning("Table equity_risk_profile does not exist, skipping drop")
        return
    
    # Drop the table
    try:
        op.drop_table('equity_risk_profile')
        logger.info("Dropped equity_risk_profile table")
    except Exception as e:
        logger.error(f"Error dropping equity_risk_profile table: {e}")
        raise
    
    logger.info("Successfully dropped equity_risk_profile table")
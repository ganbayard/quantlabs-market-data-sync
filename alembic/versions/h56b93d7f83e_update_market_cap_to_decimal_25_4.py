"""update_market_cap_to_decimal_25_4

Revision ID: h56b93d7f83e
Revises: g45b92c5e72f
Create Date: 2025-04-17 11:30:45.123456

This migration updates the market_cap column in symbol_fields table to DECIMAL(25,4)
to match the production schema.
"""
from typing import Sequence, Union
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import text

# Set up logging
logging.basicConfig()
logger = logging.getLogger("market_cap_update")
logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic
revision: str = 'h56b93d7f83e'
down_revision: Union[str, None] = 'g45b92c5e72f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Update market_cap column to DECIMAL(25,4)."""
    logger.info("Starting market_cap column update to DECIMAL(25,4)...")
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if not inspector.has_table('symbol_fields'):
        logger.error("symbol_fields table not found - cannot continue")
        return
    
    # Get existing column details
    columns = inspector.get_columns('symbol_fields')
    column_dict = {c['name']: c for c in columns}
    
    try:
        # Update market_cap to correct precision and scale
        if 'market_cap' in column_dict:
            # Since we previously scaled down large values, we need to restore them
            # But first let's make sure the column can hold the necessary precision
            with op.batch_alter_table('symbol_fields') as batch_op:
                batch_op.alter_column('market_cap',
                    existing_type=mysql.DECIMAL(precision=10, scale=2),
                    type_=mysql.DECIMAL(precision=25, scale=4),
                    nullable=True)
                logger.info("Updated market_cap column to DECIMAL(25,4)")
            
            # Restore values that were previously scaled down (if needed)
            # This is only necessary if we want to reverse the division by 1,000,000 that was done in the previous migration
            # Uncomment if you need to reverse that scaling
            """
            op.execute(text('''
                UPDATE symbol_fields 
                SET market_cap = market_cap * 1000000
                WHERE market_cap < 100000 AND market_cap > 0
            '''))
            logger.info("Restored original scale for market_cap values")
            """
            
        logger.info("Successfully updated market_cap column to DECIMAL(25,4)")
    except Exception as e:
        logger.error(f"Error updating market_cap column: {e}")
        raise


def downgrade() -> None:
    """Downgrade is not supported for this migration."""
    logger.warning("Downgrade is not supported for this schema alignment migration")
    pass
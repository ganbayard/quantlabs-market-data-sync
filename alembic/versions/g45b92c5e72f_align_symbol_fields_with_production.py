"""align_symbol_fields_with_production

Revision ID: g45b92c5e72f
Revises: f37b92c4d31e
Create Date: 2025-04-17 10:15:22.123456

This migration aligns the symbol_fields table with the production schema,
focusing on data types, column sizes, and constraints.
"""
from typing import Sequence, Union
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import text

# Set up logging
logging.basicConfig()
logger = logging.getLogger("symbol_fields_alignment")
logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic
revision: str = 'g45b92c5e72f'
down_revision: Union[str, None] = 'f37b92c4d31e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Align symbol_fields table with production schema."""
    logger.info("Starting symbol_fields table alignment with production schema...")
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if not inspector.has_table('symbol_fields'):
        logger.error("symbol_fields table not found - cannot continue")
        return
    
    # Get existing column details
    columns = inspector.get_columns('symbol_fields')
    column_dict = {c['name']: c for c in columns}
    
    try:
        # Pre-process data to handle large values before type conversion
        logger.info("Pre-processing data to handle large values...")
        
        # 1. Handle market_cap - divide by 1000 if out of range and round to 2 decimals
        op.execute(text("""
            UPDATE symbol_fields
            SET market_cap = ROUND(market_cap / 1000000, 2)
            WHERE market_cap > 99999999.99
        """))
        logger.info("Adjusted large market_cap values by dividing by 1000 and rounding to 2 decimals")
        
        # 2. Handle price/change columns that might exceed decimal(15,4) limits
        op.execute(text("""
            UPDATE symbol_fields
            SET price = ROUND(price, 2)
            WHERE price > 99999999999.9999 OR price < -99999999999.9999
        """))
        
        op.execute(text("""
            UPDATE symbol_fields
            SET `change` = ROUND(`change`, 2)
            WHERE `change` > 99999999999.9999 OR `change` < -99999999999.9999
        """))
        logger.info("Rounded large price and change values")
        
        # Now apply the schema changes
        with op.batch_alter_table('symbol_fields') as batch_op:
            # 1. Update price column to decimal(15,4)
            if 'price' in column_dict:
                batch_op.alter_column('price',
                    existing_type=sa.Float(),
                    type_=mysql.DECIMAL(precision=15, scale=4),
                    nullable=True)
                logger.info("Updated price column to decimal(15,4)")
            
            # 2. Update change column to decimal(15,4)
            if 'change' in column_dict:
                batch_op.alter_column('change',
                    existing_type=sa.Float(),
                    type_=mysql.DECIMAL(precision=15, scale=4),
                    nullable=True)
                logger.info("Updated change column to decimal(15,4)")
            
            # 3. Update volume column to bigint
            if 'volume' in column_dict:
                # First handle float values that might not convert to bigint
                op.execute(text("""
                    UPDATE symbol_fields
                    SET volume = ROUND(volume)
                    WHERE volume IS NOT NULL AND volume <> ROUND(volume)
                """))
                
                batch_op.alter_column('volume',
                    existing_type=sa.Float(),
                    type_=sa.BigInteger(),
                    nullable=True)
                logger.info("Updated volume column to bigint")
            
            # 4. Update market_cap column to decimal(10,2)
            if 'market_cap' in column_dict:
                batch_op.alter_column('market_cap',
                    existing_type=sa.Float(),
                    type_=mysql.DECIMAL(precision=10, scale=2),
                    nullable=True)
                logger.info("Updated market_cap column to decimal(10,2)")
            
            # 5. Update market column length
            if 'market' in column_dict:
                batch_op.alter_column('market',
                    existing_type=sa.String(100),
                    type_=sa.String(255),
                    nullable=True)
                logger.info("Updated market column to varchar(255)")
            
            # 6. Update industry column length
            if 'industry' in column_dict:
                batch_op.alter_column('industry',
                    existing_type=sa.String(100),
                    type_=sa.String(255),
                    nullable=True)
                logger.info("Updated industry column to varchar(255)")
            
            # 7. Update sector column length
            if 'sector' in column_dict:
                batch_op.alter_column('sector',
                    existing_type=sa.String(100),
                    type_=sa.String(255),
                    nullable=True)
                logger.info("Updated sector column to varchar(255)")
            
            # 8. Update country column length
            if 'country' in column_dict:
                batch_op.alter_column('country',
                    existing_type=sa.String(50),
                    type_=sa.String(255),
                    nullable=True)
                logger.info("Updated country column to varchar(255)")
            
            # 9. Update exchange column length
            if 'exchange' in column_dict:
                batch_op.alter_column('exchange',
                    existing_type=sa.String(50),
                    type_=sa.String(255),
                    nullable=True)
                logger.info("Updated exchange column to varchar(255)")
            
            # 10. Update date columns to datetime
            for date_col in ['earnings_release_trading_date_fq', 'earnings_release_next_trading_date_fq']:
                if date_col in column_dict:
                    # First, handle NULL values in datetime conversion
                    op.execute(text(f"""
                        UPDATE symbol_fields
                        SET {date_col} = NULL
                        WHERE {date_col} = ''
                    """))
                    
                    batch_op.alter_column(date_col,
                        existing_type=sa.String(50),
                        type_=sa.DateTime(),
                        nullable=True)
                    logger.info(f"Updated {date_col} column to datetime")
            
            # 11. Updated updated_at to timestamp
            if 'updated_at' in column_dict:
                batch_op.alter_column('updated_at',
                    type_=mysql.TIMESTAMP,
                    nullable=True,
                    server_default=sa.func.now(),
                    server_onupdate=sa.func.now())
                logger.info("Updated updated_at column to timestamp with DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")
        
        logger.info("Successfully aligned symbol_fields table with production schema.")
    except Exception as e:
        logger.error(f"Error updating symbol_fields table: {e}")
        raise


def downgrade() -> None:
    """Downgrade is not supported for this migration."""
    logger.warning("Downgrade is not supported for this schema alignment migration")
    pass
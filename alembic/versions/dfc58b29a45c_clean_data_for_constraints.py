"""clean_data_for_constraints

Revision ID: dfc58b29a45c
Revises: ffdc42a01bbb
Create Date: 2025-04-15 19:45:22.456789

"""
from typing import Sequence, Union
import logging
from datetime import datetime

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError

# Set up logging
logging.basicConfig()
logger = logging.getLogger("data_cleaning")
logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic.
revision: str = 'dfc58b29a45c'
down_revision: Union[str, None] = 'ffdc42a01bbb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Clean data to allow constraints to be added."""
    logger.info("Starting data cleaning migration...")
    conn = op.get_bind()
    
    # 1. Fix out of range values in yf_daily_bar
    logger.info("Fixing out of range values in yf_daily_bar...")
    fix_out_of_range_values(conn)
    
    # 2. Fix foreign key violations between yf_daily_bar and symbol_fields
    logger.info("Fixing foreign key violations between yf_daily_bar and symbol_fields...")
    fix_foreign_key_violations(conn)
    
    # 3. Re-apply the constraints that failed in the previous migration
    logger.info("Re-applying constraints...")
    apply_constraints(conn)
    
    logger.info("Data cleaning migration completed successfully.")


def fix_out_of_range_values(conn):
    """Find and fix price values that are too large for NUMERIC(10,4)."""
    try:
        # The max value for NUMERIC(10,4) is 999,999.9999
        max_allowed = 999999.9999
        
        # First, identify the problematic rows 
        result = conn.execute(text("""
            SELECT id, symbol, timestamp, open, high, low, close
            FROM yf_daily_bar
            WHERE open > :max_val OR high > :max_val OR low > :max_val OR close > :max_val
            OR open < -:max_val OR high < -:max_val OR low < -:max_val OR close < -:max_val
            OR open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
        """), {"max_val": max_allowed})
        
        problem_rows = result.fetchall()
        logger.info(f"Found {len(problem_rows)} rows with out-of-range values")
        
        # For each problematic row, cap the values to stay within range
        for row in problem_rows:
            row_id = row[0]
            symbol = row[1]
            timestamp = row[2]
            
            # Set NULL values to 0 and cap values that are too large
            open_val = min(max(row[3] or 0, -max_allowed), max_allowed)
            high_val = min(max(row[4] or 0, -max_allowed), max_allowed)
            low_val = min(max(row[5] or 0, -max_allowed), max_allowed)
            close_val = min(max(row[6] or 0, -max_allowed), max_allowed)
            
            logger.info(f"Fixing values for {symbol} on {timestamp}: {row[3]}->{open_val}, {row[4]}->{high_val}, {row[5]}->{low_val}, {row[6]}->{close_val}")
            
            conn.execute(text("""
                UPDATE yf_daily_bar
                SET open = :open_val, high = :high_val, low = :low_val, close = :close_val
                WHERE id = :row_id
            """), {
                "open_val": open_val,
                "high_val": high_val,
                "low_val": low_val,
                "close_val": close_val,
                "row_id": row_id
            })
        
        if problem_rows:
            logger.info(f"Fixed {len(problem_rows)} rows with out-of-range values")
        
    except SQLAlchemyError as e:
        logger.error(f"Error fixing out-of-range values: {e}")
        raise


def fix_foreign_key_violations(conn):
    """Fix foreign key violations between yf_daily_bar and symbol_fields tables."""
    try:
        # Find symbols in yf_daily_bar that aren't in symbol_fields
        result = conn.execute(text("""
            SELECT DISTINCT yf.symbol 
            FROM yf_daily_bar yf
            LEFT JOIN symbol_fields sf ON yf.symbol = sf.symbol
            WHERE sf.symbol IS NULL
        """))
        
        missing_symbols = result.fetchall()
        missing_count = len(missing_symbols)
        logger.info(f"Found {missing_count} symbols in yf_daily_bar that don't exist in symbol_fields")
        
        if missing_count == 0:
            logger.info("No foreign key violations found")
            return
        
        # Strategy decision: Add missing symbols to symbol_fields with placeholder data
        logger.info("Adding missing symbols to symbol_fields with placeholder data")
        
        for symbol_row in missing_symbols:
            symbol = symbol_row[0]
            current_time = datetime.now()
            
            # Important: Use backticks to escape reserved keywords like 'change'
            conn.execute(text("""
                INSERT INTO symbol_fields 
                (symbol, company_name, price, `change`, volume, market_cap, market, sector, industry, updated_at)
                VALUES 
                (:symbol, :company_name, 0, 0, 0, 0, 'Unknown', 'Unknown', 'Unknown', :updated_at)
            """), {
                "symbol": symbol,
                "company_name": f"Placeholder for {symbol}",
                "updated_at": current_time
            })
            
            logger.info(f"Added placeholder record for symbol: {symbol}")
        
        logger.info(f"Added {missing_count} placeholder records to symbol_fields")
            
    except SQLAlchemyError as e:
        logger.error(f"Error fixing foreign key violations: {e}")
        raise


def apply_constraints(conn):
    """Re-apply the constraints that failed in the previous migration."""
    try:
        # 1. Check for NULL values in columns that need to be NOT NULL
        logger.info("Checking for NULL values before setting NOT NULL constraints...")
        
        result = conn.execute(text("""
            SELECT COUNT(*) FROM yf_daily_bar 
            WHERE symbol IS NULL OR timestamp IS NULL OR 
                  open IS NULL OR high IS NULL OR 
                  low IS NULL OR close IS NULL OR
                  volume IS NULL
        """))
        null_count = result.scalar()
        
        if null_count > 0:
            logger.error(f"Found {null_count} rows with NULL values in yf_daily_bar. Fixing...")
            
            # Fill NULL values with appropriate defaults
            conn.execute(text("""
                UPDATE yf_daily_bar
                SET open = 0 WHERE open IS NULL
            """))
            
            conn.execute(text("""
                UPDATE yf_daily_bar
                SET high = 0 WHERE high IS NULL
            """))
            
            conn.execute(text("""
                UPDATE yf_daily_bar
                SET low = 0 WHERE low IS NULL
            """))
            
            conn.execute(text("""
                UPDATE yf_daily_bar
                SET close = 0 WHERE close IS NULL
            """))
            
            conn.execute(text("""
                UPDATE yf_daily_bar
                SET volume = 0 WHERE volume IS NULL
            """))
            
            logger.info("Fixed NULL values in yf_daily_bar")
        
        # 2. Change column types to DOUBLE instead of NUMERIC to handle larger values
        logger.info("Changing yf_daily_bar price columns to DOUBLE type...")
        
        with op.batch_alter_table('yf_daily_bar') as batch_op:
            # Change price columns to DOUBLE instead of NUMERIC
            batch_op.alter_column('open',
                       existing_type=sa.Numeric(precision=10, scale=4),
                       type_=mysql.DOUBLE(),
                       nullable=False)
            batch_op.alter_column('high',
                       existing_type=sa.Numeric(precision=10, scale=4),
                       type_=mysql.DOUBLE(),
                       nullable=False)
            batch_op.alter_column('low',
                       existing_type=sa.Numeric(precision=10, scale=4),
                       type_=mysql.DOUBLE(),
                       nullable=False)
            batch_op.alter_column('close',
                       existing_type=sa.Numeric(precision=10, scale=4),
                       type_=mysql.DOUBLE(),
                       nullable=False)
            
            # Set NOT NULL constraints on other columns
            batch_op.alter_column('symbol',
                       existing_type=sa.String(length=255),
                       nullable=False)
            batch_op.alter_column('timestamp',
                       existing_type=sa.DateTime(),
                       nullable=False)
            batch_op.alter_column('volume',
                       existing_type=sa.BigInteger(),
                       nullable=False)
        
        logger.info("Successfully applied data type and NOT NULL constraints")
        
        # 3. Add foreign key constraint from yf_daily_bar to symbol_fields
        logger.info("Adding foreign key constraint from yf_daily_bar.symbol to symbol_fields.symbol...")
        
        # First check if it already exists
        inspector = sa.inspect(conn)
        foreign_keys = inspector.get_foreign_keys('yf_daily_bar')
        has_symbol_fk = False
        
        for fk in foreign_keys:
            if fk.get('referred_table') == 'symbol_fields' and 'symbol' in fk.get('constrained_columns', []):
                has_symbol_fk = True
                break
        
        if not has_symbol_fk:
            with op.batch_alter_table('yf_daily_bar') as batch_op:
                batch_op.create_foreign_key(
                    'yf_daily_bar_symbol_foreign',
                    'symbol_fields',
                    ['symbol'],
                    ['symbol'],
                    ondelete='CASCADE'
                )
            logger.info("Foreign key constraint added successfully")
        else:
            logger.info("Foreign key constraint already exists")
            
    except SQLAlchemyError as e:
        logger.error(f"Error applying constraints: {e}")
        raise


def downgrade() -> None:
    """No downgrade is provided as this is a data cleaning migration."""
    pass
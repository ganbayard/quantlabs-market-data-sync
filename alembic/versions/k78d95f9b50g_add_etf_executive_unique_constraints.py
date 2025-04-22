"""add_etf_executive_unique_constraints

Revision ID: k78d95f9b50g
Revises: j67c94e8a94f
Create Date: 2025-04-17 16:35:45.678901

This migration adds unique constraints to the ishare_etf_holding and executives tables.
"""
from typing import Sequence, Union
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import text

# Set up logging
logging.basicConfig()
logger = logging.getLogger("etf_executive_constraints")
logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic
revision: str = 'k78d95f9b50g'
down_revision: Union[str, None] = 'j67c94e8a94f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add unique constraints to ishare_etf_holding and executives tables."""
    logger.info("Starting migration to add constraints to ETF holdings and executives tables...")
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    # Define tables and constraints to add
    tables = {
        'ishare_etf_holding': {
            'unique_constraint': 'uix_ishare_etf_holding_etf_ticker',
            'columns': ['ishare_etf_id', 'ticker'],
            'null_check': True  # Check if ticker can be NULL
        },
        'executives': {
            'unique_constraint': 'uix_executive_company_name',
            'columns': ['company_symbol', 'name'],
            'null_check': False  # Both columns should be NOT NULL
        }
    }
    
    for table_name, config in tables.items():
        if not inspector.has_table(table_name):
            logger.warning(f"Table {table_name} not found, skipping")
            continue
        
        # Check if unique constraint already exists
        unique_constraints = inspector.get_unique_constraints(table_name)
        unique_constraint_names = [uc['name'] for uc in unique_constraints if uc['name'] is not None]
        
        if config['unique_constraint'] in unique_constraint_names:
            logger.info(f"Unique constraint {config['unique_constraint']} already exists on {table_name}")
            continue
        
        # Special handling for ticker field which might be NULL
        if config['null_check'] and table_name == 'ishare_etf_holding':
            # Check if there are NULL values in ticker column that would cause issues
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM ishare_etf_holding
                WHERE ticker IS NULL
            """))
            null_count = result.scalar()
            
            if null_count > 0:
                logger.warning(f"Found {null_count} NULL values in ticker column")
                # Update NULL tickers to a placeholder value
                try:
                    op.execute(text("""
                        UPDATE ishare_etf_holding
                        SET ticker = CONCAT('NO_TICKER_', id)
                        WHERE ticker IS NULL
                    """))
                    logger.info(f"Updated {null_count} NULL ticker values to placeholders")
                except Exception as e:
                    logger.error(f"Error updating NULL ticker values: {e}")
                    raise
        
        # Check for duplicates that would violate the constraint
        columns_str = ', '.join(config['columns'])
        duplicate_check_sql = f"""
            SELECT {columns_str}, COUNT(*) as count
            FROM {table_name}
            WHERE {' AND '.join([f'{col} IS NOT NULL' for col in config['columns']])}
            GROUP BY {columns_str}
            HAVING COUNT(*) > 1
        """
        
        try:
            result = conn.execute(text(duplicate_check_sql))
            duplicates = result.fetchall()
            
            if duplicates:
                logger.error(f"Found duplicates in {table_name} that would violate unique constraint:")
                for row in duplicates:
                    logger.error(f"  {', '.join([f'{config['columns'][i]}={row[i]}' for i in range(len(config['columns']))])}, count={row[-1]}")
                
                # How to handle duplicates - keeping the most recent records
                if table_name == 'ishare_etf_holding':
                    logger.info("Resolving duplicates in ishare_etf_holding by keeping most recent records...")
                    try:
                        # First, add temporary id column to help with duplicate detection
                        op.execute(text(f"""
                            ALTER TABLE {table_name}
                            ADD COLUMN temp_dup_id INT NOT NULL AUTO_INCREMENT,
                            ADD INDEX idx_temp_dup_id (temp_dup_id)
                        """))
                        
                        # Delete older duplicates, keeping the newest by id
                        op.execute(text(f"""
                            DELETE t1 FROM {table_name} t1
                            INNER JOIN {table_name} t2
                            WHERE t1.ishare_etf_id = t2.ishare_etf_id
                            AND t1.ticker = t2.ticker
                            AND t1.id < t2.id
                        """))
                        
                        # Remove temporary column
                        op.execute(text(f"""
                            ALTER TABLE {table_name}
                            DROP COLUMN temp_dup_id
                        """))
                        
                        logger.info("Successfully resolved duplicates in ishare_etf_holding")
                    except Exception as e:
                        logger.error(f"Error resolving duplicates in {table_name}: {e}")
                        # Try to clean up temp column if it exists
                        try:
                            op.execute(text(f"""
                                ALTER TABLE {table_name}
                                DROP COLUMN IF EXISTS temp_dup_id
                            """))
                        except:
                            pass
                        raise
                        
                elif table_name == 'executives':
                    logger.info("Resolving duplicates in executives by keeping most recent records...")
                    try:
                        # For executives, we'll keep the latest record by id
                        op.execute(text(f"""
                            DELETE t1 FROM {table_name} t1
                            INNER JOIN {table_name} t2
                            WHERE t1.company_symbol = t2.company_symbol
                            AND t1.name = t2.name
                            AND t1.id < t2.id
                        """))
                        
                        logger.info("Successfully resolved duplicates in executives")
                    except Exception as e:
                        logger.error(f"Error resolving duplicates in {table_name}: {e}")
                        raise
            
            # Now add the unique constraint
            with op.batch_alter_table(table_name) as batch_op:
                batch_op.create_unique_constraint(
                    config['unique_constraint'],
                    config['columns']
                )
            logger.info(f"Added unique constraint {config['unique_constraint']} to {table_name}")
            
        except Exception as e:
            logger.error(f"Error processing {table_name}: {e}")
            raise
    
    logger.info("Successfully added constraints to ETF holdings and executives tables")


def downgrade() -> None:
    pass
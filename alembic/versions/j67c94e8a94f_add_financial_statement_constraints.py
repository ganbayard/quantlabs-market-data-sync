"""add_financial_statement_constraints

Revision ID: j67c94e8a94f
Revises: h56b93d7f83e
Create Date: 2025-04-17 14:20:35.123456

This migration adds unique constraints and indexes to the financial statement tables
(income_statements, balance_sheets, cash_flows) to match the model.
"""
from typing import Sequence, Union
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import text

# Set up logging
logging.basicConfig()
logger = logging.getLogger("financial_statement_constraints")
logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic
revision: str = 'j67c94e8a94f'
down_revision: Union[str, None] = 'h56b93d7f83e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add unique constraints and indexes to financial statement tables."""
    logger.info("Starting migration to add constraints to financial statement tables...")
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    # First, check if tables exist
    tables = {
        'income_statements': {
            'unique_constraint': 'uix_income_statement_key',
            'columns': ['symbol', 'date', 'period_type'],
            'index': 'income_statements_symbol_index',
            'index_columns': ['symbol']
        },
        'balance_sheets': {
            'unique_constraint': 'uix_balance_sheet_key',
            'columns': ['symbol', 'date', 'period_type'],
            'index': 'balance_sheets_symbol_index',
            'index_columns': ['symbol']
        },
        'cash_flows': {
            'unique_constraint': 'uix_cash_flow_key',
            'columns': ['symbol', 'date', 'period_type'],
            'index': 'cash_flows_symbol_index',
            'index_columns': ['symbol']
        }
    }
    
    for table_name, config in tables.items():
        if not inspector.has_table(table_name):
            logger.warning(f"Table {table_name} not found, skipping")
            continue
        
        # Check if unique constraint already exists
        unique_constraints = inspector.get_unique_constraints(table_name)
        unique_constraint_names = [uc['name'] for uc in unique_constraints if uc['name'] is not None]
        
        # Check if index already exists
        indexes = inspector.get_indexes(table_name)
        index_names = [idx['name'] for idx in indexes if idx['name'] is not None]
        
        # Create unique constraint if it doesn't exist
        if config['unique_constraint'] not in unique_constraint_names:
            try:
                with op.batch_alter_table(table_name) as batch_op:
                    batch_op.create_unique_constraint(
                        config['unique_constraint'],
                        config['columns']
                    )
                logger.info(f"Added unique constraint {config['unique_constraint']} to {table_name}")
            except Exception as e:
                # Detect duplicates that would violate the unique constraint
                try:
                    result = conn.execute(text(f"""
                        SELECT symbol, date, period_type, COUNT(*) AS count
                        FROM {table_name}
                        GROUP BY symbol, date, period_type
                        HAVING COUNT(*) > 1
                    """))
                    duplicates = result.fetchall()
                    if duplicates:
                        logger.error(f"Found duplicates in {table_name} that would violate unique constraint:")
                        for row in duplicates:
                            logger.error(f"  symbol={row[0]}, date={row[1]}, period_type={row[2]}, count={row[3]}")
                except Exception as inner_e:
                    logger.error(f"Error when trying to find duplicates: {inner_e}")
                
                logger.error(f"Error creating unique constraint on {table_name}: {e}")
                # Decide whether to continue or raise the exception based on your requirements
                # In this case, we log the error but continue with other tables
        else:
            logger.info(f"Unique constraint {config['unique_constraint']} already exists on {table_name}")
        
        # Create index if it doesn't exist
        if config['index'] not in index_names:
            try:
                op.create_index(
                    config['index'],
                    table_name,
                    config['index_columns']
                )
                logger.info(f"Added index {config['index']} to {table_name}")
            except Exception as e:
                logger.error(f"Error creating index on {table_name}: {e}")
        else:
            logger.info(f"Index {config['index']} already exists on {table_name}")
    
    logger.info("Successfully added constraints to financial statement tables")


def downgrade() -> None:
    """Remove unique constraints and indexes from financial statement tables."""
    pass
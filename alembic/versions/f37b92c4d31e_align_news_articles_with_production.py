"""align_news_articles_with_production

Revision ID: f37b92c4d31e
Revises: e57b92f4a21d
Create Date: 2025-04-16 16:45:22.123456

This migration aligns the news_articles table with the production schema,
including column types, nullability, and constraints.
"""
from typing import Sequence, Union
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import text

# Set up logging
logging.basicConfig()
logger = logging.getLogger("news_articles_alignment")
logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic
revision: str = 'f37b92c4d31e'
down_revision: Union[str, None] = 'e57b92f4a21d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Precisely align news_articles table with production schema."""
    logger.info("Starting news_articles table alignment with production schema...")
    
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    # First, clean up any leftover temporary tables from previous failed migrations
    if inspector.has_table('news_articles_new'):
        logger.info("Found leftover news_articles_new table, dropping it first...")
        op.drop_table('news_articles_new')
    
    if inspector.has_table('news_articles'):
        logger.info("Altering existing news_articles table...")
        
        # Get existing columns
        columns = inspector.get_columns('news_articles')
        column_dict = {c['name']: c for c in columns}
        
        # 1. Update column types and nullability constraints
        with op.batch_alter_table('news_articles') as batch_op:
            # First, fix any NULL values in columns that will become NOT NULL
            for col, new_type, nullable in [
                ('symbol', sa.String(10), False),
                ('title', sa.String(500), False),
                ('publisher', sa.String(100), False), 
                ('link', sa.String(1000), False),
                ('published_date', sa.DateTime(), False),
                ('type', sa.String(50), False),
                ('related_symbols', sa.String(200), False),
                ('preview_text', sa.Text(), False)
            ]:
                # Update NULL values first if column exists
                if col in column_dict:
                    logger.info(f"Updating any NULL values in {col} column")
                    # Different default values for different columns
                    default_value = {
                        'symbol': 'UNKNOWN',
                        'title': 'Untitled',
                        'publisher': 'Unknown Publisher',
                        'link': 'https://example.com',
                        'type': 'Unknown',
                        'related_symbols': '',
                        'preview_text': ''
                    }.get(col, '')
                    
                    # For published_date, we need to use NOW()
                    if col == 'published_date':
                        op.execute(text(f"""
                            UPDATE news_articles
                            SET {col} = NOW()
                            WHERE {col} IS NULL
                        """))
                    else:
                        op.execute(text(f"""
                            UPDATE news_articles
                            SET {col} = '{default_value}'
                            WHERE {col} IS NULL
                        """))
                    
                    # Now alter the column type and nullability
                    batch_op.alter_column(col, 
                                        existing_type=column_dict[col]['type'],
                                        type_=new_type,
                                        nullable=nullable)
                    logger.info(f"Updated {col} column type and nullability")
                else:
                    # Add missing column
                    batch_op.add_column(sa.Column(col, new_type, nullable=nullable))
                    logger.info(f"Added missing {col} column")
            
            # 2. Handle thumbnail column (changing from varchar(1000) to varchar(255))
            if 'thumbnail' in column_dict:
                # First truncate values that are too long
                op.execute(text("""
                    UPDATE news_articles
                    SET thumbnail = SUBSTRING(thumbnail, 1, 255)
                    WHERE LENGTH(thumbnail) > 255
                """))
                
                batch_op.alter_column('thumbnail',
                                    existing_type=column_dict['thumbnail']['type'],
                                    type_=sa.String(255),
                                    nullable=True)
                logger.info("Updated thumbnail column to varchar(255)")
            else:
                batch_op.add_column(sa.Column('thumbnail', sa.String(255), nullable=True))
                logger.info("Added missing thumbnail column")
            
            # 3. Handle timestamp columns
            if 'created_at' in column_dict:
                batch_op.alter_column('created_at',
                                    existing_type=column_dict['created_at']['type'],
                                    type_=mysql.TIMESTAMP,
                                    nullable=True)
                logger.info("Updated created_at column to TIMESTAMP")
            else:
                batch_op.add_column(sa.Column('created_at', mysql.TIMESTAMP, nullable=True))
                logger.info("Added missing created_at column")
                
            if 'updated_at' not in column_dict:
                batch_op.add_column(sa.Column('updated_at', mysql.TIMESTAMP, nullable=True))
                logger.info("Added missing updated_at column")
            
        # 4. Make sure the correct indexes exist
        # First get existing indexes to avoid duplicates
        indexes = inspector.get_indexes('news_articles')
        index_names = [idx['name'] for idx in indexes if idx['name'] is not None]
        
        if 'ix_news_articles_published_date' not in index_names:
            op.create_index('ix_news_articles_published_date', 'news_articles', ['published_date'])
            logger.info("Added published_date index")
            
        if 'ix_news_articles_symbol' not in index_names:
            op.create_index('ix_news_articles_symbol', 'news_articles', ['symbol'])
            logger.info("Added symbol index")
        
        # Check if unique constraint exists
        unique_constraints = inspector.get_unique_constraints('news_articles')
        unique_constraint_names = [uc['name'] for uc in unique_constraints if uc['name'] is not None]
        
        if 'uix_news_1' not in unique_constraint_names:
            with op.batch_alter_table('news_articles') as batch_op:
                batch_op.create_unique_constraint('uix_news_1', ['symbol', 'title', 'published_date'])
            logger.info("Added unique constraint on symbol, title, published_date")
        
        logger.info("Successfully aligned news_articles table with production schema")
    else:
        # Create the table if it doesn't exist
        logger.info("news_articles table doesn't exist, creating it...")
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
            logger.info("Created news_articles table with production schema")
            
            # Add indexes
            op.create_index('ix_news_articles_published_date', 'news_articles', ['published_date'])
            op.create_index('ix_news_articles_symbol', 'news_articles', ['symbol'])
            op.create_unique_constraint('uix_news_1', 'news_articles', 
                                       ['symbol', 'title', 'published_date'])
            logger.info("Added indexes to news_articles table")
        except Exception as e:
            logger.error(f"Error creating news_articles table: {e}")
            raise
    
    logger.info("Schema alignment completed successfully.")


def downgrade() -> None:
    """Downgrade is not supported for this migration."""
    logger.warning("Downgrade is not supported for this schema alignment migration")
    pass
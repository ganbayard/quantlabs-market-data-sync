import os
import sys
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.database.db_config import get_db_connection
from src.database.models import Base

def update_database_schema(db_type="sqlite"):
    """
    Update the database schema by dropping and recreating all tables.
    This ensures that all tables have the correct structure and constraints.
    """
    print("Starting database schema update...")
    
    # Get database connection
    db_config, engine, session = get_db_connection(db_type)
    
    try:
        # Drop all tables if they exist
        print("Dropping existing tables...")
        Base.metadata.drop_all(engine)
        
        # Create all tables with updated schema
        print("Creating tables with updated schema...")
        Base.metadata.create_all(engine)
        
        print("Database schema update completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error updating database schema: {str(e)}")
        return False
    
    finally:
        # Close database session
        session.close()

if __name__ == "__main__":
    # Get database type from command line argument if provided
    db_type = sys.argv[1] if len(sys.argv) > 1 else "sqlite"
    update_database_schema(db_type)

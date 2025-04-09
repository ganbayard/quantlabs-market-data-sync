import os
import sys
from logging.config import fileConfig
from dotenv import load_dotenv

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Load environment variables from .env file
load_dotenv()

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import Base and all models
try:
    from models.market_data import Base
    print("Successfully imported models")
except ImportError as e:
    print(f"Error importing models: {e}")
    sys.exit(1)

# this is the Alembic Config object
config = context.config

# Determine which environment to use (dev or prod)
migration_env = os.getenv('MIGRATION_ENV', 'dev')

# Configure the connection string dynamically
try:
    if migration_env == 'dev':
        connection_string = f"mysql+pymysql://{os.getenv('LOCAL_DB_USER')}:{os.getenv('LOCAL_DB_PASSWORD')}@{os.getenv('LOCAL_DB_HOST', 'localhost')}:{os.getenv('LOCAL_DB_PORT', '3306')}/{os.getenv('LOCAL_DB_NAME')}"
        print(f"Using DEVELOPMENT database connection")
    else:
        connection_string = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME')}"
        print(f"Using PRODUCTION database connection")
        
    config.set_main_option('sqlalchemy.url', connection_string)
except Exception as e:
    print(f"Error setting database connection: {e}")
    sys.exit(1)

# Interpret the config file for Python logging
fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

# Financial Market Data Sync

This project provides utilities for managing MySQL database connections and operations across different environments (local development and AWS cloud production).

## aws connection 

#### Private Key Permissions on Windows
icacls "C:....\quantlab-web.pem" /inheritance:r
icacls "C:....\quantlab-web.pem" /grant:r "%USERNAME%:R"

#### Git Bash & mac linux dist
chmod 600 .../quantlab-web.pem

### ssh cli

ssh -o "ServerAliveInterval 60" -i "aws-key/quantlab-web.pem" ubuntu@ec2-3-72-81-201.eu-central-1.compute.amazonaws.com

## Project Structure
```
financial-market-data-sync/
├── config/
│   ├── __init__.py
│   ├── local_config.py
│   └── production_config.py
├── database/
│   ├── __init__.py
│   └── connection.py
├── scripts/
│   ├── __init__.py
│   └── db_operations.py
├── init/
│   └── 01_create_tables.sql
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

## Local Development Setup

### Option 1: Using Docker (Recommended)

1. Start the MySQL database:
```bash
docker-compose up -d
```

2. Verify the database is running:
```bash
docker-compose ps
```

3. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

The `.env` file is already configured for the Docker setup.

### Option 2: Manual MySQL Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and update with your MySQL credentials:
```bash
cp .env.example .env
```


## Running Database Operations

To run database operations:

```bash
# For local development
python scripts/db_operations.py --env local

# For AWS production
python scripts/db_operations.py --env production
```

## Docker Commands

- Start the database: `docker-compose up -d`
- Stop the database: `docker-compose down`
- View logs: `docker-compose logs -f`
- Restart the database: `docker-compose restart`
- Remove all data: `docker-compose down -v`

## Database Migrations

The project uses Alembic for database migrations to keep the database schema in sync with our models. We provide a parametric migration script (`run_migrations.py`) to easily run migrations against different environments.

### Prerequisites

- Ensure Docker is running if targeting the local development database
- Make sure your `.env` file contains all required database credentials
- Install required packages: `pip install alembic pymysql`

### Using the Migration Script

The `run_migrations.py` script provides a convenient way to run Alembic operations against either development or production databases.

#### Basic Syntax

```bash
python run_migrations.py --env [dev|prod] [command] [options]
```

#### Available Commands

- `init` - Initialize Alembic (only needed once)
- `revision` - Create a new migration revision
- `upgrade` - Apply migrations
- `downgrade` - Revert migrations
- `migrate` - Shorthand for creating and applying migrations

#### Common Usage Examples

**Initialize Alembic (first-time setup)**
```bash
python run_migrations.py --env dev init
```

**Create a new migration example**
```bash
python run_migrations.py --env dev revision --autogenerate -m "Create all financial market tables"
python run_migrations.py --env dev revision -m "Add new table"
```

**Apply all pending migrations to development database**
```bash
python run_migrations.py --env dev upgrade
```

**Apply all pending migrations to production database**
```bash
python run_migrations.py --env prod upgrade
```

**Downgrade to a specific revision**
```bash
python run_migrations.py --env dev downgrade revision_id
```

### Environment Configuration

The script uses different database connections based on the `--env` parameter:

- **dev**: Uses local Docker database specified by `LOCAL_DB_*` variables in `.env`
- **prod**: Uses production database specified by `DB_*` variables in `.env`

### Safety Considerations

- Always run migrations against development first before applying to production
- Consider backing up your production database before running migrations
- Review generated migration files before applying them to production

## Security

- Never commit `.env` files or sensitive credentials
- Use appropriate IAM roles and security groups in AWS
- Follow the principle of least privilege for database access
- Change default passwords in production

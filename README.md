# Quantlab Market Data Sync

This project provides utilities for managing MySQL database connections and operations across different environments (local development and AWS cloud production).


## aws connection 

#### Private Key Permissions on Windows
icacls "C:\src\dev\gb\quantlabs-market-data-sync\aws-key\quantlab-web.pem" /inheritance:r
icacls "C:\src\dev\gb\quantlabs-market-data-sync\aws-key\quantlab-web.pem" /grant:r "%USERNAME%:R"

#### Git Bash & mac linux dist
chmod 600 .../quantlab-web.pem

### ssh cli

ssh -o "ServerAliveInterval 60" -i "aws-key/quantlab-web.pem" ubuntu@ec2-3-72-81-201.eu-central-1.compute.amazonaws.com

## Quick Installation and Run
```sh
# Clone the repository
git clone https://github.com/ganbayard/quantlabs-market-data-sync
cd quantlabs-market-data-sync

# Set up environment
cp .env.example .env
# Edit .env file with prod credentials

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Update Database Model Migrations for Table Columns and Data Types
python run_migration.py

# Run all Scripts as sequential

## On AWS Instance for Production Mysql
sh run_scripts_sequence.sh prod

## Local mysql test
sh run_scripts_sequence.sh


```


## Project Structure
```
quantlabs-market-data-sync/
├── alembic/                # Database migration files
│   ├── versions/           # Migration version scripts
│   ├── env.py              # Alembic environment configuration
├── database/               # Database connection utilities
│   ├── connection.py       # Database connection management
│   ├── test_mysql_tunnel.py # SSH tunnel for remote database access
├── docker-entrypoint-initdb.d/ # Docker MySQL initialization
├── logs/                   # Log output directory
├── models/                 # SQLAlchemy database models
│   ├── market_data.py      # Financial data models (YfBar1d, EquityTechnicalIndicator, etc.)
├── scripts/                # Data collection and processing scripts
│   ├── common_function.py  # Shared utilities for all scripts
│   ├── equity2user/        # Stock classification and recommendation scripts
│   │   ├── equity_technical_indicators.py  # Technical indicator calculations
│   │   ├── equity2user_history.py          # Stock classification and user recommendations
│   ├── etf/                # ETF data collection scripts
│   │   ├── holdings_ishare_etf_list.py     # iShares ETF list retrieval
│   │   ├── holdings_ishare_etf_update.py   # ETF holdings updates
│   ├── financial_details/  # Financial statement and news scripts
│   │   ├── company_financials_yfinance.py  # Income, balance sheet and cash flow data
│   │   ├── news_collector.py               # Financial news collection
│   ├── general_info/       # Basic market data scripts
│   │   ├── company_profile_yfinance.py     # Company profile information
│   │   ├── symbol_fields_update.py         # Basic symbol data updates
│   │   ├── yf_daily_bar_loader.py          # Price history collection
│   ├── sector_rotation/    # Sector analysis scripts
│   ├── symbols/            # Symbol lists for processing
│   │   ├── stock_symbols.txt               # Main list of stock symbols
├── run_migrations.py       # Database migration script
├── run_scripts_sequence.sh # Sequential script execution
├── docker-compose.yml      # Docker configuration for local development
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Local Development Setup

### Mysql Database Docker Deployment

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

## Docker Commands

- Start the database: `docker-compose up -d`
- Stop the database: `docker-compose down`
- View logs: `docker-compose logs -f`

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

## Scripts commands  

## yf daily bar

### Force specific time period but still use smart update
python scripts/general_info/yf_daily_bar_loader.py --period last_week --workers 1

### Disable smart update and use fixed periods
python scripts/general_info/yf_daily_bar_loader.py --period last_day --no-smart-update --workers 1



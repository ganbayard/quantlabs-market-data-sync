import os
import sys
import argparse
import yfinance as yf
import time
import logging
import math
import numpy as np
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import IncomeStatement, BalanceSheet, CashFlow
from scripts.common_function import get_database_engine, add_environment_args, get_session_maker

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"company_financial_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
NUM_WORKERS = 6
INPUT_FILE = script_dir / "../symbols/stock_symbols.txt"

# Thread-local storage for database sessions
thread_local = threading.local()

class Stats:
    def __init__(self, total_symbols):
        self.processed = 0
        self.successful = 0
        self.not_found = 0
        self.other_errors = 0
        self.total_symbols = total_symbols
        self.lock = threading.Lock()

    def increment(self, attribute):
        with self.lock:
            setattr(self, attribute, getattr(self, attribute) + 1)
            if attribute == "processed" and self.processed % 5 == 0:
                logger.info(
                    f"Progress: {self.processed}/{self.total_symbols} ({(self.processed / self.total_symbols) * 100:.1f}%)"
                )
                logger.info(
                    f"Success: {self.successful}, Not Found: {self.not_found}, Other Errors: {self.other_errors}"
                )


def get_thread_session():
    """Get or create a thread-local database session"""
    if not hasattr(thread_local, "session"):
        thread_local.session = Session()
    return thread_local.session


def safe_float(value):
    """Convert value to float safely, handling NaN values"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def get_financial_data(symbol):
    """Get financial data for a symbol and store in database"""
    session = get_thread_session()
    try:
        ticker = yf.Ticker(symbol)

        # Get all financial statements
        financials = {
            "annual": {
                "income": ticker.financials,
                "balance": ticker.balance_sheet,
                "cash": ticker.cashflow,
            },
            "quarterly": {
                "income": ticker.quarterly_financials,
                "balance": ticker.quarterly_balance_sheet,
                "cash": ticker.quarterly_cashflow,
            },
        }

        for period_type, statements in financials.items():
            # Process Income Statement
            for date in statements["income"].columns:
                data = statements["income"][date]

                # Check if record exists
                existing = (
                    session.query(IncomeStatement)
                    .filter_by(
                        symbol=symbol, date=date.date(), period_type=period_type.capitalize()
                    )
                    .first()
                )

                if existing:
                    income_stmt = existing
                else:
                    income_stmt = IncomeStatement(
                        symbol=symbol, date=date.date(), period_type=period_type.capitalize()
                    )

                # Update fields with safe conversions
                income_stmt.revenue = safe_float(data.get("Total Revenue"))
                income_stmt.total_revenue = safe_float(data.get("Total Revenue"))
                income_stmt.cost_of_revenue = safe_float(data.get("Cost of Revenue"))
                income_stmt.gross_profit = safe_float(data.get("Gross Profit"))
                income_stmt.operating_expense = safe_float(data.get("Operating Expense"))
                income_stmt.operating_income = safe_float(data.get("Operating Income"))
                income_stmt.net_non_operating_interest = safe_float(data.get("Net Non Operating Interest Income Expense"))
                income_stmt.other_income_expense = safe_float(data.get("Other Income Expense"))
                income_stmt.pretax_income = safe_float(data.get("Pretax Income"))
                income_stmt.tax_provision = safe_float(data.get("Tax Provision"))
                income_stmt.net_income = safe_float(data.get("Net Income Common Stockholders"))
                income_stmt.basic_shares = safe_float(data.get("Basic Average Shares"))
                income_stmt.diluted_shares = safe_float(data.get("Diluted Average Shares"))
                income_stmt.basic_eps = safe_float(data.get("Basic EPS"))
                income_stmt.diluted_eps = safe_float(data.get("Diluted EPS"))
                
                # Get EPS from data or calculate it if possible
                eps_value = safe_float(data.get("Diluted EPS") or data.get("Basic EPS"))

                # Try to calculate EPS if it's not provided but we have net income and shares
                if eps_value is None and income_stmt.net_income is not None:
                    shares = safe_float(data.get("Diluted Average Shares") or data.get("Basic Average Shares"))
                    if shares and shares > 0:
                        eps_value = income_stmt.net_income / shares

                # Use NULL if we don't have a value (no more 0.0 default)
                income_stmt.eps = eps_value
                
                income_stmt.ebit = safe_float(data.get("EBIT"))
                income_stmt.ebitda = safe_float(data.get("EBITDA"))
                income_stmt.total_operating_income = safe_float(data.get("Total Operating Income"))
                income_stmt.total_expenses = safe_float(data.get("Total Expenses"))
                income_stmt.normalized_income = safe_float(data.get("Normalized Income"))
                income_stmt.interest_income = safe_float(data.get("Interest Income"))
                income_stmt.interest_expense = safe_float(data.get("Interest Expense"))
                income_stmt.net_interest_income = safe_float(data.get("Net Interest Income"))

                session.add(income_stmt)

            # Process Balance Sheet
            for date in statements["balance"].columns:
                data = statements["balance"][date]

                existing = (
                    session.query(BalanceSheet)
                    .filter_by(
                        symbol=symbol, date=date.date(), period_type=period_type.capitalize()
                    )
                    .first()
                )

                if existing:
                    balance = existing
                else:
                    balance = BalanceSheet(
                        symbol=symbol, date=date.date(), period_type=period_type.capitalize()
                    )

                balance.total_assets = safe_float(data.get("Total Assets"))
                balance.total_liabilities = safe_float(data.get("Total Liabilities Net Minority Interest"))
                balance.total_equity = safe_float(data.get("Total Equity Gross Minority Interest"))
                balance.equity = safe_float(data.get("Stockholders Equity") or data.get("Total Equity Gross Minority Interest"))
                balance.total_capitalization = safe_float(data.get("Total Capitalization"))
                balance.common_stock_equity = safe_float(data.get("Common Stock Equity"))
                balance.capital_lease_obligations = safe_float(data.get("Capital Lease Obligations"))
                balance.net_tangible_assets = safe_float(data.get("Net Tangible Assets"))
                balance.working_capital = safe_float(data.get("Working Capital"))
                balance.invested_capital = safe_float(data.get("Invested Capital"))
                balance.tangible_book_value = safe_float(data.get("Tangible Book Value"))
                balance.total_debt = safe_float(data.get("Total Debt"))
                balance.shares_issued = safe_float(data.get("Share Issued"))
                balance.ordinary_shares_number = safe_float(data.get("Ordinary Shares Number"))

                session.add(balance)

            # Process Cash Flow
            for date in statements["cash"].columns:
                data = statements["cash"][date]

                existing = (
                    session.query(CashFlow)
                    .filter_by(
                        symbol=symbol, date=date.date(), period_type=period_type.capitalize()
                    )
                    .first()
                )

                if existing:
                    cash_flow = existing
                else:
                    cash_flow = CashFlow(
                        symbol=symbol, date=date.date(), period_type=period_type.capitalize()
                    )

                cash_flow.operating_cash_flow = safe_float(data.get("Operating Cash Flow"))
                cash_flow.investing_cash_flow = safe_float(data.get("Investing Cash Flow"))
                cash_flow.financing_cash_flow = safe_float(data.get("Financing Cash Flow"))
                cash_flow.end_cash_position = safe_float(data.get("End Cash Position"))
                cash_flow.income_tax_paid = safe_float(data.get("Income Tax Paid Supplemental Data"))
                cash_flow.interest_paid = safe_float(data.get("Interest Paid Supplemental Data"))
                cash_flow.capital_expenditure = safe_float(data.get("Capital Expenditure"))
                cash_flow.issuance_of_capital_stock = safe_float(data.get("Issuance of Capital Stock"))
                cash_flow.issuance_of_debt = safe_float(data.get("Issuance of Debt"))
                cash_flow.repayment_of_debt = safe_float(data.get("Repayment of Debt"))
                cash_flow.free_cash_flow = safe_float(data.get("Free Cash Flow"))

                session.add(cash_flow)

        try:
            session.commit()
            stats.increment("successful")
            return True
        except Exception as e:
            logger.error(f"Error committing {symbol}: {str(e)}")
            session.rollback()
            stats.increment("other_errors")
            return False

    except Exception as e:
        if "404" in str(e):
            logger.warning(f"Skipping {symbol}: Symbol not found (404)")
            stats.increment("not_found")
        else:
            logger.error(f"Error processing {symbol}: {str(e)}")
            stats.increment("other_errors")
        return False
    finally:
        stats.increment("processed")


def process_symbol_batch(symbols):
    """Process a batch of symbols"""
    for symbol in symbols:
        get_financial_data(symbol)
        # Sleep to avoid rate limiting
        time.sleep(1)


def print_statement_stats(session, statement_class, statement_name):
    """Print statistics about a financial statement table"""
    total = session.query(statement_class).count()
    logger.info(f"\n{statement_name} Statistics:")
    logger.info(f"Total records: {total}")
    logger.info(f"By period type:")
    for period_type in ["Annual", "Quarterly"]:
        count = (
            session.query(statement_class).filter_by(period_type=period_type).count()
        )
        logger.info(f"- {period_type}: {count}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Company financials data collector')
    parser = add_environment_args(parser)
    parser.add_argument('--input', type=str, default=str(INPUT_FILE),
                      help='Input file with comma-separated symbols')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                      help='Number of worker threads')
    
    args = parser.parse_args()
    
    # Setup database connection based on environment
    global engine, Session
    engine = get_database_engine(args.env)
    Session = get_session_maker(args.env)
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting company financials collector with {args.workers} threads using {env_name} database")

    try:
        # Load symbols from input file
        input_file = Path(args.input)
        with open(input_file, "r") as f:
            content = f.read()
            symbols = [
                symbol.strip() for symbol in content.split(",") if symbol.strip()
            ]
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        return

    total_symbols = len(symbols)

    # Initialize global stats
    global stats
    stats = Stats(total_symbols)

    # Split symbols into batches for workers
    batch_size = math.ceil(len(symbols) / args.workers)
    symbol_batches = [
        symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)
    ]

    logger.info(f"Starting processing with {args.workers} workers...")
    logger.info(f"Total symbols to process: {total_symbols}")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_symbol_batch, batch) for batch in symbol_batches
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in worker thread: {str(e)}")

    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user!")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        logger.info("\nProcessing Summary:")
        logger.info(f"Total symbols processed: {stats.processed}")
        logger.info(f"Successfully processed: {stats.successful}")
        logger.info(f"Not found (404): {stats.not_found}")
        logger.info(f"Other errors: {stats.other_errors}")

        # Print statistics for each statement type
        try:
            session = Session()
            print_statement_stats(session, IncomeStatement, "Income Statements")
            print_statement_stats(session, BalanceSheet, "Balance Sheets")
            print_statement_stats(session, CashFlow, "Cash Flows")
            session.close()
        except Exception as e:
            logger.error(f"Error getting database statistics: {str(e)}")

        logger.info("Financials collector run completed")


if __name__ == "__main__":
    main()

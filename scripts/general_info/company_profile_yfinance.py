import os
import sys
import argparse
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime
import time
import logging
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import math

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Now import from models
from models.market_data import Company, Executive
from scripts.common_function import get_database_engine, add_environment_args, get_session_maker

# Constants and Configuration
NUM_WORKERS = 6
BATCH_SIZE = 100
MIN_DELAY = 2.0
MAX_DELAY = 4.0
MAX_RETRIES = 2

load_dotenv()

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "../logs"
INPUT_FILE = BASE_DIR / "../symbols/stock_symbols.txt"

thread_local = threading.local()

class Stats:
    def __init__(self, total_symbols):
        self.processed = 0
        self.successful = 0
        self.lock = threading.Lock()
        self.total_symbols = total_symbols
        self.start_time = datetime.now()

    def increment(self, attribute):
        with self.lock:
            setattr(self, attribute, getattr(self, attribute) + 1)
            if attribute == "processed" and self.processed % 10 == 0:
                logger.info(
                    f"Progress: {self.processed}/{self.total_symbols} ({(self.processed / self.total_symbols) * 100:.1f}%)"
                )
                logger.info(f"Successful: {self.successful}")

def setup_logging():
    """Setup logging configuration"""
    log_file = LOG_DIR / f"scraper_{datetime.now().strftime('%Y%m%d')}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def get_thread_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = Session()
    return thread_local.session

def get_company_profile(symbol):
    """Get company profile with retry logic"""
    session = get_thread_session()
    thread_name = threading.current_thread().name

    for attempt in range(MAX_RETRIES):
        try:
            ticker = yf.Ticker(symbol)

            try:
                info = ticker.info
            except Exception as e:
                if "404" in str(e):
                    logger.warning(
                        f"[{thread_name}] Skipping {symbol}: Symbol not found (404)"
                    )
                    return False
                elif "401" in str(e) or "Too Many Requests" in str(e):
                    if attempt < MAX_RETRIES - 1:
                        wait_time = (attempt + 1) * random.uniform(5, 10)
                        logger.warning(
                            f"[{thread_name}] Rate limit hit for {symbol}, waiting {wait_time:.1f}s before retry"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(
                            f"[{thread_name}] Max retries reached for {symbol}"
                        )
                        return False
                raise

            if not isinstance(info, dict):
                logger.error(f"[{thread_name}] Invalid data format for {symbol}")
                return False

            company = session.query(Company).filter_by(symbol=symbol).first()
            if not company:
                company = Company(symbol=symbol)

            try:
                company.company_name = str(info.get("longName", "") or "")
                company.description = str(info.get("longBusinessSummary", "") or "")
                company.sector = str(info.get("sector", "") or "")
                company.industry = str(info.get("industry", "") or "")
                company.employees = int(info.get("fullTimeEmployees", 0) or 0)
                company.website = str(info.get("website", "") or "")
                company.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                company.executives = []

                officers = info.get("companyOfficers", [])
                if isinstance(officers, list):
                    for officer in officers:
                        if isinstance(officer, dict):
                            try:
                                pay_raw = officer.get("totalPay", {})
                                if isinstance(pay_raw, dict):
                                    compensation = float(pay_raw.get("raw", 0) or 0)
                                else:
                                    compensation = 0.0

                                executive = Executive(
                                    name=str(officer.get("name", "") or ""),
                                    title=str(officer.get("title", "") or ""),
                                    year_born=int(officer.get("yearBorn", 0) or 0),
                                    compensation=compensation,
                                )
                                company.executives.append(executive)
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"[{thread_name}] Error processing executive for {symbol}: {str(e)}"
                                )
                                continue

                session.add(company)
                try:
                    session.commit()
                    stats.increment("successful")
                    return True
                except Exception as e:
                    logger.error(f"[{thread_name}] Error committing {symbol}: {str(e)}")
                    session.rollback()
                    return False

            except Exception as e:
                logger.error(
                    f"[{thread_name}] Error processing company data for {symbol}: {str(e)}"
                )
                return False

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"[{thread_name}] Attempt {attempt + 1} failed for {symbol}: {str(e)}"
                )
                time.sleep(random.uniform(2, 5))
                continue
            logger.error(f"[{thread_name}] Error processing {symbol}: {str(e)}")
            return False

    return False

def process_symbol_batch(symbols):
    """Process a batch of symbols"""
    thread_name = threading.current_thread().name
    logger.info(f"[{thread_name}] Starting batch processing of {len(symbols)} symbols")

    for symbol in symbols:
        get_company_profile(symbol)
        stats.increment("processed")
        # Random delay between symbols
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)

    logger.info(f"[{thread_name}] Completed batch processing")

def get_database_stats(session):
    """Calculate database statistics including null values"""
    stats = {}

    company_total = session.query(Company).count()
    stats["companies"] = {
        "total": company_total,
        "null_values": {
            "company_name": session.query(Company)
            .filter(Company.company_name.is_(None))
            .count(),
            "description": session.query(Company)
            .filter(Company.description.is_(None))
            .count(),
            "sector": session.query(Company).filter(Company.sector.is_(None)).count(),
            "industry": session.query(Company)
            .filter(Company.industry.is_(None))
            .count(),
            "employees": session.query(Company)
            .filter(Company.employees.is_(None))
            .count(),
            "website": session.query(Company).filter(Company.website.is_(None)).count(),
        },
    }

    exec_total = session.query(Executive).count()
    stats["executives"] = {
        "total": exec_total,
        "null_values": {
            "name": session.query(Executive).filter(Executive.name.is_(None)).count(),
            "title": session.query(Executive).filter(Executive.title.is_(None)).count(),
            "year_born": session.query(Executive)
            .filter(Executive.year_born.is_(None))
            .count(),
            "compensation": session.query(Executive)
            .filter(Executive.compensation.is_(None))
            .count(),
        },
    }

    stats["companies_with_executives"] = (
        session.query(Company).join(Executive).distinct().count()
    )
    return stats

def log_database_stats(stats, logger):
    """Log database statistics"""
    logger.info("\nDatabase Statistics:")

    # Company statistics
    logger.info("\nCompany Table:")
    logger.info(f"Total Companies: {stats['companies']['total']}")
    logger.info("\nNull Values in Company Fields:")
    for field, count in stats["companies"]["null_values"].items():
        percentage = (
            (count / stats["companies"]["total"] * 100)
            if stats["companies"]["total"] > 0
            else 0
        )
        logger.info(f"- {field}: {count} ({percentage:.1f}%)")

    # Executive statistics
    logger.info("\nExecutive Table:")
    logger.info(f"Total Executives: {stats['executives']['total']}")
    logger.info("\nNull Values in Executive Fields:")
    for field, count in stats["executives"]["null_values"].items():
        percentage = (
            (count / stats["executives"]["total"] * 100)
            if stats["executives"]["total"] > 0
            else 0
        )
        logger.info(f"- {field}: {count} ({percentage:.1f}%)")

    # Relationship statistics
    companies_with_execs = stats["companies_with_executives"]
    companies_without_execs = stats["companies"]["total"] - companies_with_execs
    logger.info("\nRelationship Statistics:")
    logger.info(f"Companies with executives: {companies_with_execs}")
    logger.info(f"Companies without executives: {companies_without_execs}")

    if stats["companies"]["total"] > 0:
        logger.info(
            f"Percentage of companies with executives: {(companies_with_execs / stats['companies']['total'] * 100):.1f}%"
        )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Company profile data collector')
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
    logger.info(f"Starting company profile scraper with {args.workers} threads using {env_name} database")

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
        logger.info(f"Failed: {stats.processed - stats.successful}")

        try:
            session = Session()
            db_stats = get_database_stats(session)
            log_database_stats(db_stats, logger)
            session.close()
        except Exception as e:
            logger.error(f"Error getting database statistics: {str(e)}")

        logger.info("Scraper run completed")

if __name__ == "__main__":
    main()

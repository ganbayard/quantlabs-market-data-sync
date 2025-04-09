import os
import sys
import time
import logging
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import IShareETF, IShareETFHolding
from scripts.common_function import get_database_engine, add_environment_args, get_session_maker
from etf_scraper import ETFScraper

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"etf_holdings_{pd.Timestamp.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Track failed ETFs
failed_etfs = set()
no_listings_etfs = set()


def get_ishare_etf_symbols(session):
    """Get ETF symbols from the database"""
    try:
        etfs = session.query(IShareETF).all()
        return [etf.ticker for etf in etfs if etf.ticker]
    except Exception as e:
        logger.error(f"Error getting ETF symbols from database: {e}")
        return []


def delete_holdings_by_etf_symbol(session, etf_symbol):
    """Delete existing holdings for an ETF"""
    etf_instance = session.query(IShareETF).filter_by(ticker=etf_symbol).first()
    if etf_instance:
        try:
            # Use bulk delete for better performance with MySQL
            count = (
                session.query(IShareETFHolding)
                .filter_by(ishare_etf_id=etf_instance.id)
                .delete()
            )
            session.commit()
            logger.info(f"Deleted {count} holdings for ETF: {etf_symbol}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting holdings for ETF {etf_symbol}: {e}")
    else:
        logger.warning(f"ETF record not found for symbol: {etf_symbol}")


def add_holdings_by_etf_symbol(session, etf_symbol, holdings_df):
    """Add holdings for an ETF to the database"""
    etf_instance = session.query(IShareETF).filter_by(ticker=etf_symbol).first()
    if etf_instance:
        # Replace NaN values with None
        holdings_df = holdings_df.replace({float("nan"): None, "nan": None})
        holdings_df = holdings_df.where(pd.notna(holdings_df), None)

        successful_holdings = 0
        skipped_holdings = 0
        
        for _, row in holdings_df.iterrows():
            try:
                # Convert row to dictionary and handle NaN values
                row_dict = row.to_dict()
                for key, value in row_dict.items():
                    if pd.isna(value):
                        row_dict[key] = None

                # No ticker generation - use ticker as is (even if NULL)
                ticker = row_dict.get("ticker")

                # Truncate strings if they're too long
                if ticker and len(ticker) > 20:
                    ticker = ticker[:20]
                
                # Handle missing 'weight' column
                weight = row_dict.get("weight", 0.0)
                if weight is None or weight == "":
                    weight = 0.0
                else:
                    weight = float(weight)

                # Handle numeric values
                market_value = (
                    float(row_dict.get("market_value", 0.0))
                    if row_dict.get("market_value") is not None
                    else 0.0
                )
                notional_value = (
                    float(row_dict.get("notional_value", 0.0))
                    if row_dict.get("notional_value") is not None
                    else 0.0
                )
                amount = (
                    float(row_dict.get("amount", 0.0))
                    if row_dict.get("amount") is not None
                    else 0.0
                )
                price = (
                    float(row_dict.get("price", 0.0))
                    if row_dict.get("price") is not None
                    else 0.0
                )
                fx_rate = (
                    float(row_dict.get("fx_rate", 1.0))
                    if row_dict.get("fx_rate") is not None
                    else 1.0
                )

                # Handle dates
                accrual_date = (
                    pd.to_datetime(row_dict.get("accrual_date")).date()
                    if pd.notna(row_dict.get("accrual_date"))
                    else None
                )
                as_of_date = (
                    pd.to_datetime(row_dict.get("as_of_date")).date()
                    if pd.notna(row_dict.get("as_of_date"))
                    else None
                )

                # Truncate strings if they're too long
                name = row_dict.get("name", "")[:255] if row_dict.get("name") else ""
                sector = row_dict.get("sector", "")[:100] if row_dict.get("sector") else ""
                asset_class = row_dict.get("asset_class", "Unknown")[:100] if row_dict.get("asset_class") else "Unknown"
                location = row_dict.get("location", "")[:100] if row_dict.get("location") else ""
                exchange = row_dict.get("exchange", "")[:100] if row_dict.get("exchange") else ""
                currency = row_dict.get("currency", "")[:3] if row_dict.get("currency") else ""
                market_currency = row_dict.get("market_currency", "")[:3] if row_dict.get("market_currency") else ""
                fund_ticker = row_dict.get("fund_ticker", "")[:10] if row_dict.get("fund_ticker") else ""

                new_holding = IShareETFHolding(
                    ishare_etf_id=etf_instance.id,
                    ticker=ticker,  # Can be NULL now
                    name=name,
                    sector=sector,
                    asset_class=asset_class,
                    market_value=market_value,
                    weight=weight,
                    notional_value=notional_value,
                    amount=amount,
                    price=price,
                    location=location,
                    exchange=exchange,
                    currency=currency,
                    fx_rate=fx_rate,
                    market_currency=market_currency,
                    accrual_date=accrual_date,
                    fund_ticker=fund_ticker,
                    as_of_date=as_of_date,
                )
                session.add(new_holding)
                successful_holdings += 1
                
                # Commit in small batches to avoid large transactions
                if successful_holdings % 50 == 0:
                    session.commit()
                
            except Exception as e:
                skipped_holdings += 1
                logger.error(f"Error processing holding for ETF {etf_symbol}: {e}")
                continue

        try:
            # Commit any remaining holdings
            session.commit()
            logger.info(f"Added {successful_holdings} holdings to ETF: {etf_symbol} ({skipped_holdings} skipped)")
        except Exception as e:
            session.rollback()
            logger.error(f"Error committing holdings for ETF {etf_symbol}: {e}")
    else:
        logger.warning(f"ETF record not found for symbol: {etf_symbol}")


def update_etf_holdings(session, etf_symbol, retry=False):
    """Update holdings for an ETF"""
    etf_scraper = ETFScraper()

    try:
        holdings_df = etf_scraper.query_holdings(etf_symbol, None)
        logger.info(f"Retrieved {len(holdings_df)} holdings for ETF: {etf_symbol}")

        delete_holdings_by_etf_symbol(session, etf_symbol)
        add_holdings_by_etf_symbol(session, etf_symbol, holdings_df)

        logger.info(f"Successfully updated holdings for ETF: {etf_symbol}")
        if retry and etf_symbol in failed_etfs:
            failed_etfs.remove(etf_symbol)
    except Exception as e:
        if "No listings found" in str(e):
            logger.error(f"No listings found for ETF {etf_symbol}")
            no_listings_etfs.add(etf_symbol)
        else:
            logger.error(f"Error updating holdings for ETF {etf_symbol}: {e}")
        failed_etfs.add(etf_symbol)


def main():
    """Main function to run the ETF holdings updater"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETF holdings updater')
    parser = add_environment_args(parser)
    parser.add_argument('--retry', action='store_true', help='Retry previously failed ETFs')
    parser.add_argument('--symbol', type=str, help='Process a specific ETF symbol')
    args = parser.parse_args()
    
    # Setup database connection based on environment
    engine = get_database_engine(args.env)
    Session = get_session_maker(args.env)
    session = Session()
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting ETF holdings updater using {env_name} database")

    try:
        # Process a single symbol if specified
        if args.symbol:
            logger.info(f"Processing holdings for single ETF: {args.symbol}")
            update_etf_holdings(session, args.symbol)
            processed_count = 1
            total_count = 1
        else:
            # Get all ETF symbols
            etf_symbols = get_ishare_etf_symbols(session)
            total_count = len(etf_symbols)
            logger.info(f"Found {total_count} ETF symbols in the database")

            # Track progress
            processed_count = 0
            start_time = datetime.now()

            # Update holdings for each ETF
            for symbol in etf_symbols:
                update_etf_holdings(session, symbol)
                
                # Update and log progress
                processed_count += 1
                
                # Calculate completion percentage and ETFs per minute
                percentage = (processed_count / total_count) * 100
                elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
                etfs_per_minute = processed_count / elapsed_minutes if elapsed_minutes > 0 else 0
                
                # Calculate estimated time remaining
                remaining_etfs = total_count - processed_count
                remaining_minutes = remaining_etfs / etfs_per_minute if etfs_per_minute > 0 else 0
                
                # Log progress every 5th ETF or when reaching milestones
                if processed_count % 5 == 0 or processed_count == total_count:
                    logger.info(f"Progress: {processed_count}/{total_count} ETFs processed ({percentage:.1f}%)")
                    logger.info(f"Rate: {etfs_per_minute:.1f} ETFs/min, Est. time remaining: {remaining_minutes:.1f} minutes")
                    logger.info(f"Success: {processed_count - len(failed_etfs) - len(no_listings_etfs)}, Failed: {len(failed_etfs)}, No listings: {len(no_listings_etfs)}")
                
                # Add a short delay to avoid hammering the API
                time.sleep(1)

        # Log results
        total_etfs = processed_count
        successful_etfs_count = total_etfs - len(failed_etfs) - len(no_listings_etfs)

        logger.info("\nETF holdings update completed.")
        logger.info(f"Total ETFs processed: {processed_count}/{total_count}")
        logger.info(f"Successfully updated ETFs: {successful_etfs_count}")
        logger.info(f"Failed to update ETFs: {len(failed_etfs)}")
        logger.info(f"ETFs with no listings found: {len(no_listings_etfs)}")
        
        if failed_etfs:
            logger.info(f"ETFs that couldn't be processed: {', '.join(sorted(failed_etfs))}")
        if no_listings_etfs:
            logger.info(f"ETFs with no listings: {', '.join(sorted(no_listings_etfs))}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

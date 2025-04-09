import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
from sqlalchemy.exc import IntegrityError
from etf_scraper import ETFScraper
from etf_scraper.scrapers import (
    ISharesListings,
    SSGAListings,
    VanguardListings,
    InvescoListings,
)

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import IShareETF
from scripts.common_function import get_database_engine, add_environment_args, get_session_maker

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"etf_list_{pd.Timestamp.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_etf_symbols():
    """Fetch ETF symbols from multiple providers"""
    all_listings = []
    providers = [
        ("IShares", ISharesListings),
        ("SSGA", SSGAListings),
        ("Vanguard", VanguardListings),
        ("Invesco", InvescoListings),
    ]

    for provider_name, provider_class in providers:
        try:
            logger.info(f"Fetching listings for {provider_name}...")
            listings = provider_class.retrieve_listings()
            logger.info(f"Retrieved {len(listings)} listings for {provider_name}")

            etfs = listings[listings["fund_type"] == "ETF"]
            etfs["provider"] = provider_name
            logger.info(f"Found {len(etfs)} ETFs for {provider_name}")

            all_listings.append(etfs)
        except Exception as e:
            logger.error(f"Error fetching listings for {provider_name}: {str(e)}")

    if not all_listings:
        logger.warning("No listings were retrieved from any provider.")
        return pd.DataFrame()

    # Combine all listings
    all_etfs = pd.concat(all_listings, ignore_index=True)
    # Remove duplicates and sort
    unique_etfs = all_etfs.drop_duplicates(subset="ticker").sort_values("ticker")
    return unique_etfs


def update_database(session, etfs_df):
    """Update ETF information in the database"""
    inserted = 0
    updated = 0
    
    try:
        for _, etf in etfs_df.iterrows():
            etf_dict = etf.to_dict()
            for key, value in etf_dict.items():
                if pd.isna(value):
                    etf_dict[key] = 0 if key == "net_assets" else ""
                    
            # Clean up inception_date if it exists
            if "inception_date" in etf_dict and etf_dict["inception_date"] != "":
                try:
                    etf_dict["inception_date"] = pd.to_datetime(etf_dict["inception_date"]).date()
                except:
                    etf_dict["inception_date"] = None
            
            # Check if ETF already exists
            existing_etf = session.query(IShareETF).filter_by(ticker=etf_dict.get("ticker")).first()
            
            if existing_etf:
                # Update existing ETF
                for key, value in etf_dict.items():
                    if hasattr(existing_etf, key):
                        setattr(existing_etf, key, value)
                updated += 1
            else:
                # Create new ETF
                new_etf = IShareETF(**etf_dict)
                session.add(new_etf)
                inserted += 1
                
            # Commit in batches to avoid long transactions
            if (inserted + updated) % 100 == 0:
                session.commit()
                
        # Final commit for remaining records
        session.commit()
        logger.info(f"Successfully updated ETFs in the database: {inserted} inserted, {updated} updated")
    except Exception as e:
        session.rollback()
        logger.error(f"Error updating database: {str(e)}")
        raise


def main():
    """Main function to run the ETF list updater"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETF list data updater')
    parser = add_environment_args(parser)
    args = parser.parse_args()
    
    # Setup database connection based on environment
    engine = get_database_engine(args.env)
    Session = get_session_maker(args.env)
    session = Session()
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting ETF list updater using {env_name} database")

    try:
        logger.info("Starting ETF data retrieval and database update...")
        etf_data = get_etf_symbols()

        if etf_data.empty:
            logger.warning("No ETF data was retrieved.")
            return

        logger.info(f"Total number of unique ETFs: {len(etf_data)}")
        update_database(session, etf_data)
        logger.info("ETF database update completed.")
    except Exception as e:
        logger.error(f"ETF list update failed: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

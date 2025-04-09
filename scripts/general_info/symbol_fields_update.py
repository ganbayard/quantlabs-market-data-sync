import os
import sys
import time
import json
import requests
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import logging

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models
from models.market_data import SymbolFields
from scripts.common_function import get_database_engine, add_environment_args, get_session_maker

# Constants and Configuration
BATCH_SIZE = 1000
REQUEST_DELAY = 3  # seconds between API requests

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"symbol_fields_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# TradingView API configuration
url = "https://scanner.tradingview.com/america/scan"
payload = {
    "columns": [
        "name",
        "description",
        "close",
        "change",
        "volume",
        "market_cap_basic",
        "market",
        "sector",
        "industry.tr",
        "earnings_release_trading_date_fq",
        "earnings_release_next_trading_date_fq",
        "indexes.tr",
        "country.tr",
        "exchange.tr",
    ],
    "ignore_unknown_fields": False,
    "options": {"lang": "en"},
    "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
    "symbols": {},
    "markets": ["america"],
    "filter2": {
        "operator": "and",
        "operands": [
            {
                "operation": {
                    "operator": "or",
                    "operands": [
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [
                                    {
                                        "expression": {
                                            "left": "type",
                                            "operation": "equal",
                                            "right": "stock",
                                        }
                                    },
                                    {
                                        "expression": {
                                            "left": "typespecs",
                                            "operation": "has",
                                            "right": ["common"],
                                        }
                                    },
                                ],
                            }
                        },
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [
                                    {
                                        "expression": {
                                            "left": "type",
                                            "operation": "equal",
                                            "right": "stock",
                                        }
                                    },
                                    {
                                        "expression": {
                                            "left": "typespecs",
                                            "operation": "has",
                                            "right": ["preferred"],
                                        }
                                    },
                                ],
                            }
                        },
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [
                                    {
                                        "expression": {
                                            "left": "type",
                                            "operation": "equal",
                                            "right": "dr",
                                        }
                                    }
                                ],
                            }
                        },
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [
                                    {
                                        "expression": {
                                            "left": "type",
                                            "operation": "equal",
                                            "right": "fund",
                                        }
                                    },
                                    {
                                        "expression": {
                                            "left": "typespecs",
                                            "operation": "has_none_of",
                                            "right": ["etf"],
                                        }
                                    },
                                ],
                            }
                        },
                    ],
                }
            }
        ],
    },
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.7",
    "Content-Type": "text/plain;charset=UTF-8",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
    "Sec-GPC": "1",
}


def convert_timestamp_to_date(timestamp):
    """Convert Unix timestamp to date string"""
    if timestamp:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    return None


def fetch_and_update_data(session, batch_size=BATCH_SIZE, request_delay=REQUEST_DELAY):
    """Fetch data from TradingView API and update database"""
    offset = 0
    total_updated = 0
    total_inserted = 0

    try:
        while True:
            try:
                payload["range"] = [offset, offset + batch_size]
                logger.info(f"Fetching data from offset {offset} to {offset + batch_size}")
                
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data_json = response.json()

                if not data_json.get("data"):
                    logger.info("No more data to process")
                    break

                # Process batch items
                for item in data_json["data"]:
                    try:
                        # Create a session.begin() block to handle transactions per record
                        with session.begin_nested():
                            data = {
                                'symbol': item['d'][0],
                                'company_name': item['d'][1],
                                'price': item['d'][2],
                                'change': item['d'][3],
                                'volume': item['d'][4],
                                'market_cap': None,  # Initialize as None first
                                'market': item['d'][6],
                                'sector': item['d'][7],
                                'industry': item['d'][8],
                                'earnings_release_trading_date_fq': convert_timestamp_to_date(item['d'][9]),
                                'earnings_release_next_trading_date_fq': convert_timestamp_to_date(item['d'][10]),
                                'indexes': json.dumps(item['d'][11]) if isinstance(item['d'][11], (list, dict)) else item['d'][11],
                                'country': item['d'][12],
                                'exchange': item['d'][13]
                            }
                            
                            # Handle the market_cap field properly (convert to float or None)
                            if item['d'][5] != '' and item['d'][5] is not None:
                                try:
                                    data['market_cap'] = float(item['d'][5])
                                except (ValueError, TypeError):
                                    data['market_cap'] = None
                            
                            # Handle empty string indexes
                            if data['indexes'] == '':
                                data['indexes'] = None
                                
                            # Handle None values for string fields
                            string_fields = ['company_name', 'market', 'sector', 'industry', 'country', 'exchange']
                            for field in string_fields:
                                if data[field] is None:
                                    data[field] = ''

                            existing_record = session.query(SymbolFields).filter_by(symbol=data["symbol"]).first()
                            if existing_record:
                                for key, value in data.items():
                                    setattr(existing_record, key, value)
                                total_updated += 1
                            else:
                                new_record = SymbolFields(**data)
                                session.add(new_record)
                                total_inserted += 1
                    except Exception as item_error:
                        logger.error(f"Error processing symbol {item['d'][0] if 'd' in item and len(item['d']) > 0 else 'unknown'}: {item_error}")
                        continue

                # Commit the entire batch
                session.commit()
                
                count = len(data_json["data"])
                logger.info(f"Processed {count} tickers (offset {offset}) - Running totals: {total_inserted} inserted, {total_updated} updated")
                offset += batch_size
                time.sleep(request_delay)  # Prevent rate limiting

            except requests.RequestException as req_error:
                logger.error(f"API request failed: {req_error}")
                break
            except Exception as e:
                logger.error(f"Unexpected error during batch processing: {e}")
                session.rollback()
                break

    except Exception as main_error:
        logger.error(f"Critical error in fetch_and_update_data: {main_error}")
        session.rollback()
        raise

    logger.info(f"Total records processed: {total_updated + total_inserted}")
    logger.info(f"Total records updated: {total_updated}")
    logger.info(f"Total records inserted: {total_inserted}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Symbol Fields data updater')
    parser = add_environment_args(parser)
    parser.add_argument('--delay', type=int, default=REQUEST_DELAY,
                        help='Delay between API requests in seconds')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE,
                        help='Batch size for API requests')
    
    args = parser.parse_args()
    
    # Setup database connection based on environment
    engine = get_database_engine(args.env)
    Session = get_session_maker(args.env)
    session = Session()
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting symbol fields update using {env_name} database")
    logger.info(f"API request delay: {args.delay}s, Batch size: {args.batch}")

    try:
        fetch_and_update_data(session, args.batch, args.delay)
        logger.info("Symbol fields update completed successfully")
    except Exception as e:
        logger.error(f"Symbol fields update failed: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import sys
import time
import json
import requests
import argparse
from datetime import datetime
from pathlib import Path
from decimal import Decimal, InvalidOperation
import logging

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models
from models.market_data import SymbolFields
from scripts.common_function import get_database_engine, add_environment_args, get_session_maker

# Constants and Configuration
BATCH_SIZE = 1000  # Increased batch size for full run
REQUEST_DELAY = 1   # Reduced delay slightly for faster full run, monitor for rate limits

# Configure logging
LOG_DIR = project_root / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"symbol_fields_etf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# TradingView API configuration
url = "https://scanner.tradingview.com/america/scan"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.7",
    "Content-Type": "text/plain;charset=UTF-8",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
    "Sec-GPC": "1",
}

# --- Stock Payload --- (Based on original script)
stock_payload = {
    "columns": [
        "name",                     # 0: Symbol (e.g., AAPL)
        "description",              # 1: Company Name (e.g., Apple Inc)
        "close",                    # 2: Last Price
        "change",                   # 3: Change %
        "volume",                   # 4: Volume
        "market_cap_basic",         # 5: Market Cap
        "market",                   # 6: Market (e.g., stocks)
        "sector",                   # 7: Sector
        "industry.tr",              # 8: Industry
        "earnings_release_trading_date_fq", # 9: Last Earnings Date (Timestamp)
        "earnings_release_next_trading_date_fq", # 10: Next Earnings Date (Timestamp)
        "indexes.tr",               # 11: Indexes (List or string)
        "country.tr",               # 12: Country
        "exchange.tr",              # 13: Exchange
        "type",                     # 14: Ensure we get type to double-check
        "typespecs",                # 15: Ensure we get typespecs
    ],
    "ignore_unknown_fields": False,
    "options": {"lang": "en"},
    "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
    "symbols": {},
    "markets": ["america"],
    # Filter for common/preferred stocks, DRs, non-ETF funds (as per original script)
    "filter2": {
        "operator": "and",
        "operands": [
            {
                "operation": {
                    "operator": "or",
                    "operands": [
                        # Common Stock
                        {"operation": {"operator": "and", "operands": [
                            {"expression": {"left": "type", "operation": "equal", "right": "stock"}},
                            {"expression": {"left": "typespecs", "operation": "has", "right": ["common"]}}
                        ]}},
                        # Preferred Stock
                        {"operation": {"operator": "and", "operands": [
                            {"expression": {"left": "type", "operation": "equal", "right": "stock"}},
                            {"expression": {"left": "typespecs", "operation": "has", "right": ["preferred"]}}
                        ]}},
                        # Depository Receipt
                        {"operation": {"operator": "and", "operands": [
                            {"expression": {"left": "type", "operation": "equal", "right": "dr"}}
                        ]}},
                        # Fund (Non-ETF)
                        {"operation": {"operator": "and", "operands": [
                            {"expression": {"left": "type", "operation": "equal", "right": "fund"}},
                            {"expression": {"left": "typespecs", "operation": "has_none_of", "right": ["etf"]}}
                        ]}},
                    ]
                }
            }
        ]
    },
}

# --- ETF Payload --- (Using validated columns)
etf_payload = {
    "columns": [
        "name",                     # 0: Ticker
        "description",              # 1: Fund Name
        "close",                    # 2: Last Price
        "volume",                   # 3: Volume
        "exchange.tr",              # 4: Exchange
        "type",                     # 5: Should be 'fund'
        "typespecs",                # 6: Should include 'etf'
        "market",                   # 7: Should be 'etf' or similar
        "country.tr",               # 8: Country
        "change",                   # 9: Change %
        "expense_ratio",            # 10: Expense Ratio
        "relative_volume_10d_calc", # 11: Relative Volume
        "asset_class.tr",           # 12: Asset Class
        "focus.tr",                 # 13: Focus
    ],
    "ignore_unknown_fields": False,
    "options": {"lang": "en"},
    "sort": {"sortBy": "volume", "sortOrder": "desc"}, # Sort by volume for ETFs
    "symbols": {},
    "markets": ["america"],
    # Filter specifically for ETFs
    "filter2": {
        "operator": "and",
        "operands": [
            {
                "operation": {
                    "operator": "and",
                    "operands": [
                        {"expression": {"left": "type", "operation": "equal", "right": "fund"}},
                        {"expression": {"left": "typespecs", "operation": "has", "right": ["etf"]}}
                    ]
                }
            }
        ]
    },
}

# --- Helper Functions ---
def safe_decimal(value, default=None, precision=4):
    """Safely convert value to Decimal or return default."""
    if value is None or value == '':
        return default
    try:
        # Attempt direct conversion first
        d = Decimal(value)
        return d
    except (ValueError, TypeError, InvalidOperation):
        logger.warning(f"Could not convert '{value}' to Decimal, using default ({default})")
        return default

def safe_float(value, default=None):
    """Safely convert value to float or return default."""
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{value}' to float, using default ({default})")
        return default

def safe_int(value, default=None):
    """Safely convert value to int or return default."""
    if value is None or value == '':
        return default
    try:
        # Handle potential floats coming from API before int conversion
        return int(float(value))
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{value}' to int, using default ({default})")
        return default

def convert_timestamp_to_datetime(timestamp):
    """Convert Unix timestamp to datetime object or return None."""
    if timestamp:
        try:
            # Assuming timestamp is in seconds
            return datetime.fromtimestamp(int(timestamp))
        except (ValueError, TypeError):
            logger.warning(f"Could not convert timestamp '{timestamp}' to datetime")
            return None
    return None

def fetch_and_update_data(session, api_payload, asset_type, batch_size=BATCH_SIZE, request_delay=REQUEST_DELAY, max_records=None):
    """Fetch data from TradingView API and update database for a specific asset type."""
    offset = 0
    total_processed_session = 0
    total_updated_session = 0
    total_inserted_session = 0
    is_etf = (asset_type == "etf")
    column_map = {name: idx for idx, name in enumerate(api_payload["columns"])}

    logger.info(f"Starting fetch for asset type: {asset_type.upper()}")

    try:
        while True:
            # Check max_records limit before fetching
            if max_records is not None and total_processed_session >= max_records:
                logger.info(f"Reached max_records limit ({max_records}) for {asset_type}. Stopping fetch.")
                break

            current_batch_size = batch_size
            # Adjust last batch size if max_records is set
            if max_records is not None:
                remaining = max_records - total_processed_session
                if remaining < batch_size:
                    current_batch_size = remaining
            
            if current_batch_size <= 0:
                break

            current_payload = api_payload.copy()
            current_payload["range"] = [offset, offset + current_batch_size]
            logger.info(f"Fetching {asset_type} data from offset {offset} to {offset + current_batch_size}")

            try:
                response = requests.post(url, json=current_payload, headers=headers)
                response.raise_for_status()
                data_json = response.json()

                if not data_json.get("data"):
                    logger.info(f"No more {asset_type} data found at offset {offset}. Ending fetch for this type.")
                    break

                batch_items = data_json["data"]
                count = len(batch_items)
                logger.info(f"Received {count} {asset_type} items for offset {offset}")
                total_processed_session += count

                # Process batch items
                for item in batch_items:
                    d = item.get('d')
                    if not d or len(d) < len(column_map):
                        logger.warning(f"Skipping malformed item (data length mismatch or missing 'd'): {item}")
                        continue

                    symbol = d[column_map["name"]]
                    if not symbol:
                        logger.warning(f"Skipping item with missing symbol: {item}")
                        continue

                    try:
                        # Prepare data dictionary
                        data = {
                            'symbol': symbol,
                            'is_etf': is_etf,
                            'company_name': d[column_map["description"]],
                            'price': safe_decimal(d[column_map["close"]], precision=4),
                            'change': safe_decimal(d[column_map.get("change")], precision=4),
                            'volume': safe_int(d[column_map.get("volume")]),
                            'country': d[column_map.get("country.tr")],
                            'exchange': d[column_map.get("exchange.tr")],
                            'market': d[column_map.get("market")],
                            # Initialize all fields to None or default
                            'market_cap': None,
                            'sector': None,
                            'industry': None,
                            'earnings_release_trading_date_fq': None,
                            'earnings_release_next_trading_date_fq': None,
                            'indexes': None,
                            'relative_volume': None,
                            'aum': None, 
                            'nav_total_return_3y': None, 
                            'expense_ratio': None,
                            'asset_class': None,
                            'focus': None,
                        }

                        # Populate type-specific fields
                        if is_etf:
                            data['relative_volume'] = safe_float(d[column_map.get("relative_volume_10d_calc")])
                            data['expense_ratio'] = safe_decimal(d[column_map.get("expense_ratio")], precision=4)
                            data['asset_class'] = d[column_map.get("asset_class.tr")]
                            data['focus'] = d[column_map.get("focus.tr")]
                        else: # Stock
                            data['market_cap'] = safe_decimal(d[column_map.get("market_cap_basic")], precision=4)
                            data['sector'] = d[column_map.get("sector")]
                            data['industry'] = d[column_map.get("industry.tr")]
                            data['earnings_release_trading_date_fq'] = convert_timestamp_to_datetime(d[column_map.get("earnings_release_trading_date_fq")])
                            data['earnings_release_next_trading_date_fq'] = convert_timestamp_to_datetime(d[column_map.get("earnings_release_next_trading_date_fq")])
                            indexes_raw = d[column_map.get("indexes.tr")]
                            data['indexes'] = json.dumps(indexes_raw) if isinstance(indexes_raw, (list, dict)) else indexes_raw
                            if data['indexes'] == '': data['indexes'] = None # Handle empty string

                        # Clean None values for string fields
                        string_fields = ['company_name', 'market', 'sector', 'industry', 'country', 'exchange', 'asset_class', 'focus']
                        for field in string_fields:
                            if data[field] is None:
                                data[field] = None 

                        # Upsert logic
                        existing_record = session.query(SymbolFields).filter_by(symbol=data["symbol"]).first()
                        if existing_record:
                            # Check if asset type matches
                            if existing_record.is_etf != is_etf:
                                logger.warning(f"Type mismatch for symbol {symbol}. DB: {'ETF' if existing_record.is_etf else 'Stock'}, API: {'ETF' if is_etf else 'Stock'}. Overwriting type.")
                                existing_record.is_etf = is_etf 
                            
                            # Update existing record
                            for key, value in data.items():
                                setattr(existing_record, key, value)
                            total_updated_session += 1
                        else:
                            # Insert new record
                            new_record = SymbolFields(**data)
                            session.add(new_record)
                            total_inserted_session += 1

                    except Exception as item_error:
                        logger.error(f"Error processing symbol {symbol}: {item_error}", exc_info=True)
                        session.rollback() # Rollback specific item transaction
                        continue # Continue with the next item in the batch

                # Commit the batch transaction
                try:
                    session.commit()
                    logger.info(f"Committed batch for offset {offset}. Running totals ({asset_type}): {total_inserted_session} inserted, {total_updated_session} updated.")
                except Exception as commit_error:
                    logger.error(f"Error committing batch for offset {offset}: {commit_error}", exc_info=True)
                    session.rollback() # Rollback the whole batch on commit error
                    logger.warning("Rolling back batch due to commit error.")

                # Move to the next batch
                offset += count # Increment offset by the actual number of items received
                time.sleep(request_delay)  # Prevent rate limiting

            except requests.RequestException as req_error:
                logger.error(f"API request failed for {asset_type} at offset {offset}: {req_error}")
                if req_error.response is not None:
                    logger.error(f"API Error Response Status: {req_error.response.status_code}")
                    logger.error(f"API Error Response Body: {req_error.response.text}")
                logger.warning("Stopping fetch for this asset type due to API error.")
                break
            except Exception as e:
                logger.error(f"Unexpected error during {asset_type} batch processing at offset {offset}: {e}", exc_info=True)
                session.rollback()
                logger.warning("Stopping fetch for this asset type due to unexpected error.")
                break

    except Exception as main_error:
        logger.error(f"Critical error in fetch_and_update_data for {asset_type}: {main_error}", exc_info=True)
        session.rollback()
        raise

    logger.info(f"Finished fetch for asset type: {asset_type.upper()}")
    logger.info(f"Total {asset_type} records processed in this run: {total_processed_session}")
    logger.info(f"Total {asset_type} records inserted in this run: {total_inserted_session}")
    logger.info(f"Total {asset_type} records updated in this run: {total_updated_session}")

    return total_inserted_session, total_updated_session

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Symbol Fields data updater for Stocks and ETFs')
    parser = add_environment_args(parser)
    parser.add_argument('--delay', type=int, default=REQUEST_DELAY,
                        help='Delay between API requests in seconds')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE,
                        help='Batch size for API requests')
    parser.add_argument('--max_stocks', type=int, default=None,
                        help='Maximum number of stock records to fetch (for testing)')
    parser.add_argument('--max_etfs', type=int, default=None,
                        help='Maximum number of ETF records to fetch')

    args = parser.parse_args()

    # Setup database connection based on environment
    engine = get_database_engine(args.env)
    Session = get_session_maker(args.env)
    session = Session()
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting symbol fields update using {env_name} database")
    logger.info(f"API request delay: {args.delay}s, Batch size: {args.batch}")
    if args.max_stocks: logger.info(f"Max stock records to fetch: {args.max_stocks}")
    if args.max_etfs: logger.info(f"Max ETF records to fetch: {args.max_etfs}")

    total_inserted = 0
    total_updated = 0

    try:
        # Fetch and update Stocks
        logger.info("--- Starting Stock Update Phase ---")
        inserted, updated = fetch_and_update_data(session, stock_payload, "stock", args.batch, args.delay, args.max_stocks)
        total_inserted += inserted
        total_updated += updated
        logger.info("--- Finished Stock Update Phase ---")

        # Fetch and update ETFs
        logger.info("--- Starting ETF Update Phase ---")
        inserted, updated = fetch_and_update_data(session, etf_payload, "etf", args.batch, args.delay, args.max_etfs)
        total_inserted += inserted
        total_updated += updated
        logger.info("--- Finished ETF Update Phase ---")

        logger.info("Symbol fields update completed.")
        logger.info(f"Overall totals: {total_inserted} inserted, {total_updated} updated.")

    except Exception as e:
        logger.error(f"Symbol fields update failed during execution: {e}", exc_info=True)
    finally:
        if session:
            session.close()
            logger.info("Database session closed.")

if __name__ == "__main__":
    main()
import os
import sys
import argparse
import yfinance as yf
from datetime import datetime, timedelta
import threading
import time
import logging
import random
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from models and common functions
from models.market_data import NewsArticle
from scripts.common_function import get_database_engine, add_environment_args, get_session_maker
from sqlalchemy import func, inspect

# Constants and Configuration
NUM_WORKERS = 6
MIN_DELAY = 1.0
MAX_DELAY = 2.0
MAX_RETRIES = 1
DEFAULT_DAYS = 60
MIN_ARTICLES = 10
MAX_ARTICLES = 20

# File paths
BASE_DIR = Path(__file__).parent
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE = Path(__file__).parent.parent / "symbols/stock_symbols.txt"

# Set up logging
log_file = LOG_DIR / f"news_collector_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Thread-local storage
thread_local = threading.local()


class Stats:
    def __init__(self, total_symbols):
        self.processed = 0
        self.successful = 0
        self.articles_added = 0
        self.articles_removed = 0
        self.symbols_maintained = 0
        self.lock = threading.Lock()
        self.total_symbols = total_symbols
        self.start_time = datetime.now()
    
    def increment(self, attribute, value=1):
        with self.lock:
            setattr(self, attribute, getattr(self, attribute) + value)
            if attribute == 'processed' and self.processed % 20 == 0:
                elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                symbols_per_minute = self.processed / elapsed if elapsed > 0 else 0
                estimated_remaining = (self.total_symbols - self.processed) / symbols_per_minute if symbols_per_minute > 0 else 0
                
                logger.info(f"Progress: {self.processed}/{self.total_symbols} ({(self.processed/self.total_symbols)*100:.1f}%)")
                logger.info(f"Maintained: {self.symbols_maintained}, Added: {self.articles_added}, Removed: {self.articles_removed}")
                logger.info(f"Rate: {symbols_per_minute:.1f} symbols/min, Est. remaining: {estimated_remaining:.1f} minutes")


def get_thread_session():
    """Get thread-local SQLAlchemy session"""
    if not hasattr(thread_local, "session"):
        thread_local.session = Session()
    return thread_local.session


def extract_date_from_iso(date_str):
    """Extract date from ISO format string and return timezone-naive datetime"""
    try:
        from dateutil import parser
        
        # Parse with dateutil which handles multiple formats
        dt = parser.parse(date_str)
        
        # Convert to timezone-naive by replacing tzinfo with None
        return dt.replace(tzinfo=None)
    except Exception as e:
        logger.debug(f"Failed to parse date '{date_str}': {e}")
        return None


def fetch_news_for_symbol(symbol, days=DEFAULT_DAYS, delay=None):
    """Fetch news for a single symbol with proper timezone and related symbols handling"""
    thread_name = threading.current_thread().name
    
    # Use random delay within range if none specified
    if delay is None:
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
    
    for attempt in range(MAX_RETRIES):
        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news
            
            if not news_items:
                logger.debug(f"[{thread_name}] No news data available for {symbol}")
                time.sleep(delay)
                return []
                
            logger.debug(f"[{thread_name}] Fetched {len(news_items)} raw news items for {symbol}")
            
            articles = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for i, news_item in enumerate(news_items):
                try:
                    # Handle the nested content structure
                    if 'content' in news_item and isinstance(news_item['content'], dict):
                        article = news_item['content']
                    else:
                        article = news_item  # Fallback to direct structure
                    
                    # Extract title
                    title = article.get('title')
                    if not title:
                        continue
                    
                    # Extract date from pubDate field (seen in the actual data)
                    published_date = None
                    if 'pubDate' in article and article['pubDate']:
                        published_date = extract_date_from_iso(article['pubDate'])
                    
                    # If no valid date, skip this article
                    if not published_date:
                        continue
                    
                    # Skip old articles - both dates are now timezone-naive
                    if published_date < cutoff_date:
                        continue
                    
                    # Extract provider/publisher
                    publisher = "Yahoo Finance"  # Default
                    if 'provider' in article and isinstance(article['provider'], dict):
                        provider_obj = article['provider']
                        if 'displayName' in provider_obj:
                            publisher = provider_obj['displayName']
                    
                    # Extract link
                    link = None
                    if 'canonicalUrl' in article and isinstance(article['canonicalUrl'], dict):
                        link = article['canonicalUrl'].get('url')
                    elif 'clickThroughUrl' in article and isinstance(article['clickThroughUrl'], dict):
                        link = article['clickThroughUrl'].get('url')

                    thumbnail = None 
                    if 'thumbnail' in article and isinstance(article['thumbnail'], dict):
                        thumbnail = article['thumbnail'].get('resolutions')[1].get('url')

                    if not link:
                        # Default link to Yahoo Finance quote page
                        link = f"https://finance.yahoo.com/quote/{symbol}"
                    
                    # Extract content type
                    content_type = article.get('contentType', 'STORY')
                    
                    # Extract summary
                    summary = article.get('summary', '')
                    if not summary and 'description' in article:
                        summary = article['description']
                    
                    # Sanitize summary (remove HTML tags)
                    if summary and '<' in summary:
                        summary = re.sub('<[^<]+?>', '', summary)
                    
                    # Extract related symbols - check multiple possible fields and formats
                    related_symbols = []
                    
                    # Method 1: Direct relatedTickers field
                    if news_item.get('relatedTickers'):
                        if isinstance(news_item['relatedTickers'], list):
                            related_symbols.extend(news_item['relatedTickers'])
                        elif isinstance(news_item['relatedTickers'], str):
                            related_symbols.append(news_item['relatedTickers'])
                    
                    # Method 2: Content nested relatedTickers
                    if 'content' in news_item and isinstance(news_item['content'], dict):
                        content = news_item['content']
                        if content.get('relatedTickers'):
                            if isinstance(content['relatedTickers'], list):
                                related_symbols.extend(content['relatedTickers'])
                            elif isinstance(content['relatedTickers'], str):
                                related_symbols.append(content['relatedTickers'])
                    
                    # Method 3: Look for ticker symbols in the description/summary
                    if summary:
                        # Look for stock symbols in parentheses (e.g., "(AAPL)")
                        ticker_matches = re.findall(r'\(([A-Z]{1,5})\)', summary)
                        if ticker_matches:
                            related_symbols.extend(ticker_matches)
                    
                    # Always include the main symbol if no related symbols found
                    if not related_symbols or (len(related_symbols) == 1 and related_symbols[0] == symbol):
                        related_symbols = [symbol]
                    else:
                        # Add the main symbol if not already in the list
                        if symbol not in related_symbols:
                            related_symbols.append(symbol)
                    
                    # Remove duplicates and convert to string
                    related_symbols = list(set(related_symbols))
                    related_symbols_str = ','.join(related_symbols)
                    
                    articles.append({
                        'symbol': symbol,
                        'title': title,
                        'publisher': publisher,
                        'link': link,
                        'published_date': published_date,
                        'type': content_type,
                        'related_symbols': related_symbols_str,
                        'preview_text': summary,
                        'thumbnail': thumbnail
                    })
                    
                except Exception as e:
                    logger.debug(f"[{thread_name}] Error processing article for {symbol}: {str(e)}")
                    continue
                    
            logger.debug(f"[{thread_name}] Processed {len(articles)} valid articles for {symbol}")
            
            # Random delay to avoid rate limiting
            time.sleep(delay)
            return articles
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = (attempt + 1) * random.uniform(2, 4)
                logger.warning(f"[{thread_name}] Attempt {attempt + 1} failed for {symbol}: {str(e)}. Retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                logger.warning(f"[{thread_name}] All {MAX_RETRIES} attempts failed for {symbol}: {str(e)}")
                return []
    
    return []


def get_existing_news(symbol, days=DEFAULT_DAYS):
    """Retrieve existing news for a symbol within the specified days"""
    session = get_thread_session()
    cutoff_date = datetime.now() - timedelta(days=days)
    
    try:
        articles = session.query(NewsArticle).filter(
            NewsArticle.symbol == symbol,
            NewsArticle.published_date >= cutoff_date
        ).order_by(NewsArticle.published_date.desc()).all()
        
        return articles
    except Exception as e:
        logger.error(f"Error retrieving existing news for {symbol}: {str(e)}")
        return []


def remove_oldest_articles(symbol, current_count, target_count):
    """Remove oldest articles until count is reduced to target"""
    if current_count <= target_count:
        return 0
    
    to_remove = current_count - target_count
    session = get_thread_session()
    
    try:
        # Get the IDs of the oldest articles to remove
        cutoff_date = datetime.now() - timedelta(days=DEFAULT_DAYS)
        oldest_articles = session.query(NewsArticle).filter(
            NewsArticle.symbol == symbol,
            NewsArticle.published_date >= cutoff_date
        ).order_by(NewsArticle.published_date.asc()).limit(to_remove).all()
        
        # Delete these articles
        for article in oldest_articles:
            session.delete(article)
        
        session.commit()
        return len(oldest_articles)
    except Exception as e:
        logger.error(f"Error removing oldest articles for {symbol}: {str(e)}")
        session.rollback()
        return 0


def add_new_articles(symbol, fetched_articles, existing_articles, min_count):
    """Add new articles that are not already in the database"""
    if not fetched_articles:
        return 0
    
    # Create a set of existing article identifiers (symbol+title+date) for quick lookup
    existing_identifiers = {
        f"{article.symbol}|{article.title}|{article.published_date}"
        for article in existing_articles
    }
    
    session = get_thread_session()
    added_count = 0
    
    try:
        # Sort fetched articles by date (newest first)
        sorted_articles = sorted(
            fetched_articles, 
            key=lambda x: x['published_date'], 
            reverse=True
        )
        
        current_count = len(existing_articles)
        
        for article_data in sorted_articles:
            # Check if we already have enough articles
            if current_count >= min_count and added_count > 0:
                break
                
            # Check if this article already exists
            article_id = f"{article_data['symbol']}|{article_data['title']}|{article_data['published_date']}"
            if article_id in existing_identifiers:
                continue
            
            # Add the new article
            try:
                article = NewsArticle(**article_data)
                session.add(article)
                added_count += 1
                current_count += 1
                existing_identifiers.add(article_id)
            except Exception as e:
                logger.error(f"Error adding article for {symbol}: {str(e)}")
                continue
        
        if added_count > 0:
            session.commit()
            
        return added_count
    except Exception as e:
        logger.error(f"Error adding new articles for {symbol}: {str(e)}")
        session.rollback()
        return 0


def maintain_news_for_symbol(symbol, days=DEFAULT_DAYS):
    """Maintain 10-15 news articles for a symbol from the last 30 days"""
    try:
        # Step 1: Retrieve existing news within the last X days
        existing_articles = get_existing_news(symbol, days)
        existing_count = len(existing_articles)
        
        logger.debug(f"Found {existing_count} existing articles for {symbol} within {days} days")
        
        # Step 2: Fetch latest news from external source
        fetched_articles = fetch_news_for_symbol(symbol, days)
        fetched_count = len(fetched_articles)
        
        logger.debug(f"Fetched {fetched_count} articles for {symbol} from API")
        
        removed_count = 0
        added_count = 0
        
        # Step 3: Decide on action based on count
        
        # If more than MAX_ARTICLES (15) articles exist, remove oldest
        if existing_count > MAX_ARTICLES:
            removed_count = remove_oldest_articles(symbol, existing_count, MAX_ARTICLES)
            existing_count -= removed_count
            logger.debug(f"Removed {removed_count} oldest articles for {symbol}")
        
        # If fewer than MIN_ARTICLES (10) articles exist, add new ones
        if existing_count < MIN_ARTICLES:
            added_count = add_new_articles(symbol, fetched_articles, existing_articles, MIN_ARTICLES)
            logger.debug(f"Added {added_count} new articles for {symbol} to reach minimum")
        
        # If between 10-15 articles exist, add any new articles that don't exist
        elif existing_count >= MIN_ARTICLES and existing_count <= MAX_ARTICLES:
            added_count = add_new_articles(symbol, fetched_articles, existing_articles, existing_count)
            logger.debug(f"Added {added_count} new unique articles for {symbol}")
        
        # Update stats
        with stats.lock:
            if added_count > 0 or removed_count > 0:
                stats.symbols_maintained += 1
            stats.articles_added += added_count
            stats.articles_removed += removed_count
        
        # Return summary of actions
        return {
            'symbol': symbol,
            'existing_count': existing_count,
            'fetched_count': fetched_count,
            'removed_count': removed_count,
            'added_count': added_count,
            'final_count': existing_count - removed_count + added_count
        }
        
    except Exception as e:
        logger.error(f"Error maintaining news for {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'error': str(e),
            'existing_count': 0,
            'fetched_count': 0,
            'removed_count': 0,
            'added_count': 0,
            'final_count': 0
        }


def process_symbol_batch(symbols, days=DEFAULT_DAYS):
    """Process a batch of symbols"""
    thread_name = threading.current_thread().name
    logger.debug(f"[{thread_name}] Starting batch processing of {len(symbols)} symbols")
    
    for symbol in symbols:
        try:
            result = maintain_news_for_symbol(symbol, days)
            
            with stats.lock:
                stats.processed += 1
                
                # Print progress less frequently to avoid cluttering logs
                if stats.processed % 20 == 0 or stats.processed == stats.total_symbols:
                    elapsed = (datetime.now() - stats.start_time).total_seconds() / 60
                    symbols_per_minute = stats.processed / elapsed if elapsed > 0 else 0
                    estimated_remaining = (stats.total_symbols - stats.processed) / symbols_per_minute if symbols_per_minute > 0 else 0
                    
                    logger.info(f"Progress: {stats.processed}/{stats.total_symbols} ({(stats.processed/stats.total_symbols)*100:.1f}%)")
                    logger.info(f"Symbols maintained: {stats.symbols_maintained}, Added: {stats.articles_added}, Removed: {stats.articles_removed}")
                    if elapsed > 0 and stats.processed < stats.total_symbols:
                        logger.info(f"Rate: {symbols_per_minute:.1f} symbols/min, Est. remaining: {estimated_remaining:.1f} minutes")
                        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            with stats.lock:
                stats.processed += 1
    
    logger.debug(f"[{thread_name}] Completed batch processing")


def get_database_stats():
    """Get database statistics for news articles"""
    session = Session()
    try:
        stats = {}
        
        # Basic counts
        stats['total_articles'] = session.query(NewsArticle).count()
        stats['unique_symbols'] = session.query(func.count(func.distinct(NewsArticle.symbol))).scalar()
        
        # Date range
        if stats['total_articles'] > 0:
            latest = session.query(func.max(NewsArticle.published_date)).scalar()
            earliest = session.query(func.min(NewsArticle.published_date)).scalar()
            stats['date_range'] = f"{earliest} to {latest}"
        else:
            stats['date_range'] = "No articles"
        
        # Symbol article count distribution
        symbol_counts = session.query(
            NewsArticle.symbol,
            func.count(NewsArticle.id).label('count')
        ).group_by(NewsArticle.symbol).all()
        
        count_distribution = {
            'less_than_10': 0,
            '10_to_15': 0,
            'more_than_15': 0
        }
        
        for _, count in symbol_counts:
            if count < MIN_ARTICLES:
                count_distribution['less_than_10'] += 1
            elif MIN_ARTICLES <= count <= MAX_ARTICLES:
                count_distribution['10_to_15'] += 1
            else:
                count_distribution['more_than_15'] += 1
        
        stats['count_distribution'] = count_distribution
        
        # Top symbols by article count
        stats['top_symbols'] = session.query(
            NewsArticle.symbol, 
            func.count(NewsArticle.id).label('count')
        ).group_by(NewsArticle.symbol).order_by(
            func.count(NewsArticle.id).desc()
        ).limit(10).all()
        
        # Articles by type
        stats['types'] = session.query(
            NewsArticle.type,
            func.count(NewsArticle.id).label('count')
        ).group_by(NewsArticle.type).order_by(
            func.count(NewsArticle.id).desc()
        ).all()
        
        # Recent articles
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        stats['today_count'] = session.query(NewsArticle).filter(
            NewsArticle.published_date >= today
        ).count()
        
        week_ago = today - timedelta(days=7)
        stats['week_count'] = session.query(NewsArticle).filter(
            NewsArticle.published_date >= week_ago
        ).count()
        
        return stats
    finally:
        session.close()


def log_database_stats(db_stats):
    """Log database statistics to console"""
    logger.info("\nDatabase Statistics:")
    logger.info(f"Total articles: {db_stats['total_articles']}")
    logger.info(f"Unique symbols: {db_stats['unique_symbols']}")
    logger.info(f"Date range: {db_stats['date_range']}")
    logger.info(f"Articles today: {db_stats['today_count']}")
    logger.info(f"Articles in past week: {db_stats['week_count']}")
    
    dist = db_stats['count_distribution']
    logger.info("\nSymbol article count distribution:")
    logger.info(f"- Symbols with <{MIN_ARTICLES} articles: {dist['less_than_10']}")
    logger.info(f"- Symbols with {MIN_ARTICLES}-{MAX_ARTICLES} articles: {dist['10_to_15']}")
    logger.info(f"- Symbols with >{MAX_ARTICLES} articles: {dist['more_than_15']}")
    
    logger.info("\nTop 10 symbols by article count:")
    for symbol, count in db_stats['top_symbols']:
        logger.info(f"- {symbol}: {count} articles")
    
    logger.info("\nArticles by type:")
    for type_name, count in db_stats['types']:
        logger.info(f"- {type_name}: {count} articles")


def main():
    global stats, engine, Session
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Maintain news articles for stock symbols (10-15 articles per symbol)')
    parser = add_environment_args(parser)
    parser.add_argument('--symbol', '-s', type=str, help='Process only this symbol (for testing)')
    parser.add_argument('--days', '-d', type=int, default=DEFAULT_DAYS, help=f'Number of days of news to consider (default: {DEFAULT_DAYS})')
    parser.add_argument('--workers', '-w', type=int, default=NUM_WORKERS, help=f'Number of worker threads (default: {NUM_WORKERS})')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--input', '-i', type=str, default=str(INPUT_FILE), help=f'Input file with symbols (default: {INPUT_FILE})')
    parser.add_argument('--stats', action='store_true', help='Show database statistics and exit')
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Setup database connection based on environment
    engine = get_database_engine(args.env)
    Session = get_session_maker(args.env)
    
    env_name = "PRODUCTION" if args.env == "prod" else "DEVELOPMENT"
    logger.info(f"Starting news collector using {env_name} database")
    
    # Show stats if requested
    if args.stats:
        db_stats = get_database_stats()
        log_database_stats(db_stats)
        return
    
    # Single symbol mode
    if args.symbol:
        symbol = args.symbol.upper()
        logger.info(f"Processing single symbol: {symbol}")
        stats = Stats(1)  # Need to initialize stats for single symbol mode
        result = maintain_news_for_symbol(symbol, days=args.days)
        
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Existing articles: {result['existing_count']}")
        logger.info(f"Fetched articles: {result['fetched_count']}")
        logger.info(f"Articles removed: {result['removed_count']}")
        logger.info(f"Articles added: {result['added_count']}")
        logger.info(f"Final article count: {result['final_count']}")
        return
    
    # Normal operation - verify input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Error: Input file {input_path} not found!")
        return
    
    # Read symbols
    try:
        with open(input_path, 'r') as f:
            content = f.read().strip()
            symbols = [symbol.strip() for symbol in content.split(',') if symbol.strip()]
    except Exception as e:
        logger.error(f"Error reading input file: {str(e)}")
        return
    
    if not symbols:
        logger.error("No symbols found in input file!")
        return
    
    # Initialize stats
    total_symbols = len(symbols)
    stats = Stats(total_symbols)
    
    logger.info(f"Starting news maintenance for {total_symbols} symbols using {args.workers} workers")
    logger.info(f"Maintaining {MIN_ARTICLES}-{MAX_ARTICLES} articles per symbol from the past {args.days} days")
    
    # Split symbols into batches for workers
    batch_size = math.ceil(len(symbols) / args.workers)
    symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    try:
        # Process batches with thread pool
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_symbol_batch, batch, args.days) for batch in symbol_batches]
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in worker thread: {str(e)}")
    
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user!")
    finally:
        # Print final statistics
        elapsed_time = (datetime.now() - stats.start_time).total_seconds() / 60
        
        logger.info("\nProcessing Summary:")
        logger.info(f"Total symbols processed: {stats.processed}/{total_symbols}")
        logger.info(f"Symbols maintained: {stats.symbols_maintained}")
        logger.info(f"Articles added: {stats.articles_added}")
        logger.info(f"Articles removed: {stats.articles_removed}")
        logger.info(f"Total time: {elapsed_time:.1f} minutes")
        
        if elapsed_time > 0:
            logger.info(f"Processing rate: {stats.processed/elapsed_time:.1f} symbols per minute")
        
        # Show database statistics
        try:
            db_stats = get_database_stats()
            log_database_stats(db_stats)
        except Exception as e:
            logger.error(f"Error getting database statistics: {str(e)}")


if __name__ == "__main__":
    main()
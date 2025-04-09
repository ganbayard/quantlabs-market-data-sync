import json
import time
import yfinance as yf
from urllib.parse import urlparse

# Configuration
INPUT_FILE = "symbols/all_us_stocks_norgatedata.txt"
OUTPUT_JSON = "stock_symbols_logo.json"

def get_domain_yfinance(symbol):
    """Fetch company domain using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Try different possible keys for website
        website = info.get('website', '') or info.get('Website', '') or info.get('homepage_url', '')
        
        if website:
            # Clean up the domain
            parsed = urlparse(website)
            domain = parsed.netloc
            if not domain:
                domain = parsed.path
            # Remove www. if present
            domain = domain.replace('www.', '')
            return domain
            
        return ""
    except Exception as e:
        print(f"yfinance Error for {symbol}: {e}")
        return ""

def generate_logo_url(domain):
    """Generate Clearbit logo URL from domain."""
    if domain:
        # Ensure we're using just the domain without http/https
        return f"https://logo.clearbit.com/{domain}"
    return ""

def main():
    try:
        with open(INPUT_FILE, "r") as f:
            symbols = [s.strip() for s in f.read().split(",")]
        
        symbol_to_logo = {}
        total_symbols = len(symbols)
        
        for idx, symbol in enumerate(symbols, 1):
            try:
                print(f"Processing {idx}/{total_symbols}: {symbol}")
                
                domain = get_domain_yfinance(symbol)
                logo_url = generate_logo_url(domain)
                
                if logo_url:
                    symbol_to_logo[symbol] = logo_url
                    print(f"✓ Found logo for {symbol}: {logo_url}")
                else:
                    symbol_to_logo[symbol] = ""
                    print(f"✗ No logo found for {symbol}")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                symbol_to_logo[symbol] = ""
                continue
        
        with open(OUTPUT_JSON, "w") as f:
            json.dump(symbol_to_logo, f, indent=2)
        
        print(f"\nSummary:")
        print(f"Total symbols processed: {total_symbols}")
        print(f"Logos found: {sum(1 for url in symbol_to_logo.values() if url)}")
        print(f"Logos not found: {sum(1 for url in symbol_to_logo.values() if not url)}")
        print(f"Results saved to {OUTPUT_JSON}")
        
    except Exception as e:
        print(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
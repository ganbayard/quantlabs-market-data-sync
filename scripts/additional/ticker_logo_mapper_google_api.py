import json
import time
import yfinance as yf
from urllib.parse import urlparse, quote
import requests

class LogoFetcher:
    @staticmethod
    def get_google_favicon(domain):
        encoded_url = quote(f"http://{domain}")
        return f"https://t1.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url={encoded_url}&size=64"

    @staticmethod
    def verify_logo_url(url):
        try:
            response = requests.head(url, timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_logo(self, domain):
        if not domain:
            return ""
        
        logo_url = self.get_google_favicon(domain)
        if self.verify_logo_url(logo_url):
            print(f"✓ Found favicon for domain: {domain}")
            return logo_url
        return ""

def get_domain_yfinance(symbol):
    """Fetch company domain using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        
        try:
            info = ticker.info
        except (ValueError, Exception) as e:
            print(f"Could not get info for {symbol}: {str(e)}")
            return ""
        
        website = info.get('website', '') or info.get('Website', '') or info.get('homepage_url', '')
        
        if website:
            # Clean up the domain
            parsed = urlparse(website)
            domain = parsed.netloc or parsed.path
            # Remove www. and any trailing slashes
            domain = domain.replace('www.', '').rstrip('/')
            return domain
            
        return ""
    except Exception as e:
        print(f"yfinance Error for {symbol}: {str(e)}")
        return ""

def main():
    INPUT_FILE = "symbols/all_us_stocks_norgatedata.txt"
    OUTPUT_JSON = "stock_symbols_logo_google_api.json"
    logo_fetcher = LogoFetcher()

    try:
        # Read symbols from file
        with open(INPUT_FILE, "r") as f:
            symbols = [s.strip() for s in f.read().split(",")]
        
        symbol_to_logo = {}
        total_symbols = len(symbols)
        
        for idx, symbol in enumerate(symbols, 1):
            try:
                print(f"\nProcessing {idx}/{total_symbols}: {symbol}")
                
                domain = get_domain_yfinance(symbol)
                if domain:
                    logo_url = logo_fetcher.get_logo(domain)
                    if logo_url:
                        symbol_to_logo[symbol] = {
                            'logo_url': logo_url,
                            'domain': domain
                        }
                        print(f"✓ Success: {symbol}")
                    else:
                        symbol_to_logo[symbol] = {
                            'logo_url': '',
                            'domain': domain
                        }
                        print(f"✗ No favicon found for {symbol}")
                else:
                    symbol_to_logo[symbol] = {
                        'logo_url': '',
                        'domain': ''
                    }
                    print(f"✗ No domain found for {symbol}")
                
                time.sleep(0.2)
                
                if idx % 100 == 0:
                    with open(OUTPUT_JSON, "w") as f:
                        json.dump(symbol_to_logo, f, indent=2)
                    print(f"Progress saved at {idx} symbols")
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                symbol_to_logo[symbol] = {
                    'logo_url': '',
                    'domain': ''
                }
                continue
        
        with open(OUTPUT_JSON, "w") as f:
            json.dump(symbol_to_logo, f, indent=2)
        
        logos_found = sum(1 for data in symbol_to_logo.values() if data['logo_url'])
        domains_found = sum(1 for data in symbol_to_logo.values() if data['domain'])
        
        print(f"\nSummary:")
        print(f"Total symbols processed: {total_symbols}")
        print(f"Domains found: {domains_found}")
        print(f"Logos found: {logos_found}")
        print(f"Results saved to {OUTPUT_JSON}")
        
    except Exception as e:
        print(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
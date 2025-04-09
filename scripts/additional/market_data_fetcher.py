import requests
import json
from datetime import datetime, timezone
import time
from typing import Dict, Any
import yfinance as yf

class MarketDataFetcher:
    def __init__(self):
        pass
        
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch stock data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            info = ticker.info
            
            if not hist.empty and 'previousClose' in info:
                current_price = hist['Close'].iloc[-1]
                prev_close = info['previousClose']
                change_percent = ((current_price - prev_close) / prev_close) * 100
                
                return {
                    'price': round(current_price, 2),
                    'change_percent': round(change_percent, 2)
                }
            return {'price': 0, 'change_percent': 0}
            
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {str(e)}")
            return {'price': 0, 'change_percent': 0}

    def get_crypto_data(self) -> Dict[str, Any]:
        try:
            btc_url = 'https://api.coingecko.com/api/v3/coins/bitcoin?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false'
            btc_response = requests.get(btc_url)
            btc_response.raise_for_status()
            btc_data = btc_response.json()
            
            time.sleep(1)
            
            # Get total market cap
            market_url = 'https://api.coingecko.com/api/v3/global'
            market_response = requests.get(market_url)
            market_response.raise_for_status()
            market_data = market_response.json()
            
            btc_price = btc_data['market_data']['current_price']['usd']
            btc_change = btc_data['market_data']['price_change_percentage_24h']
            
            return {
                'btc': {
                    'price': btc_price,
                    'change_percent': btc_change
                },
                'total_market_cap': {
                    'price': market_data['data']['total_market_cap']['usd'],
                    'change_percent': market_data['data']['market_cap_change_percentage_24h_usd']
                }
            }
        except Exception as e:
            print(f"Error fetching crypto data: {str(e)}")
            return {
                'btc': {'price': 0, 'change_percent': 0},
                'total_market_cap': {'price': 0, 'change_percent': 0}
            }

    def get_forex_data(self) -> Dict[str, Any]:
        """Fetch forex data using yfinance"""
        try:
            eurusd = yf.Ticker("EURUSD=X")
            hist = eurusd.history(period="1d")
            info = eurusd.info
            
            if not hist.empty and 'previousClose' in info:
                current_price = hist['Close'].iloc[-1]
                prev_close = info['previousClose']
                change_percent = ((current_price - prev_close) / prev_close) * 100
                
                return {
                    'price': round(current_price, 4),
                    'change_percent': round(change_percent, 2)
                }
            return {'price': 0, 'change_percent': 0}
            
        except Exception as e:
            print(f"Error fetching forex data: {str(e)}")
            return {'price': 0, 'change_percent': 0}

    def get_gold_data(self) -> Dict[str, Any]:
        """Fetch gold data using yfinance"""
        try:
            gold = yf.Ticker("GC=F")
            hist = gold.history(period="1d")
            info = gold.info
            
            if not hist.empty and 'previousClose' in info:
                current_price = hist['Close'].iloc[-1]
                prev_close = info['previousClose']
                change_percent = ((current_price - prev_close) / prev_close) * 100
                
                return {
                    'price': round(current_price, 2),
                    'change_percent': round(change_percent, 2)
                }
            return {'price': 0, 'change_percent': 0}
            
        except Exception as e:
            print(f"Error fetching gold data: {str(e)}")
            return {'price': 0, 'change_percent': 0}

    def fetch_all_market_data(self) -> Dict[str, Any]:
        try:
            # Fetch all data with proper symbols
            sp500_data = self.get_stock_data("^GSPC")
            time.sleep(1)
            
            nasdaq_data = self.get_stock_data("^IXIC")
            time.sleep(1)
            
            crypto_data = self.get_crypto_data()
            time.sleep(1)
            
            forex_data = self.get_forex_data()
            time.sleep(1)
            
            gold_data = self.get_gold_data()
            time.sleep(1)
            
            vix_data = self.get_stock_data("^VIX")

            # Validate data before returning
            def validate_data(data: Dict[str, float]) -> int:
                return 1 if (data['price'] != 0 or data['change_percent'] != 0) else 0

            market_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': {
                    'S&P500': {
                        'ticker': 'SPX',
                        'price': sp500_data['price'],
                        'change_percent': sp500_data['change_percent'],
                        'currency': 'USD',
                        'valid': validate_data(sp500_data)
                    },
                    'NASDAQ': {
                        'ticker': 'IXIC',
                        'price': nasdaq_data['price'],
                        'change_percent': nasdaq_data['change_percent'],
                        'currency': 'USD',
                        'valid': validate_data(nasdaq_data)
                    },
                    'BTC': {
                        'ticker': 'BTC-USD',
                        'price': crypto_data['btc']['price'],
                        'change_percent': crypto_data['btc']['change_percent'],
                        'currency': 'USD',
                        'valid': validate_data(crypto_data['btc'])
                    },
                    'GOLD': {
                        'ticker': 'XAUUSD',
                        'price': gold_data['price'],
                        'change_percent': gold_data['change_percent'],
                        'currency': 'USD',
                        'valid': validate_data(gold_data)
                    },
                    'EUR/USD': {
                        'ticker': 'EURUSD',
                        'price': forex_data['price'],
                        'change_percent': forex_data['change_percent'],
                        'currency': 'USD',
                        'valid': validate_data(forex_data)
                    },
                    'VIX': {
                        'ticker': 'VIX',
                        'price': vix_data['price'],
                        'change_percent': vix_data['change_percent'],
                        'currency': 'USD',
                        'valid': validate_data(vix_data)
                    },
                    'CRYPTO_TMC': {
                        'ticker': 'TOTAL',
                        'price': crypto_data['total_market_cap']['price'],
                        'change_percent': crypto_data['total_market_cap']['change_percent'],
                        'currency': 'USD',
                        'valid': validate_data(crypto_data['total_market_cap'])
                    }
                }
            }
            
            # Print validation results
            for asset, data in market_data['data'].items():
                if data['valid'] == 0:
                    print(f"Warning: {asset} data may be invalid (returned zeros)")
            
            return market_data
            
        except Exception as e:
            print(f"Error in fetch_all_market_data: {str(e)}")
            return {'error': str(e)}

def save_to_json(data: Dict[str, Any], filename: str = 'market_data.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")

if __name__ == "__main__":
    fetcher = MarketDataFetcher()
    market_data = fetcher.fetch_all_market_data()
    print(json.dumps(market_data, indent=2))
    save_to_json(market_data)
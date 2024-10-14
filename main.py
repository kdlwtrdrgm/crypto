from crypto_analyzer import CryptoAnalyzer
from config import DEFAULT_SYMBOL
import argparse

def main():
    parser = argparse.ArgumentParser(description='Crypto Analyzer')
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, help='Trading symbol (e.g., BTC/USDT)')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange to use (e.g., binance, coinbase)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe for data (e.g., 1h, 4h, 1d)')
    args = parser.parse_args()

    analyzer = CryptoAnalyzer(args.exchange, args.symbol, args.timeframe)
    analyzer.analyze(args.start_date)
    
    report = analyzer.generate_report()
    print(report)

if __name__ == '__main__':
    main()
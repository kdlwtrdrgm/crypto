import ccxt
import pandas as pd
from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, DEFAULT_TIMEFRAME

class DataCollector:
    def __init__(self, exchange_id='binance', symbol=None, timeframe=DEFAULT_TIMEFRAME):
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET_KEY,
        })
        self.symbol = symbol
        self.timeframe = timeframe

    def fetch_ohlcv(self, start_date):
        since = self.exchange.parse8601(start_date)
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def fetch_order_book(self, limit=100):
        order_book = self.exchange.fetch_order_book(self.symbol, limit)
        return order_book

    def fetch_trades(self, limit=100):
        trades = self.exchange.fetch_trades(self.symbol, limit=limit)
        return trades
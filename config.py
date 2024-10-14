import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# Database configuration
DB_URI = os.getenv('DB_URI', 'sqlite:///crypto_analyzer.db')

# Other configurations
DEFAULT_TIMEFRAME = '1h'
DEFAULT_SYMBOL = 'BTC/USDT'
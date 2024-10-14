import talib
import numpy as np
import pandas as pd

class TechnicalAnalysis:
    @staticmethod
    def calculate_indicators(df):
        df['MA20'] = talib.SMA(df['close'], timeperiod=20)
        df['MA50'] = talib.SMA(df['close'], timeperiod=50)
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        return df

    @staticmethod
    def identify_patterns(df):
        patterns = {
            'CDL2CROWS': talib.CDL2CROWS,
            'CDL3BLACKCROWS': talib.CDL3BLACKCROWS,
            'CDLENGULFING': talib.CDLENGULFING,
            'CDLHAMMER': talib.CDLHAMMER,
            'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
        }
        
        for pattern_name, pattern_func in patterns.items():
            df[pattern_name] = pattern_func(df['open'], df['high'], df['low'], df['close'])
        
        return df

    @staticmethod
    def calculate_support_resistance(df, window=14):
        df['support'] = df['low'].rolling(window=window).min()
        df['resistance'] = df['high'].rolling(window=window).max()
        return df

    @staticmethod
    def calculate_volatility(df, window=14):
        df['volatility'] = df['close'].pct_change().rolling(window=window).std() * np.sqrt(252)
        return df
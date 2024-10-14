import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import talib
from data_collector import DataCollector
from technical_analysis import TechnicalAnalysis
from machine_learning import MachineLearning

class CryptoAnalyzer:
    def __init__(self, exchange_id, symbol, timeframe='1h'):
        self.data_collector = DataCollector(exchange_id, symbol, timeframe)
        self.technical_analysis = TechnicalAnalysis()
        self.machine_learning = MachineLearning()
        self.symbol = symbol
        self.data = None

    def fetch_data(self, start_date):
        self.data = self.data_collector.fetch_ohlcv(start_date)

    def analyze(self, start_date):
        self.fetch_data(start_date)
        self.data = self.technical_analysis.calculate_indicators(self.data)
        self.data = self.technical_analysis.identify_patterns(self.data)
        self.plot_data()
        self.predict_prices()

    def plot_data(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20), sharex=True)

        # Price and MA
        ax1.plot(self.data.index, self.data['close'], label='Close')
        ax1.plot(self.data.index, self.data['MA20'], label='MA20')
        ax1.plot(self.data.index, self.data['MA50'], label='MA50')
        ax1.set_title(f'{self.symbol} Price and Moving Averages')
        ax1.legend()

        # RSI
        ax2.plot(self.data.index, self.data['RSI'], label='RSI')
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.set_title('RSI')
        ax2.legend()

        # MACD
        ax3.plot(self.data.index, self.data['MACD'], label='MACD')
        ax3.plot(self.data.index, self.data['MACD_signal'], label='Signal')
        ax3.bar(self.data.index, self.data['MACD_hist'], label='Histogram')
        ax3.set_title('MACD')
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def predict_prices(self):
        X, y = self.machine_learning.prepare_data(self.data)
        self.machine_learning.build_model((X.shape[1], 1))
        self.machine_learning.train_model(X, y)
        predictions = self.machine_learning.predict(X)

        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index[-len(predictions):], self.data['close'].values[-len(predictions):], label='Actual')
        plt.plot(self.data.index[-len(predictions):], predictions, label='Predicted')
        plt.title(f'{self.symbol} Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def generate_report(self):
        report = f"Crypto Analysis Report for {self.symbol}\n"
        report += f"Date Range: {self.data.index[0]} to {self.data.index[-1]}\n\n"

        report += "Current Indicators:\n"
        report += f"Price: {self.data['close'].iloc[-1]:.2f}\n"
        report += f"RSI: {self.data['RSI'].iloc[-1]:.2f}\n"
        report += f"MACD: {self.data['MACD'].iloc[-1]:.2f}\n"
        report += f"Signal: {self.data['MACD_signal'].iloc[-1]:.2f}\n"

        report += "\nRecent Patterns:\n"
        for pattern in ['CDL2CROWS', 'CDL3BLACKCROWS', 'CDLENGULFING', 'CDLHAMMER', 'CDLMORNINGSTAR']:
            if self.data[pattern].iloc[-1] != 0:
                report += f"{pattern} detected\n"

        return report
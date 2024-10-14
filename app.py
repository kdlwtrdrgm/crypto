from flask import Flask, render_template, request
from crypto_analyzer import CryptoAnalyzer
from config import DEFAULT_SYMBOL
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form.get('symbol', DEFAULT_SYMBOL)
        start_date = request.form.get('start_date', '2023-01-01')
        exchange = request.form.get('exchange', 'binance')
        timeframe = request.form.get('timeframe', '1h')

        analyzer = CryptoAnalyzer(exchange, symbol, timeframe)
        analyzer.analyze(start_date)
        report = analyzer.generate_report()

        # Generate plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20), sharex=True)

        # Price and MA
        ax1.plot(analyzer.data.index, analyzer.data['close'], label='Close')
        ax1.plot(analyzer.data.index, analyzer.data['MA20'], label='MA20')
        ax1.plot(analyzer.data.index, analyzer.data['MA50'], label='MA50')
        ax1.set_title(f'{symbol} Price and Moving Averages')
        ax1.legend()

        # RSI
        ax2.plot(analyzer.data.index, analyzer.data['RSI'], label='RSI')
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.set_title('RSI')
        ax2.legend()

        # MACD
        ax3.plot(analyzer.data.index, analyzer.data['MACD'], label='MACD')
        ax3.plot(analyzer.data.index, analyzer.data['MACD_signal'], label='Signal')
        ax3.bar(analyzer.data.index, analyzer.data['MACD_hist'], label='Histogram')
        ax3.set_title('MACD')
        ax3.legend()

        plt.tight_layout()

        # Save plot to a base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')

        return render_template('result.html', report=report, plot_url=plot_url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
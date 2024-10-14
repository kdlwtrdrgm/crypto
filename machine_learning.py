from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

class MachineLearning:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, df, look_back=60):
        data = df[['close', 'volume', 'RSI', 'MACD']].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, X, y, epochs=100, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(np.column_stack((predictions, np.zeros((len(predictions), 3))))[:, 0]

    def evaluate_model(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        return {'MSE': mse, 'RMSE': rmse}
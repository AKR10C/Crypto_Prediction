import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import ccxt
from sklearn.preprocessing import MinMaxScaler

# 1. Get historical price data from Binance
def get_data(symbol='BTC/USDT', limit=100):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df['close'].values.reshape(-1, 1)

# 2. Preprocess data for training
def preprocess(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(10, len(scaled)):
        X.append(scaled[i-10:i, 0])
        y.append(scaled[i, 0])
    return np.array(X).reshape(-1, 10, 1), np.array(y), scaler

# 3. Build the BiLSTM model
def build_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(10, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 4. Train and predict for a single coin
def train_predict(symbol='BTC/USDT'):
    print(f"\n🔄 Predicting for {symbol}...")
    try:
        data = get_data(symbol)
        X, y, scaler = preprocess(data)
        model = build_model()
        model.fit(X, y, epochs=5, batch_size=8, verbose=0)

        last_sequence = X[-1].reshape(1, 10, 1)
        prediction = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform([[prediction[0][0]]])
        print(f"💰 Predicted next {symbol} price: ${predicted_price[0][0]:.2f}")
    except Exception as e:
        print(f"❌ Failed for {symbol}: {e}")

# 5. Main function for multiple coins
if __name__ == "__main__":
    coin_list = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'DOGE/USDT', 'SOL/USDT']
    for coin in coin_list:
        train_predict(coin)

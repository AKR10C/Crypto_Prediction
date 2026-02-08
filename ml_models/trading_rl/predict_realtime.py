import numpy as np
import ccxt
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# 🧠 ACTIONS
ACTIONS = ['Sell', 'Hold', 'Buy']

# 📌 Coins to predict
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

# Load models
models = {}
for symbol in symbols:
    model_path = f"trained_models/dqn_model_{symbol.replace('/', '_')}.h5"
    if os.path.exists(model_path):
        try:
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss=MeanSquaredError())
            models[symbol] = model
            print(f"✅ Loaded model for {symbol}")
        except Exception as e:
            print(f"❌ Error loading model for {symbol}: {e}")
    else:
        print(f"❌ Model not found for {symbol}")

# Set up Binance via ccxt
exchange = ccxt.binance()

# 📈 Fetch live closing prices
def get_latest_closes(symbol, window=10):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=window)
        closes = [x[4] for x in ohlcv]
        return np.array(closes).reshape(1, -1)
    except Exception as e:
        print(f"⚠️ Error fetching price for {symbol}: {e}")
        return None

# 🔮 Get predictions for all symbols
def get_trading_action():
    results = []

    for symbol in symbols:
        model = models.get(symbol)
        if not model:
            results.append({"symbol": symbol, "error": "Model not loaded"})
            continue

        state = get_latest_closes(symbol)
        if state is None or state.shape[1] != 10:
            results.append({"symbol": symbol, "error": "Bad or insufficient price data"})
            continue

        action_probs = model.predict(state, verbose=0)
        action_index = int(np.argmax(action_probs))

        results.append({
            "symbol": symbol,
            "current_price": float(state[0][-1]),
            "predicted_price": float(state[0][-1] + action_probs[0][action_index]),
            "action": ACTIONS[action_index]
        })

    return results

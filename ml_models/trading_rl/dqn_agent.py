import numpy as np
import ccxt
import os

from dqn_model import DQNAgent

# 📌 List of coins you want to train on
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']

# Binance exchange setup
exchange = ccxt.binance()
data_limit = 500

# Create models folder if not exists
os.makedirs("trained_models", exist_ok=True)

# ✅ 1. Fetch historical data
def fetch_data(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=data_limit)
        close_prices = [x[4] for x in ohlcv]
        return np.array(close_prices)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# ✅ 2. Convert prices to env (state, reward, next_state, done)
def create_env(data):
    env = []
    for i in range(len(data) - 10):
        state = data[i:i+10]
        next_state = data[i+1:i+11]
        reward = next_state[-1] - state[-1]  # simple reward
        done = i == len(data) - 11
        env.append((state, reward, next_state, done))
    return env

# ✅ 3. Training function per coin
def train_rl_for_symbol(symbol):
    print(f"🚀 Training for {symbol}")
    prices = fetch_data(symbol)
    if prices is None:
        return

    env = create_env(prices)

    state_size = 10
    action_size = 3  # sell, hold, buy

    agent = DQNAgent(state_size, action_size)

    for episode in range(1, 51):
        total_profit = 0
        for state, reward, next_state, done in env:
            state = np.reshape(state, [1, state_size])
            next_state = np.reshape(next_state, [1, state_size])
            action = agent.act(state)

            # Calculate profit for buy/sell
            if action == 2:  # buy
                total_profit += reward
            elif action == 0:  # sell
                total_profit -= reward

            agent.remember(state, action, reward, next_state, done)

        agent.replay(batch_size=32)
        print(f"📘 [{symbol}] Episode {episode}/50 - Total Profit: ${total_profit:.2f}")

    model_name = f"trained_models/dqn_model_{symbol.replace('/', '_')}.h5"
    agent.save(model_name)
    print(f"✅ Model saved: {model_name}\n")

# ✅ 4. Loop over all coins
if __name__ == "__main__":
    for symbol in symbols:
        train_rl_for_symbol(symbol)
 
import numpy as np
import ccxt
import pandas as pd

# Set up Binance connection
exchange = ccxt.binance()

# Fetch historical data
def fetch_historical_prices(symbol="BTC/USDT", timeframe="1h", limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        closes = [candle[4] for candle in ohlcv]
        return np.array(closes)
    except Exception as e:
        print(f"⚠️ Error fetching historical prices for {symbol}: {e}")
        return None

# Calculate log returns
def calculate_log_returns(prices):
    return np.diff(np.log(prices))

# Monte Carlo simulation for VaR
def monte_carlo_var(log_returns, initial_investment=1000, num_simulations=1000, time_horizon=1, confidence_level=0.95):
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)

    # Simulate price paths
    simulated_end_values = []
    for _ in range(num_simulations):
        random_returns = np.random.normal(mu, sigma, time_horizon)
        cumulative_return = np.sum(random_returns)
        simulated_price = initial_investment * np.exp(cumulative_return)
        simulated_end_values.append(simulated_price)

    # Calculate losses
    simulated_end_values = np.array(simulated_end_values)
    simulated_losses = initial_investment - simulated_end_values
    var = np.percentile(simulated_losses, 100 * confidence_level)

    return round(var, 2)


def calculate_risk(symbol="BTC/USDT", confidence_level=0.95):
    prices = fetch_historical_prices(symbol)
    if prices is None or len(prices) < 2:
        return {"symbol": symbol, "error": "Not enough data for risk assessment."}

    returns = calculate_log_returns(prices)
    var = monte_carlo_var(returns, confidence_level)

    return {
        "symbol": symbol,
        "confidence_level": f"{int(confidence_level * 100)}%",
        "value_at_risk": f"{abs(var)}%",
        "comment": "Potential loss over 1 hour with 95% confidence using Monte Carlo simulation"
    }

# Main function to get risk for a symbol
def get_risk(symbol="BTC/USDT", confidence_level=0.95):
    prices = fetch_historical_prices(symbol)
    if prices is None or len(prices) < 2:
        return {"symbol": symbol, "error": "Not enough data for risk assessment."}

    log_returns = calculate_log_returns(prices)
    var = monte_carlo_var(log_returns, confidence_level=confidence_level)

    return {
        "symbol": symbol,
        "confidence_level": f"{int(confidence_level * 100)}%",
        "value_at_risk": f"${var}",  # dollar risk estimate
        "comment": "Potential 1-hour loss (Monte Carlo, 1000 simulations)"
    }

# Example usage
if __name__ == "__main__":
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    for sym in symbols:
        risk = get_risk(sym)
        print(f"🛡️ Risk Assessment for {sym}: {risk}")

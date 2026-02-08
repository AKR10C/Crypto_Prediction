from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import ccxt
import os
from pydantic import BaseModel
from typing import List
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import sys
sys.path.append("E:/mad/crypto_ai_advisor/ml_models/trading_rl")  # Add path to import
sys.path.append("E:/mad/crypto_ai_advisor/ml_models/risk_analysis")  # Add path to import risk analysis functions
sys.path.append("E:/mad/crypto_ai_advisor/ml_models/sentiment_analysis")  # Add path to import sentiment analysis functions
from sentiment_nlp import fetch_and_analyze_reddit_posts, get_trading_suggestion  # Import your functions
from var_model import fetch_historical_prices, calculate_log_returns, monte_carlo_var  # Import your functions

from predict_realtime import get_trading_action  # Import your function

# Disable some TensorFlow optimizations (optional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Create FastAPI app
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Corrected to avoid the trailing slash
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Crypto Sage API!"}

# Load the trained model
model_path = "trained_models\\dqn_model_BTC_USDT.h5"
model = load_model(model_path, compile=False)
print("Model loaded successfully!")

model.compile(optimizer='adam', loss=MeanSquaredError())
exchange = ccxt.binance()

def get_latest_closes(symbol='BTC/USDT', window=10):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=window)
        closes = [x[4] for x in ohlcv]
        return np.array(closes).reshape(1, -1)
    except Exception as e:
        print(f"⚠️ Price fetch error: {e}")
        return None

@app.get("/api/predictions")
def get_predictions():
    try:
        result = get_trading_action()
        return result
    except Exception as e:
        return [{"error": str(e)}]
    

# Risk assessment endpoint
from var_model import fetch_historical_prices, calculate_log_returns, monte_carlo_var, calculate_risk

@app.get("/api/risk")
def risk_endpoint():
    try:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        results = [calculate_risk(symbol) for symbol in symbols]
        return results
    except Exception as e:
        return [{"error": str(e)}]
    
# Sentiment analysis endpoint

class SentimentResult(BaseModel):
    symbol: str
    sentiment: str
    compound_score: float
    details: dict

@app.get("/api/sentiment", response_model=List[SentimentResult])
def get_sentiment_data():
    # Replace this with your actual sentiment analysis logic
    return [
        {
            "symbol": "BTC",
            "sentiment": "positive",
            "compound_score": 0.8,
            "details": {"pos": 0.9, "neu": 0.1, "neg": 0.0}
        },
        {
            "symbol": "ETH",
            "sentiment": "neutral",
            "compound_score": 0.0,
            "details": {"pos": 0.3, "neu": 0.7, "neg": 0.0}
        }
    ]
 



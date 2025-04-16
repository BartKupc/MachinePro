import pandas as pd
import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import sys
import logging


# === CONFIG ===
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))
model_path = base_dir / 'XG' / 'model'
key_path = base_dir / 'config' / 'config.json'


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


from utils.bitget_futures import BitgetFutures
from utils.feature_calculator import calculate_all_features



# === Load latest RF model ===
# model_files = sorted(model_path.glob('rf_model_*.pkl'))
# if not model_files:
#     raise FileNotFoundError("No trained RF model found.")

# latest_model_file = model_files[-1]
# with open(latest_model_file, 'rb') as f:
#     rf_model = pickle.load(f)





def fetch_data(bitget_client, symbol, timeframe, days_to_analyze, warmup_period):
    """Fetch historical data from Bitget"""
    
    try:
        # Calculate total days needed (analysis period + warmup)
        total_days_needed = days_to_analyze + warmup_period
        
        # Calculate the start date
        start_date = (datetime.now() - timedelta(days=total_days_needed)).strftime('%Y-%m-%d')
        
        logging.info(f"Fetching {timeframe} data from {start_date} for {symbol}")
        logging.info(f"Analysis period: {days_to_analyze} days with {warmup_period} days warmup")
        logging.info(f"Expected candles: ~{days_to_analyze * 24} for analysis + ~{warmup_period * 24} for warmup")
        
        # Get the client from config and access correct properties
        data = bitget_client.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_date
        )
        
        logging.info(f"Fetched {len(data)} candles for {symbol} with {timeframe} timeframe")
        

        return data
        
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        raise


# === Load Bitget client ===
with open(key_path, 'r') as f:
    api_setup = json.load(f)['bitget']

bitget_client = BitgetFutures(api_setup)


# === Fetch data ===
symbol = 'ETH/USDT:USDT'
timeframe = '1h'
days_to_analyze = 45  # More data for backtest
warmup_period = 5

# data = fetch_data(bitget_client, symbol, timeframe, days_to_analyze, warmup_period)

data = pd.read_csv(base_dir / 'ohlcv' / '25_02_12_09_54.csv')

# === Calculate features ===
features_df, df_price = calculate_all_features(data)
features_df['past_return'] = df_price['close'].pct_change(3)
features_df['future_return'] = df_price['close'].pct_change(3).shift(-3)
features_df = features_df.dropna()


# Small model load
with open(model_path / 'xgb_classifier_2025-04-14_21-15-17.pkl', 'rb') as f:
    xg_model = pickle.load(f)

# Load feature list
with open(model_path/ 'small'/'xgb_classifier_2025-04-14_21-15-17.features.json', 'r') as f:
    feature_names = json.load(f)

# === Predict returns ===
#================Small data================
numeric_columns = features_df.select_dtypes(include=['float64', 'int64']).columns
columns_to_drop = ['future_return', 'open', 'high', 'low', 'close', 'volume', 'past_return']
X = features_df[numeric_columns].drop(columns=columns_to_drop, errors='ignore')

# Print feature names for debugging
print("\nFeature names for prediction:")
print(sorted(X.columns.tolist()))

# Get predicted probabilities and store them
pred_probs = xg_model.predict_proba(X)
direction = np.argmax(pred_probs, axis=1)
confidence = np.max(pred_probs, axis=1)  # Get confidence scores
features_df['direction'] = direction
features_df['confidence'] = confidence

# Print class distribution
print("\nPredicted class distribution (Model 1):")
print(pd.Series(direction).value_counts().sort_index())

#================Big data================
with open(model_path / 'xgb_classifier_2025-04-14_16-36-05.pkl', 'rb') as f:
    xg_model2 = pickle.load(f)

# Load feature list
with open(model_path/ 'big'/'xgb_classifier_2025-04-14_21-15-17.features.json', 'r') as f:
    feature_names = json.load(f)
pred_probs2 = xg_model2.predict_proba(X)
direction2 = np.argmax(pred_probs2, axis=1)
confidence2 = np.max(pred_probs2, axis=1)  # Get confidence scores
features_df['direction2'] = direction2
features_df['confidence2'] = confidence2

print("\nPredicted class distribution (Model 2):")
print(pd.Series(direction2).value_counts().sort_index())

# === Build Backtest DataFrame ===
bt_df = df_price[['open', 'high', 'low', 'close']].copy()
bt_df.index = pd.to_datetime(df_price['timestamp'])
bt_df.columns = ['Open', 'High', 'Low', 'Close']
bt_df = bt_df.iloc[-len(features_df):]  # Align rows
bt_df['prediction'] = features_df['direction'].values
bt_df['prediction2'] = features_df['direction2'].values
bt_df['confidence'] = features_df['confidence'].values
bt_df['confidence2'] = features_df['confidence2'].values

# Calculate rolling returns for dynamic thresholds
bt_df['returns'] = bt_df['Close'].pct_change()
bt_df['rolling_std'] = bt_df['returns'].rolling(window=24).std()
bt_df['rolling_mean'] = bt_df['returns'].rolling(window=24).mean()

# === Backtest ===
class XGStrategy(Strategy):
    def init(self):
        self.pred = self.data.prediction
        self.pred2 = self.data.prediction2

    def next(self):
        current_index = len(self.data) - 1
        pred = self.pred[current_index]
        pred2 = self.pred2[current_index]

        # Only trade when both models agree
        if pred == 1 and pred2 == 1 and not self.position:
            self.buy()
        elif pred == 0 and pred2 == 0 and not self.position:
            self.sell()

        # Close positions when models disagree
        if self.position:
            if self.position.is_long and (pred == 0 or pred2 == 0):
                self.position.close()
            elif self.position.is_short and (pred == 1 or pred2 == 1):
                self.position.close()

# ====Run Backtest====
bt = Backtest(bt_df, XGStrategy, cash = 10000, commission = 0.0001, trade_on_close = True,exclusive_orders=True)
results = bt.run()
bt.plot(filename = str(base_dir / 'backtest' / 'visualization' / 'xg_backtest.html'))

# ====Print Results====
print(f"Backtest results for {symbol} {timeframe} timeframe:")
print(results)

# Print some statistics about confidence levels
print("\nConfidence statistics:")
print("Model 1 confidence:", features_df['confidence'].describe())
print("\nModel 2 confidence:", features_df['confidence2'].describe())

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
model_path = base_dir / 'RF' / 'model'
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
model_files = sorted(model_path.glob('rf_model_*.pkl'))
if not model_files:
    raise FileNotFoundError("No trained RF model found.")

latest_model_file = model_files[-1]
with open(latest_model_file, 'rb') as f:
    rf_model = pickle.load(f)


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

# bitget_client = BitgetFutures(api_setup)

'''
# === Fetch data ===
symbol = 'ETH/USDT:USDT'
timeframe = '1h'
days_to_analyze = 45  # More data for backtest
warmup_period = 5

data = fetch_data(bitget_client, symbol, timeframe, days_to_analyze, warmup_period)
'''
# Load and check CSV data
csv_data = pd.read_csv(base_dir / 'ohlcv' / '25_02_12_09_54.csv')
print("\nCSV Data Structure:")
print(f"Columns: {csv_data.columns.tolist()}")
print(f"First few rows:")
print(csv_data.head())

# Convert DataFrame to list of lists format that calculate_all_features expects
data = csv_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()

# === Calculate features ===
features_df, df_price = calculate_all_features(data)
features_df['past_return'] = df_price['close'].pct_change(3)
features_df['future_return'] = df_price['close'].pct_change(3).shift(-3)
features_df = features_df.dropna()

# === Predict returns ===
numeric_columns = features_df.select_dtypes(include=['float64', 'int64']).columns
X = features_df[numeric_columns].drop(columns=['future_return'], errors='ignore')
predicted_returns = rf_model.predict(X)
features_df['predicted_return'] = predicted_returns

# Print unique predictions to debug
print("\nDebug - Predictions:")
print(f"Number of predictions: {len(predicted_returns)}")
print(f"Number of unique predictions: {len(np.unique(predicted_returns))}")
print(f"Unique predictions: {np.unique(predicted_returns)}")
print(f"Mean prediction: {np.mean(predicted_returns):.6f}")
print(f"Std prediction: {np.std(predicted_returns):.6f}")

# === Build Backtest DataFrame ===
bt_df = df_price[['open', 'high', 'low', 'close']].copy()
# Convert timestamp to datetime index
bt_df.index = pd.to_datetime(df_price['timestamp'])
# Rename columns to match backtesting requirements
bt_df.columns = ['Open', 'High', 'Low', 'Close']
bt_df = bt_df.iloc[-len(features_df):]  # Align rows
bt_df['prediction'] = features_df['predicted_return'].values

# === Backtest ===
class RFStrategy(Strategy):
    threshold = 0.01  # 1% threshold for trades

    def init(self):
        self.pred = self.data.prediction

    def next(self):
        current_index = len(self.data) - 1  # Get current bar's index
        pred = self.pred[current_index]  # Get prediction for current bar
        print(f"Prediction: {pred}")
        
        if pred > self.threshold and not self.position:
            self.buy()
        elif pred < -self.threshold and not self.position:
            self.sell()

        if self.position:
            if self.position.is_long and pred < 0:
                self.position.close()
            elif self.position.is_short and pred > 0:
                self.position.close()

# ====Run Backtest====
bt = Backtest(bt_df, RFStrategy, cash = 10000, commission = 0.0001, trade_on_close = True,exclusive_orders=True)
results = bt.run()
bt.plot(filename = str(base_dir / 'backtest' / 'visualization' / 'rf_backtest.html'))

# ====Print Results====
# print(f"Backtest results for {symbol} {timeframe} timeframe:")
print(results)

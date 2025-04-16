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
# model_files = sorted(model_path.glob('rf_model_*.pkl'))
# if not model_files:
#     raise FileNotFoundError("No trained RF model found.")

# latest_model_file = model_files[-1]
# with open(latest_model_file, 'rb') as f:
#     rf_model = pickle.load(f)

# Small model load
with open(model_path / 'rf_model_small_2025-04-14_09-37-03.pkl', 'rb') as f:
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

# === Predict returns ===

#================Small data================
numeric_columns = features_df.select_dtypes(include=['float64', 'int64']).columns
X = features_df[numeric_columns].drop(columns=['future_return', 'open', 'high', 'low', 'close', 'volume'], errors='ignore')
predicted_returns = rf_model.predict(X)
features_df['predicted_return'] = predicted_returns

#================Big data================

# === Load latest RF model ===
with open(model_path / 'rf_model_big_2025-04-13_23-17-25.pkl', 'rb') as f:
    rf_model2 = pickle.load(f)

predicted_returns = rf_model2.predict(X)
features_df['predicted_return2'] = predicted_returns


# === Build Backtest DataFrame ===
bt_df = df_price[['open', 'high', 'low', 'close']].copy()
# Convert timestamp to datetime index
bt_df.index = pd.to_datetime(df_price['timestamp'])
# Rename columns to match backtesting requirements
bt_df.columns = ['Open', 'High', 'Low', 'Close']
bt_df = bt_df.iloc[-len(features_df):]  # Align rows
bt_df['prediction'] = features_df['predicted_return'].values
bt_df['prediction2'] = features_df['predicted_return2'].values


bt_df.to_csv(base_dir / 'backtest' / 'features' / 'rf_hybrid_features.csv')


# === Backtest ===
class RFStrategy(Strategy):


    def init(self):
        self.pred = self.data.prediction
        self.pred2 = self.data.prediction2

        self.big_threshold_buy = self.pred2.mean() + self.pred2.std()
        self.big_threshold_sell = self.pred2.mean() - self.pred2.std()

        self.small_threshold_buy = self.pred.mean() + self.pred.std()
        self.small_threshold_sell = self.pred.mean() - self.pred.std()

        # Dynamic reversal exit logic
        self.exit_long_threshold = self.pred.mean() + 0.5 * self.pred.std()
        self.exit_short_threshold = self.pred.mean() - 0.5 * self.pred.std()

    def next(self):
        current_index = len(self.data) - 1  # Get current bar's index
        pred = self.pred[current_index]  # Get prediction for current bar
        pred2 = self.pred2[current_index]

        if pred > self.small_threshold_buy and not self.position:
            if pred2 > self.big_threshold_buy:
                self.buy()

        elif pred < self.small_threshold_sell and not self.position:
            if pred2 < self.big_threshold_sell:
                self.sell()
        

        if self.position:
            if self.position.is_long and pred < self.exit_long_threshold:
                self.position.close()
            elif self.position.is_short and pred > self.exit_short_threshold:
                self.position.close()

# ====Run Backtest====
bt = Backtest(bt_df, RFStrategy, cash = 10000, commission = 0.0001, trade_on_close = True,exclusive_orders=True)
results = bt.run()
bt.plot(filename = str(base_dir / 'backtest' / 'visualization' / 'rf_backtest.html'))

# ====Print Results====
print(f"Backtest results for {symbol} {timeframe} timeframe:")
print(results)

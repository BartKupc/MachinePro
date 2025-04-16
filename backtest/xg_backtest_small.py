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

data = fetch_data(bitget_client, symbol, timeframe, days_to_analyze, warmup_period)

# data = pd.read_csv(base_dir / 'ohlcv' / '25_02_12_09_54.csv')

# === Calculate features ===
features_df, df_price = calculate_all_features(data)
features_df['past_return'] = df_price['close'].pct_change(3)
features_df['future_return'] = df_price['close'].pct_change(3).shift(-3)
features_df = features_df.dropna()


# Small model load
with open(model_path / 'big' / 'xgb_classifier_2025-04-15_10-33-33_refit.pkl', 'rb') as f:
    xg_model = pickle.load(f)


# === Predict returns ===
# Get the model's feature names
model_features = xg_model.feature_names_in_
if model_features is None:
    # If model doesn't have feature names, use our predefined list
    model_features = [
        'rsi', 'roc_20', 'price_sma50_ratio', 'rsi_ma5', 'vwap_ratio', 
        'macd_signal', 'price_vs_kumo', 'di_minus', 'macd', 'di_plus', 
        'bb_width', 'sma50', 'tenkan_sen', 'dist_from_low_20', 'bb_lower', 
        'mfi', 'donchian_width', 'atr_ratio', 'obv', 'atr_pct', 
        'williams_r', 'ema9', 'cmf', 'obv_ma20', 'historical_volatility_30', 
        'bb_upper', 'vwap_24', 'cci', 'volume_change_3', 'stoch_rsi_k'
    ]

# Select features in the exact order required by the model
X = features_df[model_features]

# Print feature names for verification
print("\nFeature names for prediction (in order):")
print(X.columns.tolist())

# Get predicted probabilities and store them
pred_probs = xg_model.predict_proba(X)
features_df['proba_buy'] = pred_probs[:, 1]  # Probabilities for class 1 (buy)




# === Build Backtest DataFrame ===
bt_df = df_price[['open', 'high', 'low', 'close']].copy()
bt_df.index = pd.to_datetime(df_price['timestamp'])
bt_df.columns = ['Open', 'High', 'Low', 'Close']
bt_df = bt_df.iloc[-len(features_df):]  # Align rows
bt_df['proba_buy'] = features_df['proba_buy'].values



# Calculate rolling returns for dynamic thresholds
bt_df['returns'] = bt_df['Close'].pct_change()
bt_df['rolling_std'] = bt_df['returns'].rolling(window=24).std()
bt_df['rolling_mean'] = bt_df['returns'].rolling(window=24).mean()

# === Backtest ===
class XGStrategy(Strategy):
    def init(self):
        self.proba_buy = self.data.proba_buy
        self.buy_threshold = 0.7
        self.sell_threshold = 0.3


    def next(self):
        current_index = len(self.data) - 1
        proba_buy = self.proba_buy[current_index]

        # Only trade when both models agree
        if proba_buy > self.buy_threshold and not self.position:
            self.buy()
        elif proba_buy < self.sell_threshold and not self.position:
            self.sell()

        # Close positions when models disagree
        if self.position:
            if self.position.is_long and proba_buy < self.sell_threshold:
                self.position.close()
            elif self.position.is_short and proba_buy > self.buy_threshold:
                self.position.close()

# ====Run Backtest====
bt = Backtest(bt_df, XGStrategy, cash = 10000, commission = 0.0001, trade_on_close = True,exclusive_orders=True)
results = bt.run()
bt.plot(filename = str(base_dir / 'backtest' / 'visualization' / 'xg_backtest.html'))

# ====Print Results====
print(f"Backtest results for {symbol} {timeframe} timeframe:")
print(results)


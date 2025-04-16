import pandas as pd
import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
import sys
import logging

# === CONFIG ===
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))
model_path = base_dir / 'XG' / 'model'
small_model_path = model_path / 'small'
key_path = base_dir / 'config' / 'config.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.bitget_futures import BitgetFutures
from utils.feature_calculator import calculate_all_features

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


# === Load Models ===
# Load the main (big) model
with open(model_path / 'xgb_regressor.pkl', 'rb') as f:
    xgb_model_big = pickle.load(f)

# Load the small model (first pass)
with open(small_model_path / 'xgb_regressor.pkl', 'rb') as f:
    xgb_model_small = pickle.load(f)

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

# data = pd.read_csv(base_dir / 'ohlcv' / '25_02_12_09_54.csv')
data = pd.read_csv(base_dir / 'ohlcv' / 'ohlcv_from_2024-01-17_to_2025-01-25.csv')


# === Feature Engineering ===
features_df, df_price = calculate_all_features(data)
features_df['future_return'] = df_price['close'].pct_change(3).shift(-3)
features_df.dropna(inplace=True)


numeric_columns = features_df.select_dtypes(include=['float64', 'int64']).columns

top_25_features = [
    'volume_ma20',
    'atr_ratio',
    'macd_signal',
    'historical_volatility_30',
    'adx',
    'atr_pct',
    'roc_10',
    'rsi',
    'volume_ma50',
    'volume_change_3',
    'sma100',
    'donchian_width',
    'obv_ma20',
    'senkou_span_a',
    'bb_width',
    'dist_from_high_20',
    'donchian_high_20',
    'obv_ratio',
    'mfi',
    'di_plus',
    'di_minus',
    'supertrend',
    'volume_ratio_50',
    'stoch_diff',
    'atr_ma100'
]
X = features_df[top_25_features]

# === Predict with both XGBoost models ===
features_df['predicted_return_big'] = xgb_model_big.predict(X)
features_df['predicted_return_small'] = xgb_model_small.predict(X)

# === Build Backtest DataFrame ===
bt_df = df_price[['open', 'high', 'low', 'close']].copy()
bt_df.index = pd.to_datetime(df_price['timestamp'])
bt_df.columns = ['Open', 'High', 'Low', 'Close']
bt_df = bt_df.iloc[-len(features_df):]
bt_df['prediction_big'] = features_df['predicted_return_big'].values
bt_df['prediction_small'] = features_df['predicted_return_small'].values

# === Backtesting Strategy ===
class TwoPassXGBStrategy(Strategy):
    def init(self):
        # Big model predictions
        self.pred_big = self.data.prediction_big
        self.mean_big = self.pred_big.mean()
        self.std_big = self.pred_big.std()
        
        # Small model predictions
        self.pred_small = self.data.prediction_small
        self.mean_small = self.pred_small.mean()
        self.std_small = self.pred_small.std()

        # Thresholds for big model
        self.buy_threshold_big = self.mean_big + 0.3 * self.std_big
        self.sell_threshold_big = self.mean_big - 0.3 * self.std_big

        self.exit_long_threshold_big = self.mean_big - 0.1 * self.std_big
        self.exit_short_threshold_big = self.mean_big + 0.1 * self.std_big

        # Thresholds for small model
        self.buy_threshold_small = self.mean_small + 0.3 * self.std_small
        self.sell_threshold_small = self.mean_small - 0.3 * self.std_small

    def should_buy(self, i):
        # Two-pass check for buy signals
        small_pred = self.pred_small[i]
        big_pred = self.pred_big[i]
        
        # First pass: Check if small model suggests a buy
        if small_pred > self.buy_threshold_small:
            # Second pass: Confirm with big model
            if big_pred > self.buy_threshold_big:
                return True
        return False

    def should_sell(self, i):
        # Two-pass check for sell signals
        small_pred = self.pred_small[i]
        big_pred = self.pred_big[i]
        
        # First pass: Check if small model suggests a sell
        if small_pred < self.sell_threshold_small:
            # Second pass: Confirm with big model
            if big_pred < self.sell_threshold_big:
                return True
        return False

    def should_exit_long(self, i):
        # Check only big model for exit signals
        big_pred = self.pred_big[i]
        return big_pred < self.exit_long_threshold_big

    def should_exit_short(self, i):
        # Check only big model for exit signals
        big_pred = self.pred_big[i]
        return big_pred > self.exit_short_threshold_big

    def next(self):
        i = len(self.data) - 1
        
        if not self.position:
            # Entry signals - check both models via should_buy/should_sell
            if self.should_buy(i):
                self.buy()
            elif self.should_sell(i):
                self.sell()
        else:
            # Exit signals - check both models
            if self.position.is_long and self.should_exit_long(i):
                self.position.close()
            elif self.position.is_short and self.should_exit_short(i):
                self.position.close()

# === Run Backtest ===
bt = Backtest(bt_df, TwoPassXGBStrategy, cash=10000, commission=0.0001, trade_on_close=True, exclusive_orders=True)
results = bt.run()
bt.plot(filename=str(base_dir / 'backtest' / 'visualization' / 'two_pass_xgb_backtest.html'))

print(f"\nTwo-Pass Backtest results for {symbol} {timeframe} timeframe:")
print(results)

import json
import ta
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import requests
import time
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys

# === CONFIG ===
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

model_path = base_dir / 'live' / 'models'

key_path = base_dir / 'config' / 'config.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.bitget_futures import BitgetFutures
from utils.feature_calculator import calculate_all_features

# Telegram Bot Setup
TELEGRAM_BOT_TOKEN = "7292159640:AAEZnq6jOkze_PBBWMiBHnyCW_dtiHVv6xo"
TELEGRAM_CHAT_ID = "7393611077"  # Make sure this is the correct chat ID

# === END CONFIG ===


class LiveXGB:
    def __init__(self):
        self.bitget = BitgetFutures()
        self.symbol = 'ETH/USDT:USDT'
        self.timeframe = '1h'
        self.total_days_needed = 7
        self.leverage = 3

    def fetch_data(self):
        """Fetch historical data from Bitget"""
        
        try:
            # Calculate total days needed (analysis period + warmup)
            total_days_needed = self.total_days_needed
            
            # Calculate the start date
            start_date = (datetime.now() - timedelta(days=total_days_needed)).strftime('%Y-%m-%d')
            
            logging.info(f"Fetching {self.timeframe} data from {start_date} for {self.symbol}")
            logging.info(f"Analysis period: {self.total_days_needed} days")
            logging.info(f"Expected candles: ~{self.total_days_needed * 24}")
            
            # Get the client from config and access correct properties
            data = self.bitget.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_time=start_date
            )
            
            logging.info(f"Fetched {len(data)} candles for {self.symbol} with {self.timeframe} timeframe")
            

            return data
            
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            raise

    def feature_engineering(self, data):
        # === Feature Engineering ===
        features_df, df_price = calculate_all_features(data)
        features_df['future_return'] = df_price['close'].pct_change(3).shift(-3)
        features_df.dropna(inplace=True)

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

        return X
        
    def load_models(self):
        xgb_model_big = pickle.load(open(model_path / 'xgb_regressor.pkl', 'rb'))
        xgb_model_small = pickle.load(open(model_path / 'xgb_regressor_small.pkl', 'rb'))

        return xgb_model_big, xgb_model_small

    def predict(self, features_df, xgb_model_big, xgb_model_small, df_price):

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

        return bt_df
    
    def check_recent_trades(self):
        """Check recent trade history for losses with minimal logging"""
        
        try:
            # Fetch recent trades from Bitget


            # trades = self.bitget.fetch_my_trades(self.symbol, limit=20)  # Get 20 most recent trades


    def calculate_position_size(self, close_price):
        """Calculate position size based on account balance"""
        try:
            balance = self.bitget.fetch_balance()
            usdt_balance = float(balance['USDT']['free'])
            
            # Use only a portion of the balance (e.g., 95% to account for fees)
            safe_balance = usdt_balance * 0.95
            
            # Calculate position value with leverage
            position_value = safe_balance * self.leverage
            
            # Calculate quantity in contracts
            quantity = position_value / close_price
            
            # Round down to avoid exceeding available margin
            quantity = float(self.bitget.amount_to_precision(self.symbol, quantity))
            
            logging.info(f"Position Size Calculation:")
            logging.info(f"Available USDT: ${usdt_balance:.2f}")
            logging.info(f"Safe Balance (95%): ${safe_balance:.2f}")
            logging.info(f"Position Value (with {self.leverage}x leverage): ${position_value:.2f}")
            logging.info(f"Calculated Quantity: {quantity} contracts")
            
            return quantity
        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return None

    def check_adf(self, data):
        if data['adf'] > 20:
            return True
        else:
            return False


    def live_strategy(self, bt_df):
        """Live strategy for XGBoost"""






    def send_telegram_message(self, message):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message
        }
        requests.get(url, params=params)





if __name__ == "__main__":
    # === Load Bitget client ===
    with open(key_path, 'r') as f:
        api_setup = json.load(f)['bitget']

    bitget_client = BitgetFutures(api_setup)
    data = bitget_client.fetch_ohlcv()

    live_xg = LiveXGB()
    X = live_xg.feature_engineering(data)
    xgb_model_big, xgb_model_small = live_xg.load_models()
    predicted_df = live_xg.predict(X, xgb_model_big, xgb_model_small, data)
    live_xg.send_telegram_message(

    )


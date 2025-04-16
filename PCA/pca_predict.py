import numpy as np                                
import pandas as pd                               
import matplotlib.pyplot as plt                   
from datetime import datetime, timedelta
import sys                                        
import json                                       
import os                                         
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
import logging
import seaborn as sns
import pickle
import gzip
import urllib
import csv
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from sklearn.decomposition import PCA


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add this to properly import from parent directory
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))


from utils.bitget_futures import BitgetFutures
from utils.feature_calculator import calculate_all_features

class PCAPredict:
    def __init__(self, bitget_client):
        self.n_components = 5
        self.symbol ='ETH/USDT:USDT'
        self.timeframe = '1h'
        self.warmup_period = 4
        self.days_to_analyze = 30
        self.correlation_threshold = 0.1
        self.future_return_periods = [1, 3, 5, 10]
        self.bitget_client = bitget_client

    def fetch_data(self):
        """Fetch historical data from Bitget"""
        
        try:

            
            # Calculate total days needed (analysis period + warmup)
            total_days_needed = self.days_to_analyze + self.warmup_period
            
            # Calculate the start date
            start_date = (datetime.now() - timedelta(days=total_days_needed)).strftime('%Y-%m-%d')
            
            logging.info(f"Fetching {self.timeframe} data from {start_date} for {self.symbol}")
            logging.info(f"Analysis period: {self.days_to_analyze} days with {self.warmup_period} days warmup")
            logging.info(f"Expected candles: ~{self.days_to_analyze * 24} for analysis + ~{self.warmup_period * 24} for warmup")
            
            # Get the client from config and access correct properties
            data = self.bitget_client.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_time=start_date
            )
            
            logging.info(f"Fetched {len(data)} candles for {self.symbol} with {self.timeframe} timeframe")
            

            return data
            
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            raise
    

    def calculate_features(self, data):
        """Calculate all features"""
        features_df , df = calculate_all_features(data)
        return features_df , df

    def filter_features(self, features_df):
        """Filter features based on correlation threshold"""
        filtered_columns = [
        "macd", "macd_diff", "ema9", "sma20", "sma50", "sma100",
        "price_ema9_ratio", "price_sma20_ratio", "price_sma50_ratio",
        "di_plus", "di_minus", "bb_upper", "bb_lower", "bb_pct",
        "obv", "obv_ma20", "rsi", "rsi_ma5", "cci", "roc_10", "roc_20",
        "donchian_high_20", "donchian_low_20", "dist_from_high_20",
        "vwap_24", "vwap_ratio", "tenkan_sen", "kijun_sen", "senkou_span_a",
        "price_vs_kumo", "supertrend", "supertrend_direction", "mfi", "cmf",
        "ema9_ratio"
        ]
        df = features_df[[col for col in filtered_columns if col in features_df.columns]]

        return df

    def load_pca_model_and_scaler(self):
        """Load PCA model"""
        pca_model = pickle.load(open(base_dir / 'PCA' / 'model' / 'PCA_model_2025-04-05_21-05-34.pkl', 'rb'))
        scaler = pickle.load(open(base_dir / 'PCA' / 'model' / 'PCA_scaler_2025-04-05_21-05-34.pkl', 'rb'))
        return pca_model, scaler
    
    def transform_features_and_scale(self, pca_model, scaler, df):
        """Transform features"""
        latest_features = df.tail(1)
        latest_features_scaled = scaler.transform(latest_features)
        latest_features_pca = pca_model.transform(latest_features_scaled)
  
        pca1, pca2, pca3 = latest_features_pca[0][:3]

        return pca1 , pca2 , pca3


    def market_description(self, pc1, pc2, pc3):
        print("\n📊 PCA Market Summary for", self.symbol, f"({self.timeframe})")
        print()

        # Interpret each principal component
        def describe_pc(value, name):
            if value > 2:
                return f"{name}: {value:.2f}  → Very High 📈"
            elif value > 1:
                return f"{name}: {value:.2f}  → High ✅"
            elif value > 0:
                return f"{name}: {value:.2f}  → Mild Positive ↗️"
            elif value > -1:
                return f"{name}: {value:.2f}  → Neutral ⚪"
            elif value > -2:
                return f"{name}: {value:.2f}  → Low 📉"
            else:
                return f"{name}: {value:.2f}  → Very Low 🔻"

        print(describe_pc(pc1, "PC1 (Trend Strength)"))
        print(describe_pc(pc2, "PC2 (Short-Term Trend/Volume)"))
        print(describe_pc(pc3, "PC3 (Momentum Shift)"))
        print()

        # General summary
        if pc1 > 1 and pc2 > 1 and pc3 > 0:
            print("📈 Overall Interpretation: Bullish Bias — Market supports Long Entries 🚀")
        elif pc1 < -1 and pc2 < -1 and pc3 < 0:
            print("📉 Overall Interpretation: Bearish Bias — Market supports Short Entries 🧨")
        elif pc3 > 2:
            print("⚡ Momentum Spike Detected — Watch for Volatility or Breakouts!")
        elif pc3 < -2:
            print("🔄 Reversal Risk — Consider Taking Profits or Tightening Stops")
        else:
            print("🔍 Interpretation: Mixed/Neutral — Wait for clearer signal or confirmation.")




if __name__ == "__main__":
    
    # Load configuration from config.json
    key_path = base_dir / 'config' / 'config.json'
    with open(key_path, "r") as f:
        api_setup = json.load(f)['bitget']
    
    # Initialize BitgetFutures client
    bitget_client = BitgetFutures(api_setup)
    
    # Initialize class
    pcaPredict = PCAPredict(bitget_client)

    # Fetch data
    data = pcaPredict.fetch_data()

    # Calculate features
    features_df , df = pcaPredict.calculate_features(data)

    # Filter features
    df = pcaPredict.filter_features(features_df)
    print(df.head())

    # Load PCA model
    pca_model, scaler = pcaPredict.load_pca_model_and_scaler()

    # Transform features
    pca1 , pca2 , pca3 = pcaPredict.transform_features_and_scale(pca_model, scaler, df)
    print(pca1 , pca2 , pca3)

    # Get market description
    pcaPredict.market_description(pca1 , pca2 , pca3)




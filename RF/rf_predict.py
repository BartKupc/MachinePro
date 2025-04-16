import numpy as np                                
import pandas as pd                               
import matplotlib.pyplot as plt                   
from datetime import datetime, timedelta
import sys                                        
import json                                       
import os                                         
from pathlib import Path
import logging
import seaborn as sns
import pickle
import gzip
import urllib
import csv
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import optuna



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add this to properly import from parent directory
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))
model_path = base_dir / 'RF' / 'model'

from utils.bitget_futures import BitgetFutures
from utils.feature_calculator import calculate_all_features

class RFPredictor:
    def __init__(self, bitget_client):
        self.bitget_client = bitget_client
        self.symbol = 'ETH/USDT:USDT'
        self.timeframe = '1h'
        self.warmup_period = 5
        self.days_to_analyze = 10
        self.model_dir = Path(__file__).parent / 'model'
        self.visual_dir = Path(__file__).parent / 'visualizations'

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir.mkdir(parents=True, exist_ok=True)
        

    def load_model(self):

        model_files = sorted(model_path.glob('*.pkl'))
        if not model_files:
            raise FileNotFoundError("No model files found in the model directory")

        latest_model_path = model_files[-1]
        with open(latest_model_path, 'rb') as f:
            model = pickle.load(f)

        return model

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
        features_df['past_return']= df['close'].pct_change(3)

        features_df = features_df.dropna()

        return features_df , df    
    
    def predict(self, model, features_df):
        """Predict future returns"""
        latest_features = features_df.tail(1)
        predicted_return = model.predict(latest_features)[0]
        logging.info(f"Predicted 3h return: {predicted_return:.4%}")

        return predicted_return


if __name__ == "__main__":

    # Load configuration from config.json
    key_path = base_dir / 'config' / 'config.json'
    with open(key_path, "r") as f:
        api_setup = json.load(f)['bitget']

    # Initialize Bitget client
        # Initialize Bitget client
    bitget_client = BitgetFutures()

    rf = RFPredictor(bitget_client)

    # Load model
    model = rf.load_model()

    # Fetch data
    data = rf.fetch_data()

    # Calculate features
    features_df, df = rf.calculate_features(data)
    numeric_columns = features_df.select_dtypes(include=['float64', 'int64']).columns
    X = features_df[numeric_columns]
    print(X.head())
    # Predict
    predicted_return = rf.predict(model, X)



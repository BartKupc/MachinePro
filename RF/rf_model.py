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


from utils.bitget_futures import BitgetFutures
from utils.feature_calculator import calculate_all_features

class RFTrainer:
    def __init__(self, bitget_client):
        self.bitget_client = bitget_client
        self.symbol ='ETH/USDT:USDT'
        self.timeframe = '1h'
        self.warmup_period = 4
        self.days_to_analyze = 30
        # Define directories for saving models and visualizations
        self.model_dir = Path(__file__).parent / 'model'
        self.visual_dir = Path(__file__).parent / 'visualizations'
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir.mkdir(parents=True, exist_ok=True)

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

    def target_engineering(self,features_df, df):
        """Target engineering"""
        # Calculate future returns
        features_df['future_return'] = df['close'].pct_change(3).shift(-3)
        features_df['past_return'] = df['close'].pct_change(3)
        features_df['bullish'] = (features_df['future_return'] >0).astype(int)
        features_df= features_df.dropna()
        return features_df
        
    def random_forest_split(self, df):
        """Split data into training and testing sets"""
        # Drop any datetime columns, non-numeric columns, and raw OHLC values
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        columns_to_drop = ['future_return', 'bullish', 'open', 'high', 'low', 'close', 'volume']
        X = df[numeric_columns].drop(columns=columns_to_drop, errors='ignore')
        y = df['future_return']
        
        # Log the features being used
        logging.info(f"Using {len(X.columns)} features for training")
        logging.info(f"Features: {list(X.columns)}")
        logging.info(f"Excluded raw OHLCV columns to ensure scale invariance")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    

    def run_optuna(self, X_train, y_train):
        
        def objective(trial):
            """Objective function for Random Forest model"""
            n_estimators = trial.suggest_int('n_estimators',100,500)
            max_depth = trial.suggest_int('max_depth',5,50)
            min_samples_split = trial.suggest_int('min_samples_split',2,10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf',1,5)
            max_features = trial.suggest_categorical('max_features',['sqrt','log2',None])

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )
            score = cross_val_score(model,X_train, y_train, cv=5, scoring='r2', n_jobs=-1).mean()
                 
            return score
        
        """Run Optuna"""
        study=optuna.create_study(direction='maximize')
        study.optimize(objective,n_trials=50)
        logging.info(f"Best hyperparameters: {study.best_params}")

        return study

    def random_forest_final(self, study, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        best_params=study.best_params
        best_model=RandomForestRegressor(**best_params)
        best_model.fit(X_train,y_train)
        preds=best_model.predict(X_test)
        logging.info(f"Model training complete")
        
        """EVALUATE AND SAVE MODEL"""
        r2= r2_score(y_test, preds)
        mse= mean_squared_error(y_test, preds)

        logging.info(f"R2: {r2}, MSE: {mse}")

        """Save Model"""
        model_path = self.model_dir / f"rf_model_small_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        logging.info(f"Model saved to {model_path}")

        return best_model

    def feature_importance(self, model, X):
        """Feature importance"""
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances_sorted = importances.sort_values(ascending=False)

        plt.figure(figsize=(10,6))
        importances_sorted.head(20).plot(kind='bar')
        plt.title("Top 20 Feature Importances - Random Forest")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig(self.visual_dir / f"feature_importance_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
        plt.close()
        logging.info("Saved feature importance plot")

        return importances_sorted

if __name__ == "__main__":
    
    # Load configuration from config.json
    key_path = base_dir / 'config' / 'config.json'
    with open(key_path, "r") as f:
        api_setup = json.load(f)['bitget']

    
    # Initialize Bitget client
    bitget_client = BitgetFutures()

    rf = RFTrainer(bitget_client)

    # 1. Fetch the data
    logging.info("\n===== FETCHING DATA =====")
    data = rf.fetch_data()
    logging.info(f"Fetched {len(data)} data points")

    # data = pd.read_csv(base_dir / 'ohlcv' / '25_02_12_09_54.csv')


    # 2. Calculate features
    logging.info("\n===== CALCULATING FEATURES =====")
    features_df, price_df = rf.calculate_features(data)
    logging.info(f"Calculated features: {features_df.shape}")

    # 3. Target engineering
    logging.info("\n====== TARGET ENGINEERING =====")
    features_df = rf.target_engineering(features_df, price_df)

    # 4. Random Forest Split
    logging.info("\n==== RANDOM FOREST SPLIT =====")
    X_train, X_test, y_train, y_test =rf.random_forest_split(features_df)

    # 5. Run Optuna
    logging.info("\n==== RUN OPTUNA =====")
    study = rf.run_optuna(RandomForestRegressor(), X_train, y_train)

    #6. Train final model
    logging.info("\n==== TRAIN FINAL MODEL =====")
    final_model = rf.random_forest_final(study, X_train, X_test, y_train, y_test)

    #7. Feature Importance
    logging.info("\n==== FEATURE IMPORTANCE =====")
    feature_importance = rf.feature_importance(final_model, X_train)
    

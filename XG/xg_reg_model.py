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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, 
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils.bitget_futures import BitgetFutures
from utils.feature_calculator import calculate_all_features

class XGTrainer:
    def __init__(self, bitget_client):
        self.bitget_client = bitget_client
        self.symbol = 'ETH/USDT:USDT'
        self.timeframe = '1h'
        self.warmup_period = 4
        self.days_to_analyze = 30
        self.model_dir = Path(__file__).parent / 'model' / 'big'
        self.visual_dir = Path(__file__).parent / 'visualizations'

    def fetch_data(self):
        try:
            total_days_needed = self.days_to_analyze + self.warmup_period
            
            # Calculate start date to be 1 year ago
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            logging.info(f"Fetching {self.timeframe} data from {start_date} for {self.symbol}")
            logging.info(f"Expected candles: ~{365 * 24} for one year of hourly data")
            
            data = self.bitget_client.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_time=start_date  # This will use the since parameter internally
            )
            
            logging.info(f"Fetched {len(data)} candles")
            if len(data) < 1000:
                logging.warning(f"Fetched less than 1000 candles. This might not be enough for reliable training.")
            
            # Log the date range of the data
            if len(data) > 0:
                start_time = pd.to_datetime(data['timestamp'].iloc[0])
                end_time = pd.to_datetime(data['timestamp'].iloc[-1])
                logging.info(f"Data range: from {start_time} to {end_time}")
            
            return data
            
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            raise

    def calculate_features(self, data):
        return calculate_all_features(data)

    def target_engineering(self, features_df, df):
        features_df['future_return'] = df['close'].pct_change(3).shift(-3)
        features_df = features_df.dropna()
        logging.info("Target (future_return) engineered.")
        return features_df

    def xg_split(self, features_df):
        """Split data into training and testing sets using proper time series split"""
        # Get numeric columns
        numeric_columns = features_df.select_dtypes(include=['float64', 'int64']).columns
        
        # Define columns to drop (including future_return to prevent data leakage)
        columns_to_drop = ['future_return', 'open', 'high', 'low', 'close', 'volume']
        
        # Select features and target
        X = features_df[numeric_columns].drop(columns=columns_to_drop, errors='ignore')
        y = features_df['future_return']
        tscv = TimeSeriesSplit(n_splits=5, test_size=len(X) // 5)
        train_index, test_index = list(tscv.split(X))[-1]
        return X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index], X, y

    def xg_optuna(self, X_train, y_train):
        """Optimize XGBoost parameters using Optuna"""
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'tree_method': 'hist',
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),
                'lambda': trial.suggest_float('lambda', 0.1, 1.0),
                'alpha': trial.suggest_float('alpha', 0.1, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 100, 300)
            }
            
            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                     verbose=False
                )
                
                # Predict and calculate R2 score
                pred = model.predict(X_fold_val)
                score = r2_score(y_fold_val, pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        
        logging.info("\nBest trial:")
        logging.info(f"Value: {study.best_value:.4f}")
        logging.info("Params:")
        for key, value in study.best_params.items():
            logging.info(f"    {key}: {value}")
        
        return study

    def xg_final(self, X_train, X_test, y_train, y_test, study, X, y):
        """Train final model on training data and evaluate on test data"""
        best_params = study.best_params
        best_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist'
        })
        
        # Train model only on training data
        model = xgb.XGBRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        # Evaluate on test set
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        logging.info("\nModel Evaluation on Test Set:")
        logging.info(f"RMSE: {rmse:.6f}")
        logging.info(f"R^2: {r2:.6f}")
        
        # Additional evaluation metrics
        mae = np.mean(np.abs(y_test - preds))
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        logging.info(f"MAE: {mae:.6f}")
        logging.info(f"MAPE: {mape:.2f}%")
        
        # Clean up old models
        for f in self.model_dir.glob("xgb_regressor_*.pkl"):
            f.unlink()
        
        # Save model
        path = self.model_dir / f"xgb_regressor_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {path}")
        
        return model

    def xg_feature_importance(self, model, X):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top = importances.sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        top.head(20).plot(kind='bar')
        plt.title("Top 20 Feature Importances - XGBoost")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig(self.visual_dir / f"feature_importance_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
        plt.close()
        return top

    def explain_with_shap(self, model, X):
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(self.visual_dir / f"shap_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
        plt.close()

if __name__ == "__main__":
    key_path = base_dir / 'config' / 'config.json'
    with open(key_path, "r") as f:
        api_setup = json.load(f)['bitget']

    client = BitgetFutures()
    xg = XGTrainer(client)

    # logging.info("\n===== FETCHING DATA =====")
    # data = xg.fetch_data()

    data = pd.read_csv(base_dir / 'ohlcv' / '25_02_12_09_54.csv')

    logging.info("\n===== CALCULATING FEATURES =====")
    features_df, price_df = xg.calculate_features(data)
    logging.info("\n====== TARGET ENGINEERING =====")
    features_df = xg.target_engineering(features_df, price_df)
    logging.info("\n==== XGB SPLIT =====")
    X_train, X_test, y_train, y_test, X, y = xg.xg_split(features_df)
    logging.info("\n==== XGB OPTUNA =====")
    study = xg.xg_optuna(X_train, y_train)
    logging.info("\n==== XGB FINAL =====")
    final_model = xg.xg_final(X_train, X_test, y_train, y_test, study, X, y)
    logging.info("\n==== XGB FEATURE IMPORTANCE =====")
    xg.xg_feature_importance(final_model, X)
    logging.info("\n==== XGB SHAP EXPLAIN =====")
    xg.explain_with_shap(final_model, X)

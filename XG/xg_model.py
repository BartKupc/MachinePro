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
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    mean_squared_error, 
    r2_score, 
    f1_score
)
from sklearn.model_selection import train_test_split, cross_val_score
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

# Add this to properly import from parent directory
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils.bitget_futures import BitgetFutures
from utils.feature_calculator import calculate_all_features

class XGTrainer:
    def __init__(self, bitget_client):
        self.bitget_client = bitget_client
        self.symbol ='ETH/USDT:USDT'
        self.timeframe = '1h'
        self.warmup_period = 4
        self.days_to_analyze = 30

        # Define directories for saving models and visualizations
        self.model_dir = Path(__file__).parent / 'model'/'small'
        self.visual_dir = Path(__file__).parent / 'visualizations'
        
        #XGBOOST params
        self.threshold = 0.002


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
    
    def label_direction_binary(self, r):
        """
        Label direction as binary: 1 (Buy), 0 (Sell)
        Based on z-score thresholds
        """
        returns = pd.Series(r)
        
        # Calculate rolling stats
        window = 24
        rolling_std = returns.rolling(window, min_periods=1).std()
        rolling_mean = returns.rolling(window, min_periods=1).mean()

        rolling_std[:window] = returns[:window].expanding(min_periods=1).std()
        rolling_mean[:window] = returns[:window].expanding(min_periods=1).mean()

        z_scores = (returns - rolling_mean) / rolling_std
        z_scores = z_scores.fillna(0)

        labels = np.where(z_scores > 0, 1, 0)  # 1 = Buy, 0 = Sell

        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        total = len(labels)
        logging.info("\nBinary signal distribution:")
        logging.info(f"Buy (1): {dist.get(1, 0)} ({dist.get(1, 0)/total*100:.1f}%)")
        logging.info(f"Sell (0): {dist.get(0, 0)} ({dist.get(0, 0)/total*100:.1f}%)")

        return labels

    def target_engineering(self, features_df, df):
        """Target engineering with dynamic thresholds"""
        # Calculate future returns
        features_df['future_return'] = df['close'].pct_change(3).shift(-3)
        features_df = features_df.dropna()
        
        # Calculate rolling statistics for monitoring
        rolling_std = features_df['future_return'].rolling(window=24).std()
        rolling_mean = features_df['future_return'].rolling(window=24).mean()
        
        logging.info("\nReturn statistics:")
        logging.info(f"Mean return: {features_df['future_return'].mean():.4f}")
        logging.info(f"Std return: {features_df['future_return'].std():.4f}")
        logging.info(f"Min return: {features_df['future_return'].min():.4f}")
        logging.info(f"Max return: {features_df['future_return'].max():.4f}")
        
        # Apply direction labels
        features_df['direction'] = self.label_direction_binary(features_df['future_return'].values)

        
        # Log class distribution
        class_dist = pd.Series(features_df['direction']).value_counts()
        total_samples = len(features_df)
        logging.info("\nClass distribution:")
        logging.info(f"Class 0 (Sell): {class_dist.get(0, 0)} ({class_dist.get(0, 0)/total_samples*100:.1f}%)")
        logging.info(f"Class 1 (Buy): {class_dist.get(1, 0)} ({class_dist.get(1, 0)/total_samples*100:.1f}%)")
        logging.info(f"Class 2 (Hold): {class_dist.get(2, 0)} ({class_dist.get(2, 0)/total_samples*100:.1f}%)")
        
        return features_df

    def xg_split(self, features_df):
        """Split data into training and testing sets using proper time series split"""
        numeric_columns = features_df.select_dtypes(include=['float64', 'int64']).columns
        columns_to_drop = ['future_return', 'direction', 'open', 'high', 'low', 'close', 'volume']
        X = features_df[numeric_columns].drop(columns=columns_to_drop, errors='ignore')
        y = features_df['direction']
        
        # Log the features being used
        logging.info(f"\nUsing {len(X.columns)} features for training")
        logging.info(f"Features: {list(X.columns)}")
        logging.info(f"Excluded raw OHLCV columns to ensure scale invariance")
        
        # Use TimeSeriesSplit with multiple folds for validation
        n_splits = 5
        test_size = len(X) // 5  # 20% for testing
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        # Evaluate on multiple time series folds
        fold_metrics = []
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_fold_train, X_fold_test = X.iloc[train_index], X.iloc[test_index]
            y_fold_train, y_fold_test = y.iloc[train_index], y.iloc[test_index]

            # ✅ Skip folds without all classes
            if len(np.unique(y_fold_train)) < 3:
                logging.warning(f"Skipping fold {fold + 1} - Only found classes: {np.unique(y_fold_train)}")
                continue

            temp_model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                max_depth=3,
                learning_rate=0.1
            )
            temp_model.fit(X_fold_train, y_fold_train)

        
        logging.info(f"\nCross-validation accuracy: {np.mean(fold_metrics):.4f} (+/- {np.std(fold_metrics):.4f})")
        
        # Use the last split for final evaluation
        train_index, test_index = list(tscv.split(X))[-1]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        logging.info(f"\nFinal split sizes:")
        logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        logging.info("\nClass distribution in final training set:")
        logging.info(pd.Series(y_train).value_counts().sort_index())
        
        return X_train, X_test, y_train, y_test, X, y

    def xg_optuna(self, X_train, y_train):
        def objective(trial):
            """Objective function for XGBoost model"""
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'lambda': trial.suggest_float('lambda', 0.1, 5.0),
                'alpha': trial.suggest_float('alpha', 0.1, 5.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 150)
            }
            
            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                
                model = xgb.XGBClassifier(**params, use_label_encoder=False)
                model.fit(X_fold_train, y_fold_train)
                
                # Get predicted probabilities and convert to class labels
                pred_probs = model.predict_proba(X_fold_val)
                preds = np.argmax(pred_probs, axis=1)
                
                # Calculate balanced accuracy to handle class imbalance
                score = f1_score(y_fold_val, preds, average='weighted')
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
        """Train XGBoost model"""
        best_params = study.best_params
        best_params.update({
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist'
        })
        
        final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
        final_model.fit(X, y)

        # Get predicted probabilities and convert to class labels
        pred_probs = final_model.predict_proba(X_test)
        preds = np.argmax(pred_probs, axis=1)
        
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        conf_matrix = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds)

        logging.info("\n🔍 XGBoost Evaluation:")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Weighted F1 Score: {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        logging.info(f"\nClassification Report:\n{report}")
        
        xgb_path = self.model_dir / f"xgb_classifier_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        with open(xgb_path, 'wb') as f:
            pickle.dump(final_model, f)
        
        logging.info(f"Model saved to {xgb_path}")

        return final_model

    def xg_feature_importance(self, model, X):
        """Feature importance"""
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances_sorted = importances.sort_values(ascending=False)

        plt.figure(figsize=(10,6))
        importances_sorted.head(20).plot(kind='bar')
        plt.title("Top 20 Feature Importances - XGBoost")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig(self.visual_dir / f"feature_importance_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
        plt.close()
        logging.info("Saved feature importance plot")

        return importances_sorted
    
    def refit_with_top_features(self, model, study, X, y, top_n=30):
        """Refit model using only top N features from importance"""
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()
        X_top = X[top_features]

        best_params = study.best_params
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist'
        })

        model_refit = xgb.XGBClassifier(**best_params)
        model_refit.fit(X_top, y)

        # Get predicted probabilities and convert to class labels
        pred_probs = model_refit.predict_proba(X_test)
        preds = np.argmax(pred_probs, axis=1)
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        conf_matrix = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds)

        logging.info("\n🔍 XGBoost Evaluation:")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Weighted F1 Score: {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        logging.info(f"\nClassification Report:\n{report}")

        # Save model (overwrite previous)
        for f in self.model_dir.glob("xgb_classifier_*.pkl"):
            f.unlink()

        model_path = self.model_dir / f"xgb_classifier_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_refit.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_refit, f)
        logging.info(f"Refitted model saved to {model_path}")

        # Save top features
        top_features_path = self.model_dir / f"top_features_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        with open(top_features_path, 'w') as f:
            json.dump(list(X.columns), f)
        logging.info(f"Top features saved to {top_features_path}")

        return model_refit, top_features

    def explain_with_shap(self, model, X):
        """Visualize SHAP feature contributions"""
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        shap.summary_plot(shap_values, X, max_display=30, show=False)
        plt.tight_layout()
        shap_path = self.visual_dir / f"shap_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        plt.savefig(shap_path)
        plt.close()

        logging.info(f"Saved SHAP summary plot to {shap_path}")

if __name__ == "__main__":
    # Load configuration from config.json
    key_path = base_dir / 'config' / 'config.json'
    with open(key_path, "r") as f:
        api_setup = json.load(f)['bitget']

    
    # Initialize Bitget client
    bitget_client = BitgetFutures()

    xg = XGTrainer(bitget_client)

    # 1. Fetch the data
    logging.info("\n===== FETCHING DATA =====")
    data = xg.fetch_data()
    logging.info(f"Fetched {len(data)} data points")

    # data = pd.read_csv(base_dir / 'ohlcv' / '25_02_12_09_54.csv')

    # 2. Calculate features
    logging.info("\n===== CALCULATING FEATURES =====")
    features_df, price_df = xg.calculate_features(data)
    logging.info(f"Calculated features: {features_df.shape}")

    # 3. Target engineering
    logging.info("\n====== TARGET ENGINEERING =====")
    features_df = xg.target_engineering(features_df, price_df)


    #4. XGB Split
    logging.info("\n==== XGB SPLIT =====")
    X_train, X_test, y_train, y_test , X, y = xg.xg_split(features_df)

    #5. XGB Optuna
    logging.info("\n==== XGB OPTUNA =====")
    study = xg.xg_optuna(X_train, y_train)
    
    #6. XGB Final
    logging.info("\n==== XGB FINAL =====")
    final_model = xg.xg_final(X_train, X_test, y_train, y_test, study, X, y)

    #7. XGB Feature Importance
    logging.info("\n==== XGB FEATURE IMPORTANCE =====")
    xg.xg_feature_importance(final_model, X)

    logging.info("\n==== XGB REFIT TOP FEATURES =====")
    final_refit, top_feats = xg.refit_with_top_features(final_model, study, X, y)

    logging.info("\n==== XGB SHAP EXPLAIN =====")
    xg.explain_with_shap(final_refit, X[top_feats])

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
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression , Ridge , Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


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
# Import PCA Predict class
from PCA.pca_predict import PCAPredict

class LRTrainer:
    def __init__(self,bitget_client):
        self.bitget_client = BitgetFutures()
        self.visual_dir = Path(__file__).resolve().parents[1] / 'LR' / 'visual'
    
    def load_fill_pca_model(self):
        pca_predict = PCAPredict(self.bitget_client)
        data = pca_predict.fetch_data()
        features_df, df = pca_predict.calculate_features(data)
        df_filtered = pca_predict.filter_features(features_df)
        pca_model, scaler = pca_predict.load_pca_model_and_scaler()
        features_scaled = scaler.transform(df_filtered)
        pca_features = pca_model.transform(features_scaled)
        pca_df = pd.DataFrame(pca_features[:, :3], columns=['PC1', 'PC2', 'PC3'])
        price_data = features_df[['close']].copy()
        price_data['future_return'] = price_data['close'].pct_change().shift(-1)
        combined_df = pd.concat([pca_df, price_data], axis=1).dropna()
        return combined_df


    def plot_pred_vs_actual(self,y_true, y_pred, model_name):
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.title(f"{model_name} - Predicted vs Actual Returns")
        plt.xlabel("Actual Return")
        plt.ylabel("Predicted Return")
        plt.axhline(0, color='grey', linestyle='--')
        plt.axvline(0, color='grey', linestyle='--')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.visual_dir / f"{model_name}_pred_vs_actual.png")
        plt.show()

    def plot_residuals(self,y_true, y_pred, model_name):
        residuals = y_true - y_pred
        plt.figure(figsize=(6, 4))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.title(f"{model_name} - Residual Plot")
        plt.xlabel("Predicted Return")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.axhline(0, color='red', linestyle='--')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.visual_dir / f"{model_name}_residuals.png")
        plt.show()

    def plot_time_series(self,y_true, y_pred, model_name):
        plt.figure(figsize=(12, 5))
        plt.plot(y_true.reset_index(drop=True), label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.title(f"{model_name} - Return Forecast Over Time")
        plt.xlabel("Time")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.visual_dir / f"{model_name}_time_series.png")
        plt.show()


    def train_and_evaluate_models(self, data):
        features = ['PC1', 'PC2', 'PC3']
        target = 'future_return'
        X = data[features]
        y = data[target]
        X_train, X_test = X.iloc[:-int(0.2 * len(X))], X.iloc[-int(0.2 * len(X)):]
        y_train, y_test = y.iloc[:-int(0.2 * len(y))], y.iloc[-int(0.2 * len(y)) :]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'PolynomialLinearRegression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        }

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            r2 = r2_score(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            logging.info(f"\n📊 {name} Results:")
            logging.info(f"➡️ R²: {r2:.4f}")
            logging.info(f"➡️ MSE: {mse:.6f}")
            with open(base_dir / 'LR' / 'model' / f'{name}.pkl', 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"✅ Saved {name} model.")
            
            # Visualizations
            self.plot_pred_vs_actual(y_test, preds, name)
            self.plot_residuals(y_test, preds, name)
            self.plot_time_series(y_test, preds, name)


if __name__ == "__main__":
    
        # Load configuration from config.json
    key_path = base_dir / 'config' / 'config.json'
    with open(key_path, "r") as f:
        api_setup = json.load(f)['bitget']
    
    # Initialize BitgetFutures client
    bitget_client = BitgetFutures(api_setup)
    
    trainer = LRTrainer(bitget_client)
    data = trainer.load_fill_pca_model()
    trainer.train_and_evaluate_models(data)

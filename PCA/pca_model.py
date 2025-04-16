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


class PCAClass:
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




    def perform_correlation_analysis(self, features_df, price_data):
        """
        Perform Spearman correlation analysis on features against future price returns
        and filter features based on correlation threshold.
        
        Args:
            features_df: DataFrame of calculated features
            price_data: Original price data with OHLCV values
            
        Returns:
            Filtered DataFrame with only the meaningful features
        """
        logging.info("Performing correlation analysis against future returns...")

        #Copy the features and price data
        df_corr = features_df.copy()
        price_df = price_data.copy()

        # Calculate future returns for different periods
        for period in self.future_return_periods:
            price_df[f'future_return_{period}'] = price_df['close'].pct_change(period).shift(-period)
        
        # Add future returns to feature dataframe
        for period in self.future_return_periods:
            df_corr[f'future_return_{period}'] = price_df[f'future_return_{period}']
        
        # Drop NaN values created by future return calculation
        df_corr = df_corr.dropna()

        cols_to_drop = ['timestamp'] + [f'future_return_{p}' for p in self.future_return_periods]+['open', 'high', 'low']
        df_selected = df_corr.drop(columns=cols_to_drop)

        # Calculate Spearman correlation for each feature against each future return period
        correlation_results = {}
        for feature in df_selected.columns:
            if feature not in df_corr.columns:
                continue
            # Calculate correlations for each future return period
            feature_correlations = []
            for period in self.future_return_periods:
                correlation, p_value = stats.spearmanr(
                    df_corr[feature], 
                    df_corr[f'future_return_{period}'],
                    nan_policy='omit'
                )
                feature_correlations.append((period, correlation, p_value))
            
            # Store the results
            correlation_results[feature] = feature_correlations


        # Filter features based on correlation threshold
        filtered_features = []
        for feature, correlations in correlation_results.items():
            # Check if any correlation exceeds threshold (in absolute value)
            for period, corr, p_value in correlations:
                if abs(corr) >= self.correlation_threshold and p_value <= 0.05:
                    filtered_features.append(feature)
                    break  # No need to check other periods once we've selected the feature
        

        # Update the filtered_features attribute
        self.filtered_features = filtered_features
                
        # Log the filtered features
        logging.info(f"Selected {len(filtered_features)} features based on correlation threshold {self.correlation_threshold}:")
        logging.info(", ".join(filtered_features))
        
        # Return the filtered dataframe
        return features_df[filtered_features]
    



    def save(self, filtered_features_df):
        """Save the filtered features to a CSV file"""
        # Create a timestamp string
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_path = base_dir / 'PCA' / 'data' / f'{timestamp}_filtered_features.csv'
        filtered_features_df.to_csv(output_path, index=False)



    def run_pca(self, data):
        """Run PCA on the filtered features"""
        df = data.copy()
        
        # Standardize the features

        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(df))
        data_scaled.columns = df.columns
        data_scaled.index = df.index

        # Perform PCA
        pca = PCA(n_components=self.n_components)
        principal_components  = pca .fit_transform(data_scaled)

        # Create a DataFrame with the principal components
        pca_df = pd.DataFrame(principal_components, columns= [f'PC{i+1}' for i in range(principal_components.shape[1])], index=data_scaled.index)
        print("🔍 PCA Component Loadings:")
        print(pca_df.head())

        #Plot explained variance (optional but useful)
        plt.figure(figsize=(6,4))
        plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
        plt.title('Explained Variance per PCA Component')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        plt.show()

        explained_variance = pca.explained_variance_ratio_

        # Get readable loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=df.columns  # original feature names
        )

        for i in range(pca.n_components_):
            print(f"\n📦 PC{i+1} explains {explained_variance[i]*100:.2f}% of the variance")
            print("Top contributing features:")
            print(loadings.iloc[:, i].abs().sort_values(ascending=False).head(5))
        
        # Save the PCA results
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_path = base_dir / 'PCA' / 'model' / f'PCA_model_{timestamp}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(pca, f)

        # Save the scaler
        output_path = base_dir / 'PCA' / 'model' / f'PCA_scaler_{timestamp}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(scaler, f)



if __name__ == "__main__":
    
    # Load configuration from config.json
    key_path = base_dir / 'config' / 'config.json'
    with open(key_path, "r") as f:
        api_setup = json.load(f)['bitget']
    
    # Initialize BitgetFutures client
    bitget_client = BitgetFutures(api_setup)
    

    pca = PCAClass(bitget_client)

    # 1. Fetch the data
    logging.info("\n===== FETCHING DATA =====")
    data = pca.fetch_data()
    logging.info(f"Fetched {len(data)} data points")
    
    # 2. Calculate features
    logging.info("\n===== CALCULATING FEATURES =====")
    features_df, price_df = pca.calculate_features(data)
    logging.info(f"Calculated features: {features_df.shape}")


    # 3. Perform correlation analysis and filter features
    logging.info("\n===== PERFORMING CORRELATION ANALYSIS =====")
    filtered_features_df = pca.perform_correlation_analysis(features_df, price_df)
    logging.info(f"Filtered features: {filtered_features_df.shape}")
    print(filtered_features_df.head()) 

    #3.5 Remove OHLCV columns
    fully_filtered_features_df = filtered_features_df.drop(columns=['close'])
    print(fully_filtered_features_df.head())

    # 4. Save filtered features to CSV
    pca.save(fully_filtered_features_df)

    #4.5
    plt.figure(figsize=(16, 12))
    sns.heatmap(filtered_features_df.corr(), cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # 5. Run PCA on filtered features
    pca.run_pca(fully_filtered_features_df)


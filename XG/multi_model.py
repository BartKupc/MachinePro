import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import json
import os
from pathlib import Path
import logging
import pickle
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import optuna

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils.feature_calculator import calculate_all_features
from utils.bitget_futures import BitgetFutures

class ModelTrainer:
    def __init__(self, model_dir, visual_dir):
        self.model_dir = model_dir
        self.visual_dir = visual_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model_name, model, X_test, y_test):
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / np.maximum(np.abs(y_test), 1e-8))) * 100

        logging.info(f"\n📊 {model_name} Performance:")
        logging.info(f"RMSE: {rmse:.6f}, R^2: {r2:.6f}, MAE: {mae:.6f}, MAPE: {mape:.2f}%")

        return model_name, r2, rmse, mae

    def save_model(self, model, name):
        model_path = self.model_dir / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"✅ Saved {name} to {model_path}")

    def save_feature_importance(self, model_name, model, X):
        # Save built-in feature importance
        fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fi_df = pd.DataFrame({"Feature": fi.index, f"{model_name}_Importance": fi.values})

        # Save SHAP values
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap_mean = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": X.columns, f"{model_name}_SHAP": shap_mean})

        # Merge and save
        merged = pd.merge(fi_df, shap_df, on="Feature", how="outer").sort_values(by=f"{model_name}_SHAP", ascending=False)
        merged.to_csv(self.model_dir / f"feature_importance_{model_name}.csv", index=False)
        logging.info(f"📄 Saved SHAP + Importance CSV for {model_name}")

    def train_all(self, X_train, X_test, y_train, y_test):
        results = []

        # 1. Random Forest with Optuna
        def rf_objective(trial):
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                random_state=42
            )
            model.fit(X_train, y_train)
            return r2_score(y_test, model.predict(X_test))

        rf_study = optuna.create_study(direction="maximize")
        rf_study.optimize(rf_objective, n_trials=30)
        rf_model = RandomForestRegressor(**rf_study.best_params, random_state=42)
        rf_model.fit(X_train, y_train)
        results.append(self.evaluate("RandomForest", rf_model, X_test, y_test))
        self.save_model(rf_model, "random_forest_reg")
        self.save_feature_importance("RandomForest", rf_model, X_train)

        # 2. XGBoost with Optuna
        def xgb_objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'lambda': trial.suggest_float('lambda', 0.1, 3.0),
                'alpha': trial.suggest_float('alpha', 0.1, 3.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200)
            }
            model = xgb.XGBRegressor(objective='reg:squarederror', **params)
            model.fit(X_train, y_train)
            return r2_score(y_test, model.predict(X_test))

        xgb_study = optuna.create_study(direction="maximize")
        xgb_study.optimize(xgb_objective, n_trials=30)
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', **xgb_study.best_params)
        xgb_model.fit(X_train, y_train)
        results.append(self.evaluate("XGBoost", xgb_model, X_test, y_test))
        self.save_model(xgb_model, "xgb_regressor")
        self.save_feature_importance("XGBoost", xgb_model, X_train)

        # 3. GradientBoosting with Optuna
        def gb_objective(trial):
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
                max_depth=trial.suggest_int("max_depth", 2, 6),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5)
            )
            model.fit(X_train, y_train)
            return r2_score(y_test, model.predict(X_test))

        gb_study = optuna.create_study(direction="maximize")
        gb_study.optimize(gb_objective, n_trials=30)
        gb_model = GradientBoostingRegressor(**gb_study.best_params)
        gb_model.fit(X_train, y_train)
        results.append(self.evaluate("GradientBoosting", gb_model, X_test, y_test))
        self.save_model(gb_model, "gb_regressor")
        self.save_feature_importance("GradientBoosting", gb_model, X_train)

        # 4. LightGBM with Optuna
        def lgb_objective(trial):
            model = lgb.LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                num_leaves=trial.suggest_int("num_leaves", 20, 100),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 30),
                min_split_gain=1e-3
            )
            model.fit(X_train, y_train)
            return r2_score(y_test, model.predict(X_test))

        lgb_study = optuna.create_study(direction="maximize")
        lgb_study.optimize(lgb_objective, n_trials=30)
        lgb_model = lgb.LGBMRegressor(**lgb_study.best_params)
        lgb_model.fit(X_train, y_train)
        results.append(self.evaluate("LightGBM", lgb_model, X_test, y_test))
        self.save_model(lgb_model, "lgb_regressor")
        self.save_feature_importance("LightGBM", lgb_model, X_train)

        # Save results
        df = pd.DataFrame(results, columns=["Model", "R2", "RMSE", "MAE"])
        df.to_csv(self.model_dir / "model_comparison.csv", index=False)
        logging.info("\n📊 Comparison:\n" + str(df))


if __name__ == "__main__":
    bitget_client = BitgetFutures()
    model_dir = Path(__file__).parent / 'model' / 'big'
    visual_dir = Path(__file__).parent / 'visualizations'/ 'big'

    trainer = ModelTrainer(model_dir, visual_dir)

    # Load data
    # data = bitget_client.fetch_ohlcv('ETH/USDT:USDT', '1h', start_time=(datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d'))

    data = pd.read_csv(base_dir / 'ohlcv' / '25_02_12_09_54.csv')


    features_df, price_df = calculate_all_features(data)
    features_df['future_return'] = price_df['close'].pct_change(3).shift(-3)
    features_df.dropna(inplace=True)

    # === Apply log1p on obv_ratio ===
    if 'obv_ratio' in features_df.columns:
        features_df['obv_ratio'] = np.log1p(np.abs(features_df['obv_ratio'].fillna(0)))
        logging.info("📉 Applied log1p transform to obv_ratio")

    # Define top 25 features based on SHAP and importance analysis
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

    # Feature selection
    missing_features = [f for f in top_25_features if f not in features_df.columns]
    if missing_features:
        logging.warning(f"⚠️ Missing features in dataset: {missing_features}")
        available_features = [f for f in top_25_features if f in features_df.columns]
        logging.info(f"Using {len(available_features)} available features out of top 25")
    else:
        logging.info("✅ All top 25 features are available in the dataset")
        available_features = top_25_features

    X = features_df[available_features]
    y = features_df['future_return']

    # Time series split
    n_samples = len(X)
    n_splits = min(5, n_samples // 50)
    if n_splits < 2:
        raise ValueError("Not enough data for at least 2 splits")

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=n_samples // n_splits)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    trainer.train_all(X_train, X_test, y_train, y_test)

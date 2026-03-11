"""
AITrade – Model Training Script
=================================
Trains prediction models on historical stock data from Yahoo Finance.

• If TensorFlow is installed  →  trains LSTM (primary) + Random Forest (backup)
• If TensorFlow is NOT installed →  trains MLP + Random Forest (both sklearn)

Usage
-----
    cd AITrade
    python -m models.train_model --ticker AAPL
    python -m models.train_model --ticker TSLA --epochs 80

Artefacts are saved under  models/saved/<TICKER>/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import normalize_dataframe, create_sequences, SEQUENCE_LENGTH, logger
from models.lstm_model import (
    build_lstm_model,
    build_mlp_model,
    save_model,
    is_tensorflow_available,
)
from api.real_time_data import get_historical_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------------
# Random Forest helper
# ------------------------------------------------------------------
def _train_random_forest(X_train, y_train, X_test, y_test, n_estimators=200):
    X_tr = X_train.reshape(X_train.shape[0], -1)
    X_te = X_test.reshape(X_test.shape[0], -1)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_train)
    preds = rf.predict(X_te)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    logger.info("RF  → MAE=%.4f  RMSE=%.4f  R²=%.4f", mae, rmse, r2)
    return rf, {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def train_models(ticker: str = "AAPL", epochs: int = 50, batch_size: int = 32):
    logger.info("=" * 60)
    logger.info("Training models for %s", ticker)
    logger.info("=" * 60)

    # 1. Data
    df = get_historical_data(ticker, period="5y")
    if df.empty:
        logger.error("No data for %s – aborting.", ticker)
        return None

    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_features = df[feature_cols].copy()

    # 2. Normalise
    df_norm, norm_params = normalize_dataframe(df_features, feature_cols)
    data = df_norm.values.astype(np.float32)

    # 3. Sequences
    X, y = create_sequences(data, SEQUENCE_LENGTH)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    logger.info("Sequences: train=%d  test=%d", len(X_train), len(X_test))

    primary_metrics = {}

    # 4. Primary model
    if is_tensorflow_available():
        logger.info("── Training LSTM (TensorFlow) ──")
        lstm = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        history = lstm.fit(X_train, y_train,
                           validation_data=(X_test, y_test),
                           epochs=epochs, batch_size=batch_size, verbose=1)
        preds = lstm.predict(X_test, verbose=0).flatten()
        primary_model = lstm
        primary_name = "lstm"
        model_ext = "lstm_model.keras"
    else:
        logger.info("── Training MLPRegressor (sklearn) ──")
        mlp = build_mlp_model(max_iter=epochs * 10)
        X_tr_flat = X_train.reshape(X_train.shape[0], -1)
        X_te_flat = X_test.reshape(X_test.shape[0], -1)
        mlp.fit(X_tr_flat, y_train)
        preds = mlp.predict(X_te_flat)
        primary_model = mlp
        primary_name = "mlp"
        model_ext = "mlp_model.pkl"

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    primary_metrics = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
    logger.info("%s → MAE=%.4f  RMSE=%.4f  R²=%.4f", primary_name.upper(), mae, rmse, r2)

    # 5. Random Forest (backup)
    rf_model, rf_metrics = _train_random_forest(X_train, y_train, X_test, y_test)

    # 6. Save
    save_dir = os.path.join(PROJECT_ROOT, "models", "saved", ticker)
    os.makedirs(save_dir, exist_ok=True)

    save_model(primary_model, os.path.join(save_dir, model_ext))
    joblib.dump(rf_model, os.path.join(save_dir, "rf_model.pkl"))
    joblib.dump(norm_params, os.path.join(save_dir, "norm_params.pkl"))
    joblib.dump(
        {primary_name: primary_metrics, "rf": rf_metrics},
        os.path.join(save_dir, "metrics.pkl"),
    )
    joblib.dump(data[-SEQUENCE_LENGTH:], os.path.join(save_dir, "last_sequence.pkl"))

    # Write a flag so the dashboard knows which model type was saved
    joblib.dump(primary_name, os.path.join(save_dir, "model_type.pkl"))

    logger.info("All artefacts saved → %s", save_dir)
    return {
        "primary_model": primary_model,
        "rf_model": rf_model,
        "norm_params": norm_params,
        "metrics": {primary_name: primary_metrics, "rf": rf_metrics},
    }


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train AITrade prediction models")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs/iterations")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (LSTM only)")
    args = parser.parse_args()
    train_models(ticker=args.ticker, epochs=args.epochs, batch_size=args.batch)


if __name__ == "__main__":
    main()

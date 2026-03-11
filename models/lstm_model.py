"""
AITrade – ML Model Definition
================================
Primary : scikit-learn MLPRegressor (neural-net; works on ANY Python version)
Backup  : Random Forest Regressor
Optional: LSTM via TensorFlow/Keras (only if tensorflow is installed)

The module auto-detects TensorFlow.  If absent, sklearn models are used.
"""

import os
import numpy as np
import joblib

from utils.helpers import logger, SEQUENCE_LENGTH

# ---------------------------------------------------------------------------
# Optional TensorFlow
# ---------------------------------------------------------------------------
_HAS_TF = False
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    _HAS_TF = True
    logger.info("TensorFlow %s detected – LSTM available.", tf.__version__)
except ImportError:
    logger.info("TensorFlow not installed – using sklearn MLPRegressor.")

from sklearn.neural_network import MLPRegressor


# =====================================================================
# Builders
# =====================================================================

def build_lstm_model(input_shape: tuple, units: int = 64):
    """Build stacked-LSTM (requires TF). Returns None if TF absent."""
    if not _HAS_TF:
        logger.warning("Cannot build LSTM – TensorFlow missing.")
        return None
    Sequential = tf.keras.models.Sequential
    LSTM = tf.keras.layers.LSTM
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units, return_sequences=True),
        Dropout(0.2),
        LSTM(units // 2, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="mean_squared_error", metrics=["mae"])
    logger.info("LSTM built – params: %s", model.count_params())
    return model


def build_mlp_model(hidden_layers=(128, 64, 32), max_iter=500):
    """Build sklearn MLPRegressor – works everywhere."""
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    logger.info("MLPRegressor built – layers: %s", hidden_layers)
    return model


# =====================================================================
# Save / Load
# =====================================================================

def save_model(model, path: str):
    if _HAS_TF and hasattr(model, "save"):
        model.save(path)
    else:
        joblib.dump(model, path)
    logger.info("Model saved → %s", path)


def load_model(path: str):
    if _HAS_TF and path.endswith((".keras", ".h5")):
        try:
            return tf.keras.models.load_model(path)
        except Exception:
            pass
    return joblib.load(path)


# =====================================================================
# Prediction
# =====================================================================

def predict_next_price(model, recent_data: np.ndarray,
                       scaler_params: dict | None = None) -> float:
    """Predict next-day Close. Works with both keras and sklearn models."""
    if _HAS_TF and hasattr(model, "layers"):
        X = recent_data.reshape(1, recent_data.shape[0], recent_data.shape[1])
        pred_norm = float(model.predict(X, verbose=0)[0][0])
    else:
        X = recent_data.flatten().reshape(1, -1)
        pred_norm = float(model.predict(X)[0])

    if scaler_params:
        cmin, cmax = scaler_params["min"], scaler_params["max"]
        return pred_norm * (cmax - cmin) + cmin
    return pred_norm


def is_tensorflow_available() -> bool:
    return _HAS_TF

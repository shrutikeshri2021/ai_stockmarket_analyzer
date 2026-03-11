"""
AITrade – Utility Helper Functions
===================================
Reusable helpers for data processing, formatting, alerts, and common operations.
"""

import os
import json
import smtplib
import logging
import requests
import numpy as np
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("AITrade")

# ---------------------------------------------------------------------------
# Configuration defaults (override via environment variables)
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
SEQUENCE_LENGTH = 60  # look-back window for LSTM
PREDICTION_DAYS = 1   # how many days ahead to predict
BUY_THRESHOLD = 0.02  # 2 %
SELL_THRESHOLD = -0.02

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def normalize_dataframe(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Min-Max normalise selected columns. Returns (normalised_df, params_dict)."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    params: dict[str, dict] = {}
    df_norm = df.copy()
    for col in columns:
        cmin = df[col].min()
        cmax = df[col].max()
        if cmax - cmin == 0:
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df[col] - cmin) / (cmax - cmin)
        params[col] = {"min": float(cmin), "max": float(cmax)}
    return df_norm, params


def denormalize_value(value: float, col_params: dict) -> float:
    """Reverse min-max scaling for a single value."""
    return value * (col_params["max"] - col_params["min"]) + col_params["min"]


def create_sequences(data: np.ndarray, seq_len: int = SEQUENCE_LENGTH):
    """Create overlapping sequences (X) and labels (y) for time-series models."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i])
        y.append(data[i, 0])  # predict first feature (Close by convention)
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_pct(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value:+.2f}%"


def fmt_currency(value: float, symbol: str = "$") -> str:
    """Format a float as currency."""
    return f"{symbol}{value:,.2f}"


def color_signal(signal: str) -> str:
    """Return a colour hex for a trading signal."""
    mapping = {"BUY": "#00c853", "SELL": "#ff1744", "HOLD": "#ffc107"}
    return mapping.get(signal.upper(), "#ffffff")


def risk_label(score: float) -> str:
    """Convert a 0-100 risk score to a human label."""
    if score < 35:
        return "Low Risk"
    elif score < 65:
        return "Medium Risk"
    return "High Risk"


def risk_color(label: str) -> str:
    mapping = {"Low Risk": "#00c853", "Medium Risk": "#ffc107", "High Risk": "#ff1744"}
    return mapping.get(label, "#ffffff")


# ---------------------------------------------------------------------------
# Alert helpers
# ---------------------------------------------------------------------------

def send_email_alert(subject: str, body: str, to_email: str | None = None):
    """Send an email alert.  Requires env vars SMTP_SERVER, SMTP_PORT,
    EMAIL_USER, EMAIL_PASS, ALERT_EMAIL."""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    email_user = os.getenv("EMAIL_USER", "")
    email_pass = os.getenv("EMAIL_PASS", "")
    to_email = to_email or os.getenv("ALERT_EMAIL", "")
    if not all([email_user, email_pass, to_email]):
        logger.warning("Email alert skipped – credentials not configured.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = email_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(msg)
        logger.info("Email alert sent to %s", to_email)
        return True
    except Exception as exc:
        logger.error("Email alert failed: %s", exc)
        return False


def send_telegram_alert(message: str):
    """Send a Telegram alert.  Requires env vars TELEGRAM_BOT_TOKEN and
    TELEGRAM_CHAT_ID."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not all([token, chat_id]):
        logger.warning("Telegram alert skipped – credentials not configured.")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram alert sent.")
        return True
    except Exception as exc:
        logger.error("Telegram alert failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def safe_division(numerator, denominator, default=0.0):
    """Safe division avoiding ZeroDivisionError."""
    try:
        return numerator / denominator if denominator != 0 else default
    except Exception:
        return default


def get_project_root() -> str:
    """Return absolute path to AITrade project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

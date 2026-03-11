"""
AITrade – Technical Indicators Module
=======================================
Computes RSI, MACD, Moving Averages, Bollinger Bands, Volume Trend,
and an aggregate technical score used in signal generation.
"""

import pandas as pd
import numpy as np

try:
    import pandas_ta as ta  # preferred
    _USE_PANDAS_TA = True
except ImportError:
    _USE_PANDAS_TA = False

from utils.helpers import logger


# =====================================================================
# Individual indicator functions (fallback pure-pandas implementations)
# =====================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    if _USE_PANDAS_TA:
        return ta.rsi(series, length=period)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD line, Signal line, Histogram.
    Returns DataFrame with columns: MACD, Signal, Histogram.
    """
    if _USE_PANDAS_TA:
        macd_df = ta.macd(series, fast=fast, slow=slow, signal=signal)
        if macd_df is not None and not macd_df.empty:
            macd_df.columns = ["MACD", "Histogram", "Signal"]
            return macd_df
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Histogram": histogram})


def compute_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=1).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands: Upper, Middle (SMA), Lower.
    """
    middle = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({"BB_Upper": upper, "BB_Middle": middle, "BB_Lower": lower})


def compute_volume_trend(volume: pd.Series, window: int = 20) -> pd.DataFrame:
    """Volume moving average and relative volume."""
    vol_ma = volume.rolling(window=window, min_periods=1).mean()
    rel_vol = volume / vol_ma.replace(0, np.nan)
    return pd.DataFrame({"Volume_MA": vol_ma, "Relative_Volume": rel_vol})


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range – used for volatility / risk scoring."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


# =====================================================================
# All-in-one indicator calculator
# =====================================================================

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach all technical indicators to a DataFrame that has at minimum
    columns: Open, High, Low, Close, Volume.

    Returns the enriched DataFrame.
    """
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # RSI
    df["RSI"] = compute_rsi(close)

    # MACD
    macd_df = compute_macd(close)
    df["MACD"] = macd_df["MACD"]
    df["MACD_Signal"] = macd_df["Signal"]
    df["MACD_Hist"] = macd_df["Histogram"]

    # Moving averages
    df["MA_20"] = compute_moving_average(close, 20)
    df["MA_50"] = compute_moving_average(close, 50)
    df["MA_200"] = compute_moving_average(close, 200)
    df["EMA_12"] = compute_ema(close, 12)
    df["EMA_26"] = compute_ema(close, 26)

    # Bollinger Bands
    bb = compute_bollinger_bands(close)
    df["BB_Upper"] = bb["BB_Upper"]
    df["BB_Middle"] = bb["BB_Middle"]
    df["BB_Lower"] = bb["BB_Lower"]

    # Volume trend
    vt = compute_volume_trend(volume)
    df["Volume_MA"] = vt["Volume_MA"]
    df["Relative_Volume"] = vt["Relative_Volume"]

    # ATR (volatility)
    df["ATR"] = compute_atr(high, low, close)

    return df


# =====================================================================
# Aggregate technical score  (-100 … +100)
# =====================================================================

def technical_score(df: pd.DataFrame) -> float:
    """
    Compute a composite technical score based on the latest row of
    indicator-enriched DataFrame.  Score ranges from -100 (very bearish)
    to +100 (very bullish).
    """
    if df.empty:
        return 0.0

    latest = df.iloc[-1]
    score = 0.0
    weights_sum = 0.0

    # RSI scoring (weight 20)
    rsi = latest.get("RSI", 50)
    if pd.notna(rsi):
        if rsi < 30:
            score += 20  # oversold → bullish
        elif rsi > 70:
            score -= 20  # overbought → bearish
        else:
            score += (50 - rsi) * (20 / 50)
        weights_sum += 20

    # MACD scoring (weight 25)
    macd_val = latest.get("MACD", 0)
    macd_sig = latest.get("MACD_Signal", 0)
    if pd.notna(macd_val) and pd.notna(macd_sig):
        if macd_val > macd_sig:
            score += 25
        else:
            score -= 25
        weights_sum += 25

    # Price vs MA50 (weight 20)
    price = latest.get("Close", 0)
    ma50 = latest.get("MA_50", 0)
    if pd.notna(ma50) and ma50 > 0:
        pct_above = ((price - ma50) / ma50) * 100
        score += max(min(pct_above * 2, 20), -20)
        weights_sum += 20

    # Price vs MA200 (weight 15)
    ma200 = latest.get("MA_200", 0)
    if pd.notna(ma200) and ma200 > 0:
        pct_above = ((price - ma200) / ma200) * 100
        score += max(min(pct_above * 1.5, 15), -15)
        weights_sum += 15

    # Bollinger Band position (weight 10)
    bb_upper = latest.get("BB_Upper", None)
    bb_lower = latest.get("BB_Lower", None)
    if bb_upper and bb_lower and pd.notna(bb_upper) and pd.notna(bb_lower):
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_pos = (price - bb_lower) / bb_range  # 0..1
            score += (0.5 - bb_pos) * 20  # near lower → bullish
        weights_sum += 10

    # Volume trend (weight 10)
    rel_vol = latest.get("Relative_Volume", 1)
    if pd.notna(rel_vol):
        if rel_vol > 1.5:
            score += 10  # high volume confirms trend
        elif rel_vol < 0.5:
            score -= 5
        weights_sum += 10

    # Normalise to -100 … +100
    if weights_sum > 0:
        return round((score / weights_sum) * 100, 2)
    return 0.0

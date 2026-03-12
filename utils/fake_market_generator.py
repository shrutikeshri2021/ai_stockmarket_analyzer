"""
AITrade – Fake Market Generator
=================================
Generates synthetic OHLCV data for practice stocks so beginners can
learn and trade without touching real markets.

Uses geometric Brownian motion with configurable drift & volatility.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

# =====================================================================
# FAKE STOCK CONFIGURATIONS
# =====================================================================
FAKE_STOCKS = {
    "TECHX": {
        "name": "TechX Corp",
        "sector": "Technology",
        "base_price": 150.0,
        "volatility": 0.024,
        "drift": 0.0003,
        "description": "Leading AI and cloud computing company.",
    },
    "MEDICO": {
        "name": "Medico Health Inc",
        "sector": "Healthcare",
        "base_price": 85.0,
        "volatility": 0.020,
        "drift": 0.0002,
        "description": "Pharmaceutical and biotech giant.",
    },
    "GREENENERGY": {
        "name": "GreenEnergy Ltd",
        "sector": "Energy",
        "base_price": 45.0,
        "volatility": 0.030,
        "drift": 0.0004,
        "description": "Renewable energy and solar technology.",
    },
    "AUTOMAX": {
        "name": "AutoMax Motors",
        "sector": "Automotive",
        "base_price": 200.0,
        "volatility": 0.022,
        "drift": 0.0001,
        "description": "Electric vehicle manufacturer and battery tech.",
    },
    "DIGITALPAY": {
        "name": "DigitalPay Inc",
        "sector": "Finance",
        "base_price": 120.0,
        "volatility": 0.018,
        "drift": 0.0003,
        "description": "Digital payments and fintech platform.",
    },
}


# =====================================================================
# DATA GENERATION
# =====================================================================
def generate_fake_ohlcv(ticker: str, days: int = 252, seed: int | None = None) -> pd.DataFrame:
    """
    Generate realistic OHLCV data using geometric Brownian motion.

    Parameters
    ----------
    ticker : str   – one of the FAKE_STOCKS keys
    days   : int   – number of trading days to generate
    seed   : int   – random seed for reproducibility (default: hash of ticker)

    Returns
    -------
    pd.DataFrame with columns Open, High, Low, Close, Volume and DatetimeIndex.
    """
    if ticker not in FAKE_STOCKS:
        raise ValueError(f"Unknown fake ticker: {ticker}")

    cfg = FAKE_STOCKS[ticker]
    rng = np.random.default_rng(seed if seed is not None else abs(hash(ticker)) % 2**31)

    dt = 1.0 / 252
    drift = cfg["drift"]
    vol = cfg["volatility"]

    # Generate Close prices via GBM
    closes = [cfg["base_price"]]
    for _ in range(days - 1):
        shock = rng.normal()
        new = closes[-1] * np.exp((drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shock)
        closes.append(max(new, 0.50))  # price floor

    closes = np.array(closes)

    # Derive OHLV from Close
    noise_o = 1.0 + rng.normal(0, 0.004, days)
    opens = np.roll(closes, 1) * noise_o
    opens[0] = closes[0] * (1 + rng.normal(0, 0.003))

    highs = np.maximum(opens, closes) * (1.0 + np.abs(rng.normal(0, 0.006, days)))
    lows = np.minimum(opens, closes) * (1.0 - np.abs(rng.normal(0, 0.006, days)))
    volumes = rng.integers(200_000, 6_000_000, size=days)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes.astype(float),
    }, index=dates)
    df.index.name = "Date"
    return df


# =====================================================================
# SCENARIO OVERLAY
# =====================================================================
def apply_scenario(df: pd.DataFrame, price_effect: float = 1.0,
                   volatility_mult: float = 1.0) -> pd.DataFrame:
    """
    Apply a market scenario by scaling prices and increasing volatility
    over the last 60 trading days.
    """
    out = df.copy()
    n = min(60, len(out))
    idx = out.index[-n:]

    # Gradual price shift
    ramp = np.linspace(1.0, price_effect, n)
    for col in ("Open", "High", "Low", "Close"):
        out.loc[idx, col] = out.loc[idx, col] * ramp

    # Volatility noise
    if volatility_mult > 1.0:
        rng = np.random.default_rng(42)
        noise = 1.0 + rng.normal(0, 0.005 * volatility_mult, n)
        out.loc[idx, "Close"] = out.loc[idx, "Close"] * noise
        out.loc[idx, "High"] = out.loc[idx, [("Open"), ("Close")]].max(axis=1) * (
            1.0 + np.abs(rng.normal(0, 0.004 * volatility_mult, n))
        )
        out.loc[idx, "Low"] = out.loc[idx, [("Open"), ("Close")]].min(axis=1) * (
            1.0 - np.abs(rng.normal(0, 0.004 * volatility_mult, n))
        )

    return out


# =====================================================================
# SIMPLE INDICATOR COMPUTATION  (self-contained, no external imports)
# =====================================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, SMA_20, SMA_50, MACD, Signal, BB_upper, BB_lower to *df*."""
    out = df.copy()
    c = out["Close"]

    # SMA
    out["SMA_20"] = c.rolling(20).mean()
    out["SMA_50"] = c.rolling(50).mean()

    # RSI-14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    # Bollinger Bands
    out["BB_Mid"] = out["SMA_20"]
    std20 = c.rolling(20).std()
    out["BB_Upper"] = out["BB_Mid"] + 2 * std20
    out["BB_Lower"] = out["BB_Mid"] - 2 * std20

    return out


# =====================================================================
# FAKE AI PREDICTION
# =====================================================================
def generate_fake_prediction(df: pd.DataFrame) -> float:
    """
    Simple weighted-average prediction for fake stocks.
    Mimics an AI model without actually training one.
    """
    if len(df) < 10:
        return float(df["Close"].iloc[-1])

    recent = df["Close"].tail(10).values
    weights = np.arange(1, 11, dtype=float)
    weighted_avg = np.average(recent, weights=weights)

    # Add slight random nudge based on recent momentum
    momentum = (recent[-1] - recent[0]) / recent[0]
    nudge = momentum * 0.3 + np.random.uniform(-0.005, 0.005)
    pred = weighted_avg * (1 + nudge)
    return round(max(pred, 0.50), 2)

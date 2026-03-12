"""
AITrade – Candlestick Pattern Detection
=========================================
Detects common candlestick patterns from OHLC data.

Supported patterns:
  • Doji              — open ≈ close (indecision)
  • Hammer            — small body at top, long lower shadow (bullish reversal)
  • Inverted Hammer   — small body at bottom, long upper shadow (bullish reversal)
  • Bullish Engulfing — green candle fully engulfs prior red candle (bullish)
  • Bearish Engulfing — red candle fully engulfs prior green candle (bearish)
  • Morning Star      — 3-candle bullish reversal pattern
  • Evening Star      — 3-candle bearish reversal pattern

Usage:
    from indicators.candlestick_patterns import detect_patterns
    results = detect_patterns(df)
    # results → list[dict] with pattern name, type, and description
"""

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("AITrade.candlestick_patterns")


# ──────────────────────────────────────────────────────────────────────
# Helper: body and shadow calculations
# ──────────────────────────────────────────────────────────────────────
def _candle_metrics(row):
    """
    Calculate body size, upper shadow, lower shadow for a single candle.

    Parameters
    ----------
    row : pd.Series or dict
        Must contain 'Open', 'High', 'Low', 'Close'.

    Returns
    -------
    dict with keys: body, upper_shadow, lower_shadow, full_range, is_bullish
    """
    o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
    body = abs(c - o)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    full_range = h - l if h != l else 0.0001  # avoid division by zero
    is_bullish = c >= o
    return {
        "body": body,
        "upper_shadow": upper_shadow,
        "lower_shadow": lower_shadow,
        "full_range": full_range,
        "is_bullish": is_bullish,
    }


# ──────────────────────────────────────────────────────────────────────
# Individual Pattern Detectors
# ──────────────────────────────────────────────────────────────────────
def _is_doji(m, threshold=0.05):
    """
    Doji: body is very small relative to total range.
    Signals market indecision — neither buyers nor sellers dominate.
    """
    return m["body"] / m["full_range"] < threshold


def _is_hammer(m, body_ratio=0.30, shadow_ratio=2.0):
    """
    Hammer: small body near the top, long lower shadow (≥ 2× body).
    Appears in a downtrend → potential bullish reversal.
    """
    if m["body"] / m["full_range"] > body_ratio:
        return False
    if m["body"] == 0:
        return False
    return m["lower_shadow"] >= shadow_ratio * m["body"] and \
           m["upper_shadow"] < m["body"]


def _is_inverted_hammer(m, body_ratio=0.30, shadow_ratio=2.0):
    """
    Inverted Hammer: small body near the bottom, long upper shadow.
    Appears in a downtrend → potential bullish reversal.
    """
    if m["body"] / m["full_range"] > body_ratio:
        return False
    if m["body"] == 0:
        return False
    return m["upper_shadow"] >= shadow_ratio * m["body"] and \
           m["lower_shadow"] < m["body"]


def _is_bullish_engulfing(curr, prev, cm, pm):
    """
    Bullish Engulfing: current green candle's body fully wraps
    the previous red candle's body.
    Strong bullish reversal signal.
    """
    if pm["is_bullish"] or not cm["is_bullish"]:
        return False  # need prev=red, curr=green
    return curr["Open"] <= prev["Close"] and curr["Close"] >= prev["Open"]


def _is_bearish_engulfing(curr, prev, cm, pm):
    """
    Bearish Engulfing: current red candle's body fully wraps
    the previous green candle's body.
    Strong bearish reversal signal.
    """
    if not pm["is_bullish"] or cm["is_bullish"]:
        return False  # need prev=green, curr=red
    return curr["Open"] >= prev["Close"] and curr["Close"] <= prev["Open"]


def _is_morning_star(c3, c2, c1, m3, m2, m1):
    """
    Morning Star (3-candle pattern):
      1. Large red candle
      2. Small-bodied candle (star) that gaps down
      3. Large green candle that closes above midpoint of candle 1
    Bullish reversal.
    """
    # Candle 1 (c3 = 3 bars ago): large bearish
    if m3["is_bullish"] or m3["body"] / m3["full_range"] < 0.4:
        return False
    # Candle 2 (c2 = 2 bars ago): small body (star)
    if m2["body"] / m2["full_range"] > 0.25:
        return False
    # Candle 3 (c1 = 1 bar ago / latest): large bullish
    if not m1["is_bullish"] or m1["body"] / m1["full_range"] < 0.4:
        return False
    # Close of candle 3 should be above midpoint of candle 1
    midpoint = (c3["Open"] + c3["Close"]) / 2
    return c1["Close"] > midpoint


def _is_evening_star(c3, c2, c1, m3, m2, m1):
    """
    Evening Star (3-candle pattern):
      1. Large green candle
      2. Small-bodied candle (star) that gaps up
      3. Large red candle that closes below midpoint of candle 1
    Bearish reversal.
    """
    # Candle 1 (c3 = 3 bars ago): large bullish
    if not m3["is_bullish"] or m3["body"] / m3["full_range"] < 0.4:
        return False
    # Candle 2 (c2 = 2 bars ago): small body (star)
    if m2["body"] / m2["full_range"] > 0.25:
        return False
    # Candle 3 (c1 = latest): large bearish
    if m1["is_bullish"] or m1["body"] / m1["full_range"] < 0.4:
        return False
    # Close of candle 3 should be below midpoint of candle 1
    midpoint = (c3["Open"] + c3["Close"]) / 2
    return c1["Close"] < midpoint


# ──────────────────────────────────────────────────────────────────────
# Main Detection Function
# ──────────────────────────────────────────────────────────────────────
def detect_patterns(df: pd.DataFrame, lookback: int = 1) -> list:
    """
    Scan the most recent candles in the DataFrame for candlestick patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close.
        Should be sorted by date ascending (oldest first).
    lookback : int
        Number of recent candles to check for single-candle patterns.
        Multi-candle patterns always use the latest 2–3 candles.

    Returns
    -------
    list[dict]
        Each dict has:
          - "pattern"     : str  — pattern name
          - "type"        : str  — "bullish", "bearish", or "neutral"
          - "icon"        : str  — emoji icon
          - "description" : str  — short explanation
    """
    detected = []

    try:
        if df is None or len(df) < 3:
            logger.warning("Not enough data for pattern detection (need ≥ 3 rows).")
            return detected

        # Ensure required columns exist
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(df.columns):
            logger.warning("Missing OHLC columns for pattern detection.")
            return detected

        # Get latest 3 candles
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        ml = _candle_metrics(latest)
        mp = _candle_metrics(prev)
        mp2 = _candle_metrics(prev2)

        # ── Single-candle patterns (latest candle) ───────────────
        if _is_doji(ml):
            detected.append({
                "pattern": "Doji",
                "type": "neutral",
                "icon": "➕",
                "description": "Open ≈ Close — market indecision. "
                               "Could signal a trend reversal.",
            })

        if _is_hammer(ml):
            detected.append({
                "pattern": "Hammer",
                "type": "bullish",
                "icon": "🔨",
                "description": "Small body at top with long lower shadow — "
                               "buyers pushed price back up. Bullish reversal signal.",
            })

        if _is_inverted_hammer(ml):
            detected.append({
                "pattern": "Inverted Hammer",
                "type": "bullish",
                "icon": "⬆️",
                "description": "Small body at bottom with long upper shadow — "
                               "buying pressure emerging. Potential bullish reversal.",
            })

        # ── Two-candle patterns ──────────────────────────────────
        if _is_bullish_engulfing(latest, prev, ml, mp):
            detected.append({
                "pattern": "Bullish Engulfing",
                "type": "bullish",
                "icon": "🟢",
                "description": "Green candle fully engulfs prior red candle — "
                               "strong bullish reversal signal.",
            })

        if _is_bearish_engulfing(latest, prev, ml, mp):
            detected.append({
                "pattern": "Bearish Engulfing",
                "type": "bearish",
                "icon": "🔴",
                "description": "Red candle fully engulfs prior green candle — "
                               "strong bearish reversal signal.",
            })

        # ── Three-candle patterns ────────────────────────────────
        if _is_morning_star(prev2, prev, latest, mp2, mp, ml):
            detected.append({
                "pattern": "Morning Star",
                "type": "bullish",
                "icon": "🌅",
                "description": "3-candle bullish reversal — large red, small star, "
                               "large green. Downtrend may be ending.",
            })

        if _is_evening_star(prev2, prev, latest, mp2, mp, ml):
            detected.append({
                "pattern": "Evening Star",
                "type": "bearish",
                "icon": "🌇",
                "description": "3-candle bearish reversal — large green, small star, "
                               "large red. Uptrend may be ending.",
            })

    except Exception as exc:
        logger.error("Candlestick pattern detection failed: %s", exc)

    return detected


# ──────────────────────────────────────────────────────────────────────
# Summary Helper (for dashboard display)
# ──────────────────────────────────────────────────────────────────────
def patterns_summary_text(patterns: list) -> str:
    """
    Return a single-line summary of detected patterns.

    Parameters
    ----------
    patterns : list[dict]
        Output from detect_patterns().

    Returns
    -------
    str
        e.g. "🟢 Bullish Engulfing, 🔨 Hammer" or "No patterns detected"
    """
    if not patterns:
        return "No patterns detected on the latest candles."
    return ", ".join(f'{p["icon"]} {p["pattern"]}' for p in patterns)

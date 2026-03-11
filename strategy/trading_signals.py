"""
AITrade – Trading Signals & Stock Ranking Module
==================================================
Generates BUY / SELL / HOLD signals, confidence scores, risk scores,
and ranks multiple stocks for the dashboard.
"""

import numpy as np
import pandas as pd

from utils.helpers import (
    logger,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    safe_division,
    DEFAULT_TICKERS,
)
from indicators.technical_indicators import calculate_all_indicators, technical_score
from api.real_time_data import get_historical_data, get_live_price
from sentiment.news_sentiment import get_stock_sentiment


# =====================================================================
# Signal generation
# =====================================================================

def generate_signal(
    current_price: float,
    predicted_price: float,
    tech_score: float = 0.0,
    sentiment_score: float = 0.0,
) -> dict:
    """
    Generate a trading signal with confidence.

    Logic
    -----
    1. Price prediction component  (weight 50 %)
    2. Technical score component   (weight 30 %)
    3. Sentiment score component   (weight 20 %)

    Returns dict with keys: signal, confidence, components
    """
    if current_price <= 0:
        return {"signal": "HOLD", "confidence": 0, "components": {}}

    # 1. Prediction delta
    pred_delta = (predicted_price - current_price) / current_price
    pred_score = np.clip(pred_delta / 0.05, -1, 1)  # ±5 % → ±1

    # 2. Tech score (-100..+100) → -1..+1
    tech_norm = np.clip(tech_score / 100, -1, 1)

    # 3. Sentiment (-1..+1) already in range
    sent_norm = np.clip(sentiment_score, -1, 1)

    # Weighted composite
    composite = 0.50 * pred_score + 0.30 * tech_norm + 0.20 * sent_norm

    # Determine signal
    if composite > 0.15:
        signal = "BUY"
    elif composite < -0.15:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = round(min(abs(composite) * 100, 100), 1)

    return {
        "signal": signal,
        "confidence": confidence,
        "composite_score": round(composite, 4),
        "components": {
            "prediction": round(pred_score, 4),
            "technical": round(tech_norm, 4),
            "sentiment": round(sent_norm, 4),
        },
    }


# =====================================================================
# Risk scoring  (0 – 100)
# =====================================================================

def calculate_risk_score(df: pd.DataFrame) -> dict:
    """
    Calculate risk score based on volatility, price variance, and trend.

    Returns dict with: score (0-100), label, components
    """
    if df.empty or len(df) < 30:
        return {"score": 50, "label": "Medium Risk", "components": {}}

    close = df["Close"]

    # 1. Annualised volatility from daily returns
    daily_returns = close.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100  # annualised %

    # 2. Price variance (coefficient of variation)
    cv = (close.std() / close.mean()) * 100 if close.mean() != 0 else 0

    # 3. Recent trend stability (R² of linear fit over last 30 days)
    recent = close.tail(30).values
    x = np.arange(len(recent))
    if len(recent) > 1:
        slope, intercept = np.polyfit(x, recent, 1)
        fitted = slope * x + intercept
        ss_res = np.sum((recent - fitted) ** 2)
        ss_tot = np.sum((recent - recent.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    else:
        r2 = 0
    trend_instability = (1 - max(r2, 0)) * 100  # 0=stable, 100=chaotic

    # 4. Max drawdown (last 90 days)
    window = close.tail(90)
    rolling_max = window.cummax()
    drawdown = ((window - rolling_max) / rolling_max).min() * -100  # positive %

    # Composite risk
    score = (
        0.35 * min(volatility, 100)
        + 0.20 * min(cv, 100)
        + 0.25 * trend_instability
        + 0.20 * min(drawdown, 100)
    )
    score = round(np.clip(score, 0, 100), 1)

    if score < 35:
        label = "Low Risk"
    elif score < 65:
        label = "Medium Risk"
    else:
        label = "High Risk"

    return {
        "score": score,
        "label": label,
        "components": {
            "volatility_pct": round(volatility, 2),
            "coefficient_of_variation": round(cv, 2),
            "trend_instability": round(trend_instability, 2),
            "max_drawdown_pct": round(drawdown, 2),
        },
    }


# =====================================================================
# Stock ranking
# =====================================================================

def rank_stocks(
    tickers: list[str] | None = None,
    predicted_prices: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Analyse and rank multiple stocks.

    For each ticker:
      - Fetch historical data & current price
      - Compute technical score
      - Compute risk score
      - Compute sentiment score
      - Compute predicted growth (if predictions supplied)
      - Create composite ranking score

    Returns a DataFrame sorted by rank.
    """
    tickers = tickers or DEFAULT_TICKERS
    predicted_prices = predicted_prices or {}
    rows = []

    for ticker in tickers:
        try:
            live = get_live_price(ticker)
            if "error" in live:
                continue

            df = get_historical_data(ticker, period="1y")
            if df.empty:
                continue

            df_ind = calculate_all_indicators(df)
            t_score = technical_score(df_ind)
            risk = calculate_risk_score(df)

            # Sentiment (lightweight – may be slow, so catch errors)
            try:
                sentiment = get_stock_sentiment(ticker, max_articles=5)
                sent_score = sentiment["overall_score"]
            except Exception:
                sent_score = 0.0

            current = live["price"]
            predicted = predicted_prices.get(ticker, current)
            pred_growth = ((predicted - current) / current) * 100 if current else 0

            # Composite ranking (higher = better opportunity)
            rank_score = (
                0.35 * np.clip(pred_growth, -20, 20) / 20 * 100
                + 0.25 * (t_score + 100) / 2          # 0-100
                + 0.20 * (1 - risk["score"] / 100) * 100
                + 0.20 * (sent_score + 1) / 2 * 100   # 0-100
            )

            rows.append({
                "Ticker": ticker,
                "Price": current,
                "Predicted": round(predicted, 2),
                "Growth %": round(pred_growth, 2),
                "Tech Score": t_score,
                "Risk Score": risk["score"],
                "Risk Label": risk["label"],
                "Sentiment": round(sent_score, 3),
                "Rank Score": round(rank_score, 2),
            })
        except Exception as exc:
            logger.warning("Ranking %s failed: %s", ticker, exc)
            continue

    if not rows:
        return pd.DataFrame()

    ranking = pd.DataFrame(rows).sort_values("Rank Score", ascending=False).reset_index(drop=True)
    ranking.index += 1  # 1-based rank
    ranking.index.name = "Rank"
    return ranking

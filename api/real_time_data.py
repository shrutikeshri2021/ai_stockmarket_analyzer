"""
AITrade – Real-Time Stock Data Module
======================================
Fetches live and historical stock data via Yahoo Finance (yfinance).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.helpers import logger, DEFAULT_TICKERS


# ---------------------------------------------------------------------------
# Core data fetching
# ---------------------------------------------------------------------------

def get_live_price(ticker_symbol: str) -> dict:
    """
    Return a dict with the latest price snapshot for *ticker_symbol*.

    Keys: symbol, price, open, high, low, volume, previous_close,
          change, change_pct, timestamp
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        hist = ticker.history(period="5d")

        if hist.empty:
            return {"error": f"No data for {ticker_symbol}"}

        latest = hist.iloc[-1]
        prev_close = hist.iloc[-2]["Close"] if len(hist) >= 2 else latest["Close"]

        price = float(latest["Close"])
        change = price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0.0

        return {
            "symbol": ticker_symbol,
            "price": round(price, 2),
            "open": round(float(latest["Open"]), 2),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "volume": int(latest["Volume"]),
            "previous_close": round(float(prev_close), 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "timestamp": str(latest.name),
        }
    except Exception as exc:
        logger.error("get_live_price(%s) failed: %s", ticker_symbol, exc)
        return {"error": str(exc)}


def get_intraday_data(ticker_symbol: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
    """Fetch intraday OHLCV data."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            logger.warning("No intraday data for %s", ticker_symbol)
        return df
    except Exception as exc:
        logger.error("get_intraday_data(%s) failed: %s", ticker_symbol, exc)
        return pd.DataFrame()


def get_historical_data(
    ticker_symbol: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical OHLCV data.
    Default: 2 years of daily bars.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            logger.warning("No historical data for %s", ticker_symbol)
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        return df
    except Exception as exc:
        logger.error("get_historical_data(%s) failed: %s", ticker_symbol, exc)
        return pd.DataFrame()


def get_multiple_tickers_data(
    tickers: list[str] | None = None,
    period: str = "1y",
) -> dict[str, pd.DataFrame]:
    """Download historical data for a list of tickers."""
    tickers = tickers or DEFAULT_TICKERS
    result = {}
    for t in tickers:
        df = get_historical_data(t, period=period)
        if not df.empty:
            result[t] = df
    return result


# ---------------------------------------------------------------------------
# Yesterday performance
# ---------------------------------------------------------------------------

def get_yesterday_performance(ticker_symbol: str) -> dict:
    """
    Analyse yesterday's trading session vs today's open.

    Returns: prev_close, today_open, daily_change_pct, volume_prev,
             volume_today, trend
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5d")
        if len(hist) < 2:
            return {"error": "Insufficient data"}

        today = hist.iloc[-1]
        yesterday = hist.iloc[-2]

        prev_close = float(yesterday["Close"])
        today_open = float(today["Open"])
        daily_change = today_open - prev_close
        daily_change_pct = (daily_change / prev_close) * 100 if prev_close else 0.0

        vol_prev = int(yesterday["Volume"])
        vol_today = int(today["Volume"])

        trend = "Bullish" if daily_change_pct > 0.1 else ("Bearish" if daily_change_pct < -0.1 else "Neutral")

        return {
            "symbol": ticker_symbol,
            "previous_close": round(prev_close, 2),
            "today_open": round(today_open, 2),
            "daily_change": round(daily_change, 2),
            "daily_change_pct": round(daily_change_pct, 2),
            "volume_previous": vol_prev,
            "volume_today": vol_today,
            "volume_change_pct": round(((vol_today - vol_prev) / vol_prev) * 100, 2) if vol_prev else 0,
            "trend": trend,
        }
    except Exception as exc:
        logger.error("get_yesterday_performance(%s) failed: %s", ticker_symbol, exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Company info
# ---------------------------------------------------------------------------

def get_company_info(ticker_symbol: str) -> dict:
    """Return a subset of company metadata."""
    try:
        info = yf.Ticker(ticker_symbol).info
        return {
            "name": info.get("shortName", ticker_symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
        }
    except Exception as exc:
        logger.error("get_company_info(%s) failed: %s", ticker_symbol, exc)
        return {}

"""
AITrade – Earnings Calendar
=============================
Fetches upcoming earnings dates and basic earnings data for a stock
using the yfinance library.

Usage:
    from api.earnings_calendar import get_earnings_info
    info = get_earnings_info("AAPL")
    # info → dict with next_earnings_date, days_until, recent_earnings, etc.
"""

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import logging
from datetime import datetime, date
import yfinance as yf
import pandas as pd

logger = logging.getLogger("AITrade.earnings_calendar")


# ──────────────────────────────────────────────────────────────────────
# Core Function
# ──────────────────────────────────────────────────────────────────────
def get_earnings_info(ticker_symbol: str) -> dict:
    """
    Fetch earnings-related data for a given stock ticker.

    Parameters
    ----------
    ticker_symbol : str
        A valid Yahoo Finance ticker symbol (e.g. "AAPL", "TSLA").

    Returns
    -------
    dict with keys:
        - "next_earnings_date"  : str or None — formatted date string
        - "next_earnings_raw"   : date or None — raw date object
        - "days_until"          : int or None — days until next earnings
        - "earnings_status"     : str — "upcoming", "today", "unknown"
        - "recent_earnings"     : list[dict] — last 4 quarters of EPS data
        - "error"               : str or None — error message if failed
    """
    result = {
        "next_earnings_date": None,
        "next_earnings_raw": None,
        "days_until": None,
        "earnings_status": "unknown",
        "recent_earnings": [],
        "error": None,
    }

    try:
        ticker = yf.Ticker(ticker_symbol)

        # ── 1. Next Earnings Date ────────────────────────────────
        # yfinance exposes earnings dates via the calendar property
        try:
            cal = ticker.calendar
            if cal is not None:
                # calendar can be a dict or DataFrame depending on yfinance version
                earnings_date = None

                if isinstance(cal, dict):
                    # Newer yfinance versions return a dict
                    ed = cal.get("Earnings Date")
                    if ed is not None:
                        if isinstance(ed, list) and len(ed) > 0:
                            earnings_date = ed[0]
                        elif isinstance(ed, (datetime, date)):
                            earnings_date = ed
                elif isinstance(cal, pd.DataFrame):
                    # Older versions return a DataFrame
                    if "Earnings Date" in cal.columns:
                        earnings_date = cal["Earnings Date"].iloc[0]
                    elif "Earnings Date" in cal.index:
                        val = cal.loc["Earnings Date"]
                        if hasattr(val, "iloc"):
                            earnings_date = val.iloc[0]
                        else:
                            earnings_date = val

                if earnings_date is not None:
                    # Convert to date if it's a datetime/Timestamp
                    if isinstance(earnings_date, pd.Timestamp):
                        earnings_date = earnings_date.date()
                    elif isinstance(earnings_date, datetime):
                        earnings_date = earnings_date.date()

                    result["next_earnings_raw"] = earnings_date
                    result["next_earnings_date"] = earnings_date.strftime(
                        "%B %d, %Y"
                    )

                    # Calculate days until
                    today = date.today()
                    delta = (earnings_date - today).days
                    result["days_until"] = delta

                    if delta > 0:
                        result["earnings_status"] = "upcoming"
                    elif delta == 0:
                        result["earnings_status"] = "today"
                    else:
                        # Date is in the past — might be stale data
                        result["earnings_status"] = "past"

        except Exception as cal_exc:
            logger.debug("Calendar fetch failed for %s: %s",
                         ticker_symbol, cal_exc)

        # ── 2. Recent Earnings (last 4 quarters) ─────────────────
        try:
            earnings_hist = ticker.earnings_dates
            if earnings_hist is not None and not earnings_hist.empty:
                # earnings_dates is a DataFrame indexed by date
                # with columns like 'EPS Estimate', 'Reported EPS',
                # 'Surprise(%)'
                recent = earnings_hist.head(4)
                for idx, row in recent.iterrows():
                    q_date = idx
                    if isinstance(q_date, pd.Timestamp):
                        q_date = q_date.strftime("%Y-%m-%d")
                    result["recent_earnings"].append({
                        "date": str(q_date),
                        "eps_estimate": _safe_float(
                            row.get("EPS Estimate")),
                        "eps_reported": _safe_float(
                            row.get("Reported EPS")),
                        "surprise_pct": _safe_float(
                            row.get("Surprise(%)")),
                    })
        except Exception as eh_exc:
            logger.debug("Earnings history fetch failed for %s: %s",
                         ticker_symbol, eh_exc)

    except Exception as exc:
        logger.error("get_earnings_info(%s) failed: %s",
                     ticker_symbol, exc)
        result["error"] = str(exc)

    return result


# ──────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────
def _safe_float(val) -> float:
    """Convert a value to float safely, returning None on failure."""
    if val is None:
        return None
    try:
        import numpy as np
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None

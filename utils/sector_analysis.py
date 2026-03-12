"""
AITrade – Sector Performance Analysis
=======================================
Calculates average daily performance for predefined market sectors
using representative ticker symbols.

Each sector is defined by a small basket of liquid, well-known stocks.
The module fetches live prices and computes the average percentage
change to give a quick snapshot of sector-level momentum.

Usage:
    from utils.sector_analysis import get_sector_performance

    sectors = get_sector_performance()
    # sectors → list of dicts sorted by performance (best first)
    # [{"sector": "Technology", "change_pct": 1.52, "tickers": [...], ...}, ...]
"""

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import logging
import numpy as np

logger = logging.getLogger("AITrade.sector_analysis")


# ──────────────────────────────────────────────────────────────────────
# Sector Definitions
# ──────────────────────────────────────────────────────────────────────
# Each sector maps to a list of representative tickers.
# These are large-cap, liquid stocks that serve as sector proxies.
SECTOR_TICKERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META"],
    "Finance": ["JPM", "BAC", "GS", "V"],
    "Energy": ["XOM", "CVX", "COP"],
    "Healthcare": ["JNJ", "UNH", "PFE"],
    "Consumer": ["AMZN", "WMT", "HD"],
    "Automotive": ["TSLA", "F", "GM"],
    "Telecom": ["T", "VZ", "TMUS"],
}

# Icons for each sector (used in dashboard display)
SECTOR_ICONS = {
    "Technology": "💻",
    "Finance": "🏦",
    "Energy": "⛽",
    "Healthcare": "🏥",
    "Consumer": "🛒",
    "Automotive": "🚗",
    "Telecom": "📡",
}


# ──────────────────────────────────────────────────────────────────────
# Core Function
# ──────────────────────────────────────────────────────────────────────
def get_sector_performance(price_fetcher=None) -> list:
    """
    Calculate the average daily percentage change for each sector.

    Parameters
    ----------
    price_fetcher : callable, optional
        A function that takes a ticker string and returns a dict with
        at least a 'change_pct' key (e.g. get_live_price from
        api/real_time_data.py).  If None, the function will import
        get_live_price automatically.

    Returns
    -------
    list[dict]
        Sorted by change_pct descending (best-performing first).
        Each dict contains:
          - "sector"     : str   — sector name
          - "icon"       : str   — emoji icon
          - "change_pct" : float — average % change across tickers
          - "tickers"    : list  — individual ticker results
          - "count"      : int   — number of tickers with valid data
    """
    # Lazy import to avoid circular dependency
    if price_fetcher is None:
        try:
            from api.real_time_data import get_live_price
            price_fetcher = get_live_price
        except ImportError:
            logger.error("Cannot import get_live_price. "
                         "Pass a price_fetcher function explicitly.")
            return []

    results = []

    for sector, tickers in SECTOR_TICKERS.items():
        changes = []
        ticker_details = []

        for tkr in tickers:
            try:
                data = price_fetcher(tkr)
                if "error" not in data and "change_pct" in data:
                    chg = float(data["change_pct"])
                    changes.append(chg)
                    ticker_details.append({
                        "ticker": tkr,
                        "price": data.get("price", 0),
                        "change_pct": chg,
                    })
                else:
                    logger.debug("No data for %s in sector %s", tkr, sector)
            except Exception as exc:
                logger.warning("Failed to fetch %s for sector %s: %s",
                               tkr, sector, exc)

        # Calculate sector average
        avg_change = float(np.mean(changes)) if changes else 0.0

        results.append({
            "sector": sector,
            "icon": SECTOR_ICONS.get(sector, "📊"),
            "change_pct": round(avg_change, 2),
            "tickers": ticker_details,
            "count": len(changes),
        })

    # Sort by performance: best sector first
    results.sort(key=lambda x: x["change_pct"], reverse=True)

    return results

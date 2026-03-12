"""
AITrade – Trading Journal (Paper Trading Log)
================================================
Allows users to log paper trades manually, calculates profit/loss,
win rate, and maintains a persistent CSV-based trade history.

Storage:  data/trade_journal.csv
Columns:  date, ticker, action, buy_price, sell_price, quantity,
          profit_loss, profit_pct

Usage:
    from utils.trade_journal import add_trade, load_journal, get_journal_stats

    add_trade("AAPL", 185.0, 192.5, 10)   # logs a completed trade
    df    = load_journal()                  # returns full trade history
    stats = get_journal_stats()             # returns summary statistics
"""

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import os
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger("AITrade.trade_journal")

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
# Path to the CSV file that stores all trades
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL_PATH = os.path.join(_PROJECT_ROOT, "data", "trade_journal.csv")

# CSV column definitions
COLUMNS = [
    "date",         # trade date (YYYY-MM-DD HH:MM)
    "ticker",       # stock ticker symbol
    "buy_price",    # entry price per share
    "sell_price",   # exit price per share
    "quantity",     # number of shares
    "profit_loss",  # total profit or loss in dollars
    "profit_pct",   # percentage return on the trade
]


# ──────────────────────────────────────────────────────────────────────
# Ensure CSV Exists
# ──────────────────────────────────────────────────────────────────────
def _ensure_journal_exists():
    """Create the trade journal CSV with headers if it doesn't exist."""
    try:
        os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
        if not os.path.exists(JOURNAL_PATH):
            df = pd.DataFrame(columns=COLUMNS)
            df.to_csv(JOURNAL_PATH, index=False)
            logger.info("Created new trade journal at %s", JOURNAL_PATH)
    except Exception as exc:
        logger.error("Failed to create trade journal: %s", exc)


# ──────────────────────────────────────────────────────────────────────
# Core Functions
# ──────────────────────────────────────────────────────────────────────
def add_trade(
    ticker: str,
    buy_price: float,
    sell_price: float,
    quantity: int,
) -> dict:
    """
    Log a completed paper trade to the journal CSV.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    buy_price : float
        Entry price per share.
    sell_price : float
        Exit price per share.
    quantity : int
        Number of shares traded.

    Returns
    -------
    dict
        The trade record that was saved, including calculated P/L.
    """
    _ensure_journal_exists()

    # Calculate profit/loss
    profit_loss = round((sell_price - buy_price) * quantity, 2)
    profit_pct = round(((sell_price - buy_price) / buy_price) * 100, 2) \
        if buy_price > 0 else 0.0

    trade = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": ticker.upper().strip(),
        "buy_price": round(buy_price, 2),
        "sell_price": round(sell_price, 2),
        "quantity": int(quantity),
        "profit_loss": profit_loss,
        "profit_pct": profit_pct,
    }

    try:
        # Append to CSV
        df_new = pd.DataFrame([trade])
        df_new.to_csv(JOURNAL_PATH, mode="a", header=False, index=False)
        logger.info("Trade logged: %s %s shares of %s → P/L: $%.2f",
                     quantity, ticker, ticker, profit_loss)
    except Exception as exc:
        logger.error("Failed to save trade: %s", exc)
        trade["error"] = str(exc)

    return trade


def load_journal() -> pd.DataFrame:
    """
    Load the full trade journal from CSV.

    Returns
    -------
    pd.DataFrame
        All trades with columns as defined in COLUMNS.
        Returns empty DataFrame if no trades exist.
    """
    _ensure_journal_exists()

    try:
        df = pd.read_csv(JOURNAL_PATH)
        if df.empty:
            return pd.DataFrame(columns=COLUMNS)
        return df
    except Exception as exc:
        logger.error("Failed to load trade journal: %s", exc)
        return pd.DataFrame(columns=COLUMNS)


def delete_trade(index: int) -> bool:
    """
    Delete a trade by its row index.

    Parameters
    ----------
    index : int
        Zero-based row index of the trade to delete.

    Returns
    -------
    bool
        True if deletion was successful, False otherwise.
    """
    try:
        df = load_journal()
        if 0 <= index < len(df):
            df = df.drop(df.index[index]).reset_index(drop=True)
            df.to_csv(JOURNAL_PATH, index=False)
            logger.info("Deleted trade at index %d", index)
            return True
        else:
            logger.warning("Invalid trade index: %d (journal has %d rows)",
                           index, len(df))
            return False
    except Exception as exc:
        logger.error("Failed to delete trade: %s", exc)
        return False


def clear_journal() -> bool:
    """
    Delete all trades from the journal.

    Returns
    -------
    bool
        True if cleared successfully.
    """
    try:
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(JOURNAL_PATH, index=False)
        logger.info("Trade journal cleared.")
        return True
    except Exception as exc:
        logger.error("Failed to clear journal: %s", exc)
        return False


def get_journal_stats() -> dict:
    """
    Calculate summary statistics from the trade journal.

    Returns
    -------
    dict with keys:
        - "total_trades"   : int
        - "total_profit"   : float — sum of all P/L
        - "winning_trades" : int   — trades with P/L > 0
        - "losing_trades"  : int   — trades with P/L < 0
        - "win_rate"       : float — percentage of winning trades
        - "avg_profit"     : float — average P/L per trade
        - "best_trade"     : float — largest single profit
        - "worst_trade"    : float — largest single loss
        - "total_invested" : float — sum of (buy_price * quantity)
    """
    df = load_journal()

    stats = {
        "total_trades": 0,
        "total_profit": 0.0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "avg_profit": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "total_invested": 0.0,
    }

    if df.empty or len(df) == 0:
        return stats

    try:
        total = len(df)
        pl = df["profit_loss"].astype(float)

        stats["total_trades"] = total
        stats["total_profit"] = round(float(pl.sum()), 2)
        stats["winning_trades"] = int((pl > 0).sum())
        stats["losing_trades"] = int((pl < 0).sum())
        stats["win_rate"] = round(
            (stats["winning_trades"] / total) * 100, 1
        ) if total > 0 else 0.0
        stats["avg_profit"] = round(float(pl.mean()), 2)
        stats["best_trade"] = round(float(pl.max()), 2)
        stats["worst_trade"] = round(float(pl.min()), 2)
        stats["total_invested"] = round(
            float((df["buy_price"].astype(float) *
                   df["quantity"].astype(float)).sum()), 2
        )
    except Exception as exc:
        logger.error("Failed to compute journal stats: %s", exc)

    return stats

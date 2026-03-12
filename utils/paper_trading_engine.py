"""
AITrade – Paper Trading Engine
================================
Session-state-backed portfolio manager for beginner simulations.
Supports multiple independent portfolios via a *prefix* parameter
so the Fake Simulator and Paper Trading sections don't collide.
"""

import streamlit as st
from datetime import datetime


# =====================================================================
# INITIALISATION
# =====================================================================
def init_state(prefix: str = "pt", initial_cash: float = 10_000.0):
    """Ensure session-state keys exist for the given *prefix*."""
    k_cash = f"bm_{prefix}_cash"
    k_hold = f"bm_{prefix}_holdings"
    k_hist = f"bm_{prefix}_history"

    if k_cash not in st.session_state:
        st.session_state[k_cash] = initial_cash
    if k_hold not in st.session_state:
        st.session_state[k_hold] = {}          # {ticker: {"shares": n, "avg_cost": x}}
    if k_hist not in st.session_state:
        st.session_state[k_hist] = []           # list of trade dicts


# =====================================================================
# TRADE OPERATIONS
# =====================================================================
def buy(prefix: str, ticker: str, qty: int, price: float):
    """Buy *qty* shares at *price*. Returns (success: bool, message: str)."""
    init_state(prefix)
    k_cash = f"bm_{prefix}_cash"
    k_hold = f"bm_{prefix}_holdings"
    k_hist = f"bm_{prefix}_history"

    cost = qty * price
    if cost > st.session_state[k_cash]:
        return False, "❌ Not enough cash!"
    if qty <= 0:
        return False, "❌ Quantity must be at least 1."

    st.session_state[k_cash] -= cost
    holdings = st.session_state[k_hold]

    if ticker in holdings:
        old = holdings[ticker]
        new_qty = old["shares"] + qty
        new_avg = ((old["avg_cost"] * old["shares"]) + cost) / new_qty
        holdings[ticker] = {"shares": new_qty, "avg_cost": round(new_avg, 2)}
    else:
        holdings[ticker] = {"shares": qty, "avg_cost": round(price, 2)}

    st.session_state[k_hist].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "BUY",
        "ticker": ticker,
        "qty": qty,
        "price": round(price, 2),
        "total": round(cost, 2),
    })
    return True, f"✅ Bought {qty} × {ticker} @ ${price:,.2f}"


def sell(prefix: str, ticker: str, qty: int, price: float):
    """Sell *qty* shares at *price*. Returns (success: bool, message: str)."""
    init_state(prefix)
    k_cash = f"bm_{prefix}_cash"
    k_hold = f"bm_{prefix}_holdings"
    k_hist = f"bm_{prefix}_history"

    holdings = st.session_state[k_hold]
    if ticker not in holdings or holdings[ticker]["shares"] < qty:
        return False, "❌ Not enough shares to sell!"
    if qty <= 0:
        return False, "❌ Quantity must be at least 1."

    proceeds = qty * price
    st.session_state[k_cash] += proceeds
    holdings[ticker]["shares"] -= qty
    if holdings[ticker]["shares"] == 0:
        del holdings[ticker]

    st.session_state[k_hist].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "SELL",
        "ticker": ticker,
        "qty": qty,
        "price": round(price, 2),
        "total": round(proceeds, 2),
    })
    return True, f"✅ Sold {qty} × {ticker} @ ${price:,.2f}"


# =====================================================================
# GETTERS
# =====================================================================
def get_cash(prefix: str) -> float:
    init_state(prefix)
    return st.session_state[f"bm_{prefix}_cash"]


def get_holdings(prefix: str) -> dict:
    init_state(prefix)
    return st.session_state[f"bm_{prefix}_holdings"]


def get_history(prefix: str) -> list:
    init_state(prefix)
    return st.session_state[f"bm_{prefix}_history"]


# =====================================================================
# ANALYTICS
# =====================================================================
def get_portfolio_stats(prefix: str, current_prices: dict,
                        initial_cash: float = 10_000.0) -> dict:
    """
    Compute portfolio-level statistics.

    Parameters
    ----------
    current_prices : dict  – {ticker: latest_price}
    initial_cash   : float – starting balance for return calculation
    """
    init_state(prefix)
    cash = get_cash(prefix)
    holdings = get_holdings(prefix)
    history = get_history(prefix)

    holdings_value = 0.0
    total_pnl = 0.0
    for tkr, info in holdings.items():
        mkt_price = current_prices.get(tkr, info["avg_cost"])
        mkt_val = info["shares"] * mkt_price
        cost_basis = info["shares"] * info["avg_cost"]
        holdings_value += mkt_val
        total_pnl += mkt_val - cost_basis

    total_value = cash + holdings_value
    total_return = ((total_value / initial_cash) - 1) * 100 if initial_cash else 0.0

    wins = sum(1 for t in history if t["action"] == "SELL"
               and any(b["ticker"] == t["ticker"] and b["action"] == "BUY" and b["price"] < t["price"]
                       for b in history))
    sells = sum(1 for t in history if t["action"] == "SELL")
    win_rate = (wins / sells * 100) if sells > 0 else 0.0

    return {
        "cash": round(cash, 2),
        "holdings_value": round(holdings_value, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return": round(total_return, 2),
        "num_trades": len(history),
        "win_rate": round(win_rate, 1),
        "num_holdings": len(holdings),
    }


# =====================================================================
# RESET
# =====================================================================
def reset(prefix: str, initial_cash: float = 10_000.0):
    """Clear all holdings and restore initial cash."""
    st.session_state[f"bm_{prefix}_cash"] = initial_cash
    st.session_state[f"bm_{prefix}_holdings"] = {}
    st.session_state[f"bm_{prefix}_history"] = []

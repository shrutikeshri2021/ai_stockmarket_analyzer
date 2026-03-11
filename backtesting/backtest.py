"""
AITrade – Backtesting Engine
==============================
Simulates a simple long-only trading strategy on historical data and reports
key performance metrics + equity curve data for charting.
"""

import numpy as np
import pandas as pd

from utils.helpers import logger, safe_division
from indicators.technical_indicators import calculate_all_indicators, technical_score


# =====================================================================
# Backtester
# =====================================================================

class Backtester:
    """
    Simple event-driven backtester.

    Strategy
    --------
    Uses a combined signal from technical indicators:
      • tech_score > +20  →  open long (if flat)
      • tech_score < -20  →  close long (if in position)

    Tracks: equity curve, trades, return metrics.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        buy_threshold: float = 20.0,
        sell_threshold: float = -20.0,
        commission_pct: float = 0.001,  # 0.1 %
    ):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.commission = commission_pct

        self.position = 0        # shares held
        self.entry_price = 0.0
        self.trades: list[dict] = []
        self.equity_curve: list[float] = []
        self.dates: list = []

    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute backtest and return results dict.
        """
        # Ensure indicators exist
        if "RSI" not in self.df.columns:
            self.df = calculate_all_indicators(self.df)

        # Use a rolling window to compute technical_score row by row
        close_col = self.df["Close"].values
        n = len(self.df)

        for i in range(60, n):  # need at least 60 rows for indicators
            window = self.df.iloc[max(0, i - 200) : i + 1]
            t_score = technical_score(window)
            price = float(close_col[i])
            date = self.df.iloc[i].get("Date", i)

            # Portfolio value
            portfolio_value = self.capital + self.position * price
            self.equity_curve.append(portfolio_value)
            self.dates.append(date)

            # ---- Signal logic ----
            if self.position == 0 and t_score > self.buy_threshold:
                # BUY
                shares = int(self.capital * 0.95 / price)  # invest 95 % of capital
                if shares > 0:
                    cost = shares * price * (1 + self.commission)
                    self.capital -= cost
                    self.position = shares
                    self.entry_price = price
                    self.trades.append({
                        "type": "BUY",
                        "date": str(date),
                        "price": round(price, 2),
                        "shares": shares,
                        "cost": round(cost, 2),
                    })

            elif self.position > 0 and t_score < self.sell_threshold:
                # SELL
                revenue = self.position * price * (1 - self.commission)
                pnl = revenue - self.position * self.entry_price
                self.capital += revenue
                self.trades.append({
                    "type": "SELL",
                    "date": str(date),
                    "price": round(price, 2),
                    "shares": self.position,
                    "revenue": round(revenue, 2),
                    "pnl": round(pnl, 2),
                })
                self.position = 0
                self.entry_price = 0.0

        # Close any open position at the end
        if self.position > 0:
            final_price = float(close_col[-1])
            revenue = self.position * final_price * (1 - self.commission)
            pnl = revenue - self.position * self.entry_price
            self.capital += revenue
            self.trades.append({
                "type": "SELL (close)",
                "date": str(self.df.iloc[-1].get("Date", n - 1)),
                "price": round(final_price, 2),
                "shares": self.position,
                "revenue": round(revenue, 2),
                "pnl": round(pnl, 2),
            })
            self.position = 0

        return self._compute_metrics()

    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict:
        """Compute performance metrics from trade list and equity curve."""
        final_value = self.capital
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100

        # Win / loss
        sell_trades = [t for t in self.trades if t["type"].startswith("SELL")]
        wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
        losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]
        win_rate = safe_division(len(wins), len(sell_trades)) * 100

        # Max drawdown
        equity = np.array(self.equity_curve) if self.equity_curve else np.array([self.initial_capital])
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak * 100
        max_drawdown = float(drawdowns.min()) if len(drawdowns) else 0.0

        # Sharpe ratio (annualised, assuming daily)
        if len(equity) > 1:
            daily_rets = np.diff(equity) / equity[:-1]
            sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() != 0 else 0
        else:
            sharpe = 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return, 2),
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "dates": self.dates,
        }


# =====================================================================
# Convenience function
# =====================================================================

def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    buy_threshold: float = 20.0,
    sell_threshold: float = -20.0,
) -> dict:
    """Run a backtest on the given DataFrame and return metrics dict."""
    bt = Backtester(
        df,
        initial_capital=initial_capital,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    return bt.run()

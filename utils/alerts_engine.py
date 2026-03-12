"""
AITrade – Price Alert Engine
=============================
Evaluates user-defined alert conditions against live market data.

Supports three alert types:
  1. Price Alert   — triggers when price crosses a user-defined target
  2. RSI Alert     — triggers when RSI drops below a threshold (oversold)
  3. Signal Alert  — triggers when the composite signal equals BUY

Usage:
    from utils.alerts_engine import evaluate_alerts

    alerts = evaluate_alerts(
        current_price=185.50,
        rsi_value=28.3,
        signal="BUY",
        price_alert_enabled=True,
        price_target=180.0,
        price_direction="above",
        rsi_alert_enabled=True,
        rsi_threshold=30,
        signal_alert_enabled=True,
    )
    # alerts → list of dicts: [{"type": "...", "level": "...", "message": "..."}]
"""

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import logging

logger = logging.getLogger("AITrade.alerts_engine")


# ──────────────────────────────────────────────────────────────────────
# Core Evaluation Function
# ──────────────────────────────────────────────────────────────────────
def evaluate_alerts(
    current_price: float,
    rsi_value: float,
    signal: str,
    price_alert_enabled: bool = False,
    price_target: float = 0.0,
    price_direction: str = "above",
    rsi_alert_enabled: bool = False,
    rsi_threshold: float = 30.0,
    signal_alert_enabled: bool = False,
) -> list:
    """
    Evaluate all user-defined alert conditions and return a list of
    triggered alerts.

    Parameters
    ----------
    current_price : float
        The current live stock price.
    rsi_value : float
        The latest RSI (14-period) value.
    signal : str
        The composite trading signal — "BUY", "SELL", or "HOLD".
    price_alert_enabled : bool
        Whether the price alert is active.
    price_target : float
        The target price the user set.
    price_direction : str
        "above" — alert when price > target.
        "below" — alert when price < target.
    rsi_alert_enabled : bool
        Whether the RSI alert is active.
    rsi_threshold : float
        RSI value below which the alert fires (e.g. 30 = oversold).
    signal_alert_enabled : bool
        Whether to alert when signal == "BUY".

    Returns
    -------
    list[dict]
        Each dict has keys:
          - "type"    : str   — "price", "rsi", or "signal"
          - "level"   : str   — "success", "warning", or "error"
                                (maps to Streamlit alert colours)
          - "message" : str   — human-readable alert text
    """
    triggered = []

    # ── 1. Price Alert ───────────────────────────────────────────────
    if price_alert_enabled and price_target > 0:
        try:
            if price_direction == "above" and current_price > price_target:
                triggered.append({
                    "type": "price",
                    "level": "success",
                    "message": (
                        f"🎯 **Price Alert:** {current_price:,.2f} is "
                        f"**above** your target of {price_target:,.2f}"
                    ),
                })
                logger.info("Price alert triggered: %.2f > %.2f",
                            current_price, price_target)

            elif price_direction == "below" and current_price < price_target:
                triggered.append({
                    "type": "price",
                    "level": "error",
                    "message": (
                        f"🎯 **Price Alert:** {current_price:,.2f} is "
                        f"**below** your target of {price_target:,.2f}"
                    ),
                })
                logger.info("Price alert triggered: %.2f < %.2f",
                            current_price, price_target)
        except (TypeError, ValueError) as exc:
            logger.warning("Price alert evaluation failed: %s", exc)

    # ── 2. RSI Alert ─────────────────────────────────────────────────
    if rsi_alert_enabled and rsi_threshold > 0:
        try:
            if rsi_value is not None and rsi_value < rsi_threshold:
                triggered.append({
                    "type": "rsi",
                    "level": "warning",
                    "message": (
                        f"📉 **RSI Alert:** RSI is **{rsi_value:.2f}** — "
                        f"below your threshold of {rsi_threshold:.0f} "
                        f"(potential oversold condition)"
                    ),
                })
                logger.info("RSI alert triggered: %.2f < %.0f",
                            rsi_value, rsi_threshold)
        except (TypeError, ValueError) as exc:
            logger.warning("RSI alert evaluation failed: %s", exc)

    # ── 3. Signal Alert ──────────────────────────────────────────────
    if signal_alert_enabled:
        try:
            if signal == "BUY":
                triggered.append({
                    "type": "signal",
                    "level": "success",
                    "message": (
                        "🚀 **Signal Alert:** Composite signal is **BUY** — "
                        "AI model, technicals, and sentiment are aligned bullish!"
                    ),
                })
                logger.info("Signal alert triggered: signal=BUY")
        except (TypeError, ValueError) as exc:
            logger.warning("Signal alert evaluation failed: %s", exc)

    return triggered


# ──────────────────────────────────────────────────────────────────────
# Alert Message Formatter (for email / Telegram)
# ──────────────────────────────────────────────────────────────────────
def format_alerts_for_notification(alerts: list, ticker: str) -> str:
    """
    Combine a list of triggered alerts into a single HTML-formatted
    message suitable for email or Telegram dispatch.

    Parameters
    ----------
    alerts : list[dict]
        Output from evaluate_alerts().
    ticker : str
        Stock ticker symbol (e.g. "AAPL").

    Returns
    -------
    str
        HTML-formatted notification body, or empty string if no alerts.
    """
    if not alerts:
        return ""

    lines = [f"<b>🚨 AITrade Alerts for {ticker}</b>\n"]
    for a in alerts:
        # Strip markdown bold markers for HTML context
        clean = a["message"].replace("**", "")
        lines.append(f"• {clean}")

    return "\n".join(lines)

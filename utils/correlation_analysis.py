"""
AITrade – Stock Correlation Analyzer
======================================
Computes a pairwise correlation matrix of closing prices across
multiple stock tickers and generates a Plotly heatmap for
dashboard display.

Usage:
    from utils.correlation_analysis import compute_correlation, build_correlation_heatmap

    corr_df = compute_correlation(["AAPL", "TSLA", "MSFT"])
    fig     = build_correlation_heatmap(corr_df)
"""

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger("AITrade.correlation_analysis")


# ──────────────────────────────────────────────────────────────────────
# Core Functions
# ──────────────────────────────────────────────────────────────────────
def compute_correlation(
    tickers: list,
    period: str = "1y",
    data_fetcher=None,
) -> pd.DataFrame:
    """
    Build a correlation matrix of daily closing prices for the
    given tickers.

    Parameters
    ----------
    tickers : list[str]
        List of Yahoo Finance ticker symbols.
    period : str
        Historical period to fetch (e.g. "6mo", "1y", "2y").
    data_fetcher : callable, optional
        A function(ticker, period) → pd.DataFrame with a 'Close' column.
        Defaults to get_historical_data from api/real_time_data.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix (tickers × tickers), values in [-1, 1].
        Returns an empty DataFrame if insufficient data.
    """
    # Lazy import to avoid circular dependency
    if data_fetcher is None:
        try:
            from api.real_time_data import get_historical_data
            data_fetcher = get_historical_data
        except ImportError:
            logger.error("Cannot import get_historical_data.")
            return pd.DataFrame()

    # Collect closing prices into a combined DataFrame
    close_data = {}
    for tkr in tickers:
        try:
            df = data_fetcher(tkr, period=period)
            if df is not None and not df.empty and "Close" in df.columns:
                # Use date as index for alignment
                if "Date" in df.columns:
                    series = df.set_index("Date")["Close"]
                else:
                    series = df["Close"]
                close_data[tkr] = series
            else:
                logger.debug("No close data for %s", tkr)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", tkr, exc)

    if len(close_data) < 2:
        logger.warning("Need at least 2 tickers with data for correlation.")
        return pd.DataFrame()

    # Combine into a single DataFrame, aligning on dates
    combined = pd.DataFrame(close_data)
    combined.dropna(inplace=True)

    if combined.empty or len(combined) < 10:
        logger.warning("Insufficient overlapping data for correlation.")
        return pd.DataFrame()

    # Compute Pearson correlation matrix
    corr_matrix = combined.corr(method="pearson")

    return corr_matrix


def build_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """
    Create an annotated Plotly heatmap from a correlation matrix.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Square correlation matrix (output of compute_correlation).

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive heatmap figure ready for st.plotly_chart().
    """
    tickers = list(corr_df.columns)
    values = corr_df.values

    # Round values for annotation text
    annotations = np.round(values, 2).astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=tickers,
        y=tickers,
        text=annotations,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#fff"),
        colorscale=[
            [0.0, "#ff1744"],    # strong negative correlation → red
            [0.5, "#1a1a2e"],    # zero correlation → dark
            [1.0, "#00e676"],    # strong positive correlation → green
        ],
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title="Correlation",
            titlefont=dict(color="#c9d1d9"),
            tickfont=dict(color="#8b949e"),
        ),
        hovertemplate=(
            "%{y} vs %{x}<br>"
            "Correlation: %{z:.3f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        template="plotly_dark",
        height=max(350, 70 * len(tickers)),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        xaxis=dict(side="bottom", tickfont=dict(size=12, color="#58a6ff")),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color="#58a6ff")),
    )

    return fig


# ──────────────────────────────────────────────────────────────────────
# Summary Helper
# ──────────────────────────────────────────────────────────────────────
def get_top_correlations(corr_df: pd.DataFrame, n: int = 5) -> list:
    """
    Extract the top-N strongest pairwise correlations (excluding self).

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation matrix.
    n : int
        Number of top pairs to return.

    Returns
    -------
    list[dict]
        Each dict: {"pair": "AAPL vs MSFT", "correlation": 0.87}
        Sorted by absolute correlation descending.
    """
    pairs = []
    tickers = list(corr_df.columns)

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            pairs.append({
                "pair": f"{tickers[i]} vs {tickers[j]}",
                "ticker_a": tickers[i],
                "ticker_b": tickers[j],
                "correlation": round(corr_df.iloc[i, j], 4),
            })

    # Sort by absolute correlation (strongest first)
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return pairs[:n]

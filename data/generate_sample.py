"""
AITrade – Sample Data Generator
=================================
Downloads real historical stock data from Yahoo Finance and saves it
as data/historical_data.csv.  Run once before training.

Usage:
    cd AITrade
    python data/generate_sample.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yfinance as yf
import pandas as pd

TICKERS = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def generate():
    os.makedirs(DATA_DIR, exist_ok=True)
    all_frames = []

    for ticker in TICKERS:
        print(f"  Downloading {ticker} …", end=" ", flush=True)
        try:
            df = yf.Ticker(ticker).history(period="5y")
            if df.empty:
                print("NO DATA")
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            df.reset_index(inplace=True)
            df.insert(0, "Ticker", ticker)
            all_frames.append(df)
            print(f"{len(df)} rows ✓")
        except Exception as e:
            print(f"FAILED ({e})")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        out_path = os.path.join(DATA_DIR, "historical_data.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n✅ Saved {len(combined)} total rows → {out_path}")
    else:
        print("⚠️  No data downloaded.")


if __name__ == "__main__":
    print("=" * 50)
    print("AITrade – Downloading historical stock data")
    print("=" * 50)
    generate()

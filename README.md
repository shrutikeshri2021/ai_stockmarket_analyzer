# 📈 AITrade – AI Stock Market Prediction & Advisory System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

</div>

---

## 📌 About

**AITrade** is an AI-powered stock market analysis and prediction system that provides real-time insights, price predictions, and trading signals through an interactive web dashboard.

It combines machine learning, technical analysis, sentiment analysis, and backtesting into a single unified platform — all running locally with no paid API keys required.

> ⚠️ **Disclaimer:** This project is for **educational and research purposes only**. Not financial advice.

---

## ✨ What It Does

- **Predicts next-day stock prices** using a trained neural network model
- **Computes 14 technical indicators** like RSI, MACD, Bollinger Bands, Moving Averages, ATR, etc.
- **Analyzes news sentiment** from live headlines using NLP
- **Generates trading signals** (BUY / SELL / HOLD) with confidence scores
- **Assesses risk** based on volatility, drawdown, and trend stability
- **Backtests strategies** with equity curves, Sharpe ratio, and win rate
- **Scans multiple stocks** simultaneously with a live scanner and rankings
- **Suggests portfolio allocation** across top signals
- **Generates EDA reports** with interactive charts and downloadable PDF/Word documents
- **Auto-refreshes** data for live market monitoring

---

## 🧠 How It Works

The system follows a multi-stage pipeline:

1. **Data Ingestion** — Fetches real-time and historical stock data (OHLCV) from Yahoo Finance
2. **Technical Analysis** — Calculates 14 indicators and produces a technical score
3. **Sentiment Analysis** — Scrapes news headlines and scores them using VADER NLP
4. **ML Prediction** — A trained MLP neural network predicts the next-day closing price
5. **Signal Generation** — Combines prediction, technicals, and sentiment into a composite BUY/SELL/HOLD signal
6. **Risk Assessment** — Evaluates risk from volatility, coefficient of variation, trend instability, and max drawdown

All results are rendered in a **dark-themed Streamlit dashboard** with interactive Plotly charts.

---

## 🖥️ Dashboard Highlights

The dashboard includes **17+ interactive sections**:

- Market overview with live price metrics
- Candlestick chart with overlays (MAs, Bollinger Bands, AI prediction marker)
- RSI, MACD, and Volume charts
- Technical indicators grid
- News sentiment analysis with headline breakdown
- Risk assessment visualization
- Signal breakdown with component weights
- Multi-stock live scanner
- Portfolio allocation suggestions
- Backtesting results with equity curve and trade log
- Stock ranking table
- Company information panel
- Quick comparison bar for all tracked tickers
- Exploratory Data Analysis (EDA) section with 5 charts and downloadable reports

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/supercooledliq/stock_analyzer.git
cd stock_analyzer

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Train Models

```bash
python models/train_model.py
```

This trains prediction models for all default tickers using 5 years of historical data.

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`.

---

## 📊 Supported Stocks

Works with any valid Yahoo Finance ticker. Default tickers include:

- 🇺🇸 AAPL, TSLA, MSFT
- 🇮🇳 RELIANCE.NS, TCS.NS, INFY.NS
- Custom tickers can be entered in the sidebar

---

## 📸 Screenshots

*Dashboard screenshots coming soon.*

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push and open a Pull Request

---

## 📝 License

This project is for **educational and research purposes only**. Not intended for real trading or financial decisions.

---

<div align="center">

### Built with ❤️ by AITrade

⚠️ *Educational purposes only. Not financial advice.*

</div>

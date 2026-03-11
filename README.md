# 📈 AITrade – Intelligent AI Stock Market Prediction & Advisory System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/Yahoo%20Finance-Data-720e9e?logo=yahoo&logoColor=white)
![VADER](https://img.shields.io/badge/VADER-NLP%20Sentiment-green)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

**A full-stack AI-powered stock market analysis and prediction system built with Python, Streamlit, scikit-learn, and real-time Yahoo Finance data.**

[Features](#-features) · [How It Works](#-how-it-works) · [Installation](#-installation) · [Usage](#-usage) · [Project Structure](#-project-structure) · [Dashboard Sections](#-dashboard-sections-explained) · [Tech Stack](#-technology-stack)

</div>

---

## 📌 What Is AITrade?

**AITrade** is an end-to-end AI stock market prediction and advisory system that combines:

- 🤖 **Machine Learning** (MLP Neural Network + Random Forest) to predict next-day stock prices
- 📊 **14 Technical Indicators** (RSI, MACD, Bollinger Bands, Moving Averages, ATR, etc.) for market analysis
- 🧠 **NLP Sentiment Analysis** (VADER) on live Google News headlines to gauge market mood
- 📡 **Real-Time Data** from Yahoo Finance — no API key required
- 🔄 **Backtesting Engine** to simulate historical trading strategies with performance metrics
- 🚦 **Composite Signal Generation** (BUY / SELL / HOLD) combining prediction, technicals, and sentiment
- ⚠️ **Risk Assessment** with volatility, drawdown, and trend instability scoring
- 🏆 **Multi-Stock Ranking & Scanner** for comparing and selecting the best opportunities

All of this is presented through an **interactive, vibrant Streamlit dashboard** with neon-themed dark UI, live charts, and real-time data.

> ⚠️ **Disclaimer:** This project is for **educational and research purposes only**. It is NOT financial advice. Always consult a licensed financial advisor before making investment decisions.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 **AI Price Prediction** | MLP Neural Network trained on 5 years of historical data predicts next-day closing price |
| 📐 **14 Technical Indicators** | RSI, MACD, MA(20/50/200), EMA(12/26), Bollinger Bands, ATR, Volume Trend |
| 🧠 **News Sentiment** | VADER NLP analyzes Google News RSS headlines in real-time |
| 🚦 **Composite Signals** | BUY/SELL/HOLD with confidence % based on weighted scoring formula |
| ⚠️ **Risk Assessment** | 4-component risk score: volatility, CV, trend instability, max drawdown |
| 🔄 **Backtesting** | Simulates tech-score-based strategy with equity curve, Sharpe ratio, win rate |
| 🔴 **Live Scanner** | Scans all 6 default tickers in real-time with predictions and signals |
| 💼 **Portfolio Allocation** | Suggests how to distribute $10,000 across BUY signals by confidence |
| 🏆 **Stock Rankings** | Ranks stocks by composite score combining growth, technicals, risk, sentiment |
| 📋 **Raw Data Preview** | Shows first 30 rows of live OHLCV data from Yahoo Finance |
| 🏢 **Company Info** | Displays sector, industry, market cap, P/E ratio, 52W high/low |
| 📈 **Quick Comparison** | At-a-glance price & change for all 6 tracked stocks |
| ⏱️ **Auto-Refresh** | Optional 60-second auto-refresh for live monitoring |
| 🔔 **Alert System** | Optional email and Telegram alerts on BUY signals |

---

## 📊 Where Does the Data Come From?

| Data Type | Source | Library | API Key? |
|-----------|--------|---------|:--------:|
| **Stock Prices** (real-time + historical OHLCV) | [Yahoo Finance](https://finance.yahoo.com) | `yfinance` | ❌ No |
| **Company Info** (sector, market cap, P/E) | [Yahoo Finance](https://finance.yahoo.com) | `yfinance` | ❌ No |
| **News Headlines** (sentiment analysis) | Google News RSS feed | `feedparser` | ❌ No |
| **News Headlines** (optional, richer) | [NewsAPI.org](https://newsapi.org) | `requests` | ✅ Optional |
| **Sentiment Scoring** | VADER NLP Engine | `vaderSentiment` | ❌ No |

**Everything works out-of-the-box with zero API keys.** NewsAPI is an optional enhancement.

---

## 🧠 How It Works

AITrade operates through a **6-stage pipeline** from data ingestion to signal generation:

### Stage 1: Data Ingestion
```
Yahoo Finance API (yfinance) → OHLCV Data (Open, High, Low, Close, Volume)
```
- Uses the `yfinance` Python library — **no API key required**
- Fetches historical data (up to 5 years) and real-time/delayed prices
- Supports any Yahoo Finance ticker (US stocks, Indian NSE stocks like `RELIANCE.NS`, etc.)
- Also fetches company fundamentals: name, sector, industry, market cap, P/E ratio, 52-week high/low

### Stage 2: Technical Analysis
```
Raw OHLCV Data → 14 Technical Indicators → Technical Score (-100 to +100)
```

The system computes **14 indicators** from price and volume data:

| Indicator | Formula / Method | Purpose |
|-----------|-----------------|---------|
| **RSI (14)** | Relative Strength Index, 14-period | Measures overbought (>70) / oversold (<30) |
| **MACD** | EMA(12) − EMA(26) | Trend direction & momentum |
| **MACD Signal** | EMA(9) of MACD | MACD smoothing for crossover detection |
| **MACD Histogram** | MACD − Signal Line | Momentum strength visualization |
| **MA 20** | Simple Moving Average, 20-period | Short-term trend |
| **MA 50** | Simple Moving Average, 50-period | Medium-term trend |
| **MA 200** | Simple Moving Average, 200-period | Long-term trend |
| **EMA 12** | Exponential Moving Average, 12-period | Fast EMA for MACD |
| **EMA 26** | Exponential Moving Average, 26-period | Slow EMA for MACD |
| **Bollinger Upper** | MA(20) + 2 × StdDev(20) | Upper volatility band |
| **Bollinger Lower** | MA(20) − 2 × StdDev(20) | Lower volatility band |
| **ATR (14)** | Average True Range, 14-period | Volatility measurement |
| **Volume MA** | Volume SMA(20) | Average trading volume |
| **Relative Volume** | Current Volume / Volume MA | Volume confirmation |

**Technical Score Calculation** (weighted, normalized to −100 … +100):

```
Tech Score = Σ(component × weight), clamped to [-100, +100]

Components & Weights:
  • RSI component     → weight 20%  (oversold = bullish, overbought = bearish)
  • MACD component    → weight 25%  (positive histogram = bullish)
  • Price vs MA 50    → weight 20%  (price above MA50 = bullish)
  • Price vs MA 200   → weight 15%  (price above MA200 = bullish)
  • Bollinger Band %B → weight 10%  (position within bands)
  • Volume trend      → weight 10%  (above-average volume confirms trend)
```

**How to interpret the Tech Score:**
- **+50 to +100**: Strong bullish momentum — multiple indicators align upward
- **+1 to +49**: Mild bullish — some indicators positive
- **0**: Neutral — mixed signals
- **−1 to −49**: Mild bearish — some indicators negative
- **−50 to −100**: Strong bearish momentum — multiple indicators align downward

### Stage 3: Sentiment Analysis
```
Google News RSS → Article Headlines → VADER NLP → Sentiment Score (-1.0 to +1.0)
```

1. **News Fetching**: Queries Google News RSS feed (and optionally NewsAPI) for the stock ticker name
2. **VADER Scoring**: Each headline is analyzed using VADER (Valence Aware Dictionary and sEntiment Reasoner)
   - Returns compound score: −1.0 (most negative) to +1.0 (most positive)
   - Classification: **Positive** (≥ 0.05), **Negative** (≤ −0.05), **Neutral** (between)
3. **Aggregation**: Overall sentiment = weighted average of all article compound scores
4. **Output**: Overall score, label, positive/neutral/negative percentages, article list with individual scores

**How VADER works:** VADER is a rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media and news. It uses a lexicon of words rated for positive/negative valence, plus rules for handling negation, emphasis (ALL CAPS), punctuation (!!!), and degree modifiers (very, extremely).

### Stage 4: ML Price Prediction
```
60-day Price Sequences → Trained MLP Neural Network → Predicted Next-Day Close
```

**Model Architecture — MLP (Multi-Layer Perceptron) Regressor:**
```
Input Layer  (60 features — last 60 days of normalized close prices)
     ↓
Hidden Layer 1  (128 neurons, ReLU activation)
     ↓
Hidden Layer 2  (64 neurons, ReLU activation)
     ↓
Hidden Layer 3  (32 neurons, ReLU activation)
     ↓
Output Layer  (1 neuron — predicted normalized price)
     ↓
Denormalization  →  Predicted closing price in dollars
```

**Training Details:**
| Parameter | Value |
|-----------|-------|
| Training Data | 5 years of daily closing prices (Yahoo Finance) |
| Normalization | Min-Max scaling to [0, 1] range |
| Sequence Length | 60 days (each sample = 60 consecutive days → predict day 61) |
| Train/Test Split | 80% training / 20% testing |
| Optimizer | Adam solver |
| Early Stopping | Yes (patience based on validation loss, validation_fraction=0.15) |
| Hidden Layers | (128, 64, 32) |
| Max Iterations | Configurable via `--epochs` flag (default 200) |
| Backup Model | Random Forest Regressor (200 estimators) trained alongside |
| Fallback | If no trained model exists, uses weighted moving average of last 10 days |

**Pre-Trained Models Included For:**
| Ticker | Company | R² Score | Notes |
|--------|---------|----------|-------|
| 🇺🇸 AAPL | Apple Inc. | **0.9438** | Excellent fit on Apple's price history |
| 🇺🇸 TSLA | Tesla Inc. | High | Higher volatility, trained successfully |
| 🇺🇸 MSFT | Microsoft Corp. | High | Stable stock, reliable predictions |
| 🇮🇳 RELIANCE.NS | Reliance Industries | High | Indian market, NSE data |
| 🇮🇳 TCS.NS | Tata Consultancy Services | High | IT sector, NSE data |
| 🇮🇳 INFY.NS | Infosys Ltd. | High | IT sector, NSE data |

Each trained model saves **6 artifacts** per ticker in `models/saved/<TICKER>/`:
| File | Contents |
|------|----------|
| `mlp_model.pkl` | Trained MLP Neural Network (joblib serialized) |
| `rf_model.pkl` | Trained Random Forest backup model |
| `norm_params.pkl` | Min/Max values for each feature (for denormalization) |
| `last_sequence.pkl` | Last 60-day sequence used for live prediction |
| `metrics.pkl` | R², MSE, MAE evaluation metrics |
| `model_type.pkl` | Model type identifier string |

> **Note:** High R² on historical data does not guarantee future accuracy. Stock markets are inherently unpredictable. The model captures historical patterns but cannot foresee black swan events.

### Stage 5: Signal Generation
```
Prediction + Tech Score + Sentiment → Composite Signal (BUY / SELL / HOLD)
```

**Composite Scoring Formula:**
```
composite_score = (0.50 × prediction_component)
                + (0.30 × technical_component)
                + (0.20 × sentiment_component)
```

Where:
- **Prediction component** (50%) = `(predicted_price − current_price) / current_price` — expected percentage price change
- **Technical component** (30%) = `technical_score / 100` — normalized technical indicator score
- **Sentiment component** (20%) = `sentiment_score` — VADER compound score (already −1 to +1)

**Signal Decision Thresholds:**
| Composite Score | Signal | Meaning |
|-----------------|--------|---------|
| > +0.15 | **🟢 BUY** | Strong bullish indicators across prediction, technicals, and sentiment |
| < −0.15 | **🔴 SELL** | Strong bearish indicators — consider exiting position |
| Between | **🟡 HOLD** | Mixed signals — maintain current position |

**Confidence Calculation:**
```
Confidence = min(abs(composite_score) × 200, 100)%
```
Higher deviation from zero = higher confidence in the signal.

### Stage 6: Risk Assessment
```
Historical Price Data → 4 Risk Components → Risk Score (0–100) → Risk Label
```

**Risk Score Formula:**
```
Risk Score = (0.35 × volatility_score)
           + (0.20 × coefficient_of_variation_score)
           + (0.25 × trend_instability_score)
           + (0.20 × max_drawdown_score)
```

| Component | Weight | What It Measures | Calculation |
|-----------|--------|-----------------|-------------|
| **Volatility** | 35% | How much the price swings | Annualized std deviation of daily returns × √252 |
| **Coefficient of Variation** | 20% | Price variability relative to average | StdDev(Close) / Mean(Close) |
| **Trend Instability** | 25% | How consistently the price moves in one direction | Mean of absolute daily returns |
| **Max Drawdown** | 20% | Worst peak-to-trough decline in history | Largest % drop from any peak to subsequent trough |

**Risk Labels:**
| Score Range | Label | Color | Meaning |
|-------------|-------|-------|---------|
| 0 – 30 | **Low Risk** | 🟢 Green | Stable, low volatility, small drawdowns |
| 30 – 60 | **Medium Risk** | 🟡 Yellow | Moderate swings, average volatility |
| 60 – 100 | **High Risk** | 🔴 Red | Highly volatile, large drawdowns possible |

---

## 🛠️ Installation

### Prerequisites
- **Python 3.10+** (tested on Python 3.14.2; TensorFlow is NOT required)
- **pip** (Python package manager)
- **Internet connection** (for Yahoo Finance & Google News data)
- No GPU needed — all models are CPU-based

### Step 1: Clone or Download
```bash
git clone https://github.com/yourusername/AITrade.git
cd AITrade
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**All 13 packages in `requirements.txt`:**
| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `yfinance` | Yahoo Finance data fetching |
| `pandas` | Data manipulation & time series |
| `numpy` | Numerical computations |
| `scikit-learn` | MLP Neural Network & Random Forest models |
| `plotly` | Interactive charts (candlestick, line, bar, pie) |
| `nltk` | Natural language processing utilities |
| `vaderSentiment` | VADER sentiment scoring engine |
| `requests` | HTTP requests (for optional NewsAPI) |
| `feedparser` | Google News RSS feed parsing |
| `joblib` | Model serialization/deserialization |
| `matplotlib` | Backup charting (used during training) |
| `ta` | Technical analysis library (with pure-pandas fallback) |

### Step 3: Download Sample Data (Optional)
```bash
python data/generate_sample.py
```
Downloads 5 years of historical data for all 6 default tickers from Yahoo Finance and saves to `data/historical_data.csv`.

### Step 4: Train ML Models (Optional — pre-trained models are included)
```bash
# Train for a specific ticker
python -m models.train_model --ticker AAPL --epochs 50

# Train for all default tickers
python -m models.train_model --ticker ALL --epochs 50
```

> **Training takes ~30-60 seconds per ticker.** No GPU needed. The dashboard also works without trained models (uses weighted moving average fallback).

### Step 5: Launch the Dashboard
```bash
cd AITrade
streamlit run dashboard/app.py
```

The dashboard opens automatically at **http://localhost:8501** in your browser.

---

## 🚀 Usage

### Dashboard Controls (Sidebar)

| Control | Location | What to Do |
|---------|----------|------------|
| **🔎 Select Stock** | Sidebar dropdown | Choose from AAPL, TSLA, MSFT, RELIANCE.NS, TCS.NS, INFY.NS |
| **Custom Ticker** | Sidebar text input | Type any valid Yahoo Finance ticker (e.g., `GOOGL`, `AMZN`, `META`, `WIPRO.NS`) |
| **📅 Historical Period** | Sidebar dropdown | Choose 6 months, 1 year, 2 years, or 5 years of data |
| **📋 Show Raw Data** | Sidebar checkbox | Toggle first 30 rows of OHLCV data table |
| **🔄 Show Backtesting** | Sidebar checkbox | Toggle backtesting engine section |
| **🏆 Show Stock Rankings** | Sidebar checkbox | Toggle stock ranking comparison |
| **🔴 Real-Time Scanner** | Sidebar checkbox | Toggle multi-stock live scanner |
| **⏱️ Auto Refresh** | Sidebar checkbox | Enable 60-second auto-refresh for live monitoring |
| **🔔 Email Alerts** | Sidebar checkbox | Enable email notifications on BUY signals |
| **🔔 Telegram Alerts** | Sidebar checkbox | Enable Telegram notifications on BUY signals |

### Supported Stock Tickers

Any valid Yahoo Finance ticker works. Pre-configured defaults:

| Market | Example Tickers |
|--------|----------------|
| 🇺🇸 US (NASDAQ/NYSE) | `AAPL`, `TSLA`, `MSFT`, `GOOGL`, `AMZN`, `META`, `NVDA` |
| 🇮🇳 India (NSE) | `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `WIPRO.NS` |
| 🇬🇧 UK (LSE) | `SHEL.L`, `HSBA.L` |
| 🇯🇵 Japan (TSE) | `7203.T` (Toyota) |

Type any ticker in the sidebar's custom ticker input box.

### Custom CSV Data

You can also analyze your own data by providing a CSV file with these required columns:
```
Date, Open, High, Low, Close, Volume
```

---

## 📂 Project Structure

```
AITrade/
│
├── 📄 requirements.txt              # 13 Python dependencies
├── 📄 README.md                      # This comprehensive documentation
│
├── 📁 api/                           # ── Data Fetching Layer ──
│   └── real_time_data.py             # Yahoo Finance API integration (174 lines)
│       ├── get_live_price(ticker)          → Current price, change %, volume
│       ├── get_historical_data(ticker)     → DataFrame of OHLCV history
│       ├── get_intraday_data(ticker)       → Intraday price data
│       ├── get_multiple_tickers_data()     → Batch fetch multiple stocks
│       ├── get_yesterday_performance()     → Previous close vs today open
│       └── get_company_info(ticker)        → Name, sector, market cap, P/E
│
├── 📁 models/                        # ── Machine Learning Layer ──
│   ├── lstm_model.py                 # Model builders & prediction (120 lines)
│   │   ├── build_mlp_model()               → Create sklearn MLPRegressor
│   │   ├── build_lstm_model()              → Create Keras LSTM (if TF installed)
│   │   ├── predict_next_price()            → Run prediction with trained model
│   │   ├── save_model() / load_model()     → Serialize/deserialize models
│   │   └── is_tensorflow_available()       → Auto-detect TF presence
│   │
│   ├── train_model.py                # Training pipeline with CLI (160 lines)
│   │   ├── Downloads 5y data via yfinance
│   │   ├── Normalizes with Min-Max scaling
│   │   ├── Creates 60-day sliding window sequences
│   │   ├── 80/20 train/test split
│   │   ├── Trains MLP (128→64→32) + Random Forest (200 trees)
│   │   └── Saves 6 artifacts per ticker
│   │
│   └── 📁 saved/                     # Trained model artifacts
│       ├── AAPL/                     # ── 6 .pkl files per ticker ──
│       │   ├── mlp_model.pkl
│       │   ├── rf_model.pkl
│       │   ├── norm_params.pkl
│       │   ├── last_sequence.pkl
│       │   ├── metrics.pkl
│       │   └── model_type.pkl
│       ├── TSLA/
│       ├── MSFT/
│       ├── RELIANCE.NS/
│       ├── TCS.NS/
│       └── INFY.NS/
│
├── 📁 indicators/                    # ── Technical Analysis Engine ──
│   └── technical_indicators.py       # 14 indicator functions (219 lines)
│       ├── compute_rsi(df, period=14)
│       ├── compute_macd(df, fast=12, slow=26, signal=9)
│       ├── compute_moving_average(df, window)
│       ├── compute_ema(df, span)
│       ├── compute_bollinger_bands(df, window=20, num_std=2.0)
│       ├── compute_volume_trend(df, window=20)
│       ├── compute_atr(df, period=14)
│       ├── calculate_all_indicators(df)    → Adds all 14 columns to DataFrame
│       └── technical_score(df)             → Returns -100 to +100 score
│
├── 📁 sentiment/                     # ── NLP Sentiment Engine ──
│   └── news_sentiment.py             # Google News + VADER (189 lines)
│       ├── _fetch_google_news(query)       → RSS feed parsing
│       ├── _fetch_newsapi(query)           → Optional NewsAPI (env var)
│       ├── fetch_news(ticker)              → Tries NewsAPI, falls back to Google
│       ├── analyse_sentiment(text)         → VADER compound + label
│       └── get_stock_sentiment(ticker)     → Overall score, %, article list
│
├── 📁 strategy/                      # ── Signal & Risk Logic ──
│   └── trading_signals.py            # Composite signals + ranking (230 lines)
│       ├── generate_signal(price, predicted, tech, sentiment)
│       │     → BUY/SELL/HOLD + confidence + component breakdown
│       ├── calculate_risk_score(df)
│       │     → Score 0-100 + label + 4 component values
│       └── rank_stocks(tickers)
│             → Ranked DataFrame with composite ranking score
│
├── 📁 backtesting/                   # ── Strategy Simulation ──
│   └── backtest.py                   # Backtester class (189 lines)
│       └── Backtester
│             ├── $100K initial capital
│             ├── Tech score > +20 → BUY
│             ├── Tech score < -20 → SELL
│             ├── 0.1% commission per trade
│             ├── Tracks equity curve
│             ├── Computes: total return, win rate, max drawdown, Sharpe
│             └── Generates trade log
│
├── 📁 dashboard/                     # ── Streamlit Web UI ──
│   └── app.py                        # Main dashboard (~870 lines)
│       └── 15+ interactive sections with Plotly charts
│           ├── Vibrant CSS (dark gradient, neon metric cards)
│           ├── Sidebar controls (ticker, period, toggles)
│           ├── Market overview (6 metric cards)
│           ├── Raw data preview (first 30 rows)
│           ├── Price chart (candlestick + MAs + BBs + AI prediction star)
│           ├── RSI chart (overbought/oversold zones)
│           ├── Volume chart (colored bars + MA)
│           ├── MACD chart (line + signal + histogram)
│           ├── Technical indicators grid (14 values)
│           ├── Yesterday performance grid
│           ├── Sentiment analysis (donut chart + headlines)
│           ├── Risk assessment (bar + 4 metrics)
│           ├── Signal breakdown (4 components)
│           ├── Multi-stock scanner (live grid)
│           ├── Portfolio allocation ($10K suggestion)
│           ├── Backtesting (equity curve + trade log)
│           ├── Stock rankings (top 5)
│           ├── Company info (8 fields)
│           ├── Quick comparison bar (all 6 tickers)
│           └── Footer
│
├── 📁 utils/                         # ── Shared Utilities ──
│   └── helpers.py                    # Constants, formatting, alerts (171 lines)
│       ├── DEFAULT_TICKERS = ["AAPL","TSLA","MSFT","RELIANCE.NS","TCS.NS","INFY.NS"]
│       ├── SEQUENCE_LENGTH = 60
│       ├── BUY_THRESHOLD = 0.02
│       ├── SELL_THRESHOLD = -0.02
│       ├── normalize_dataframe()           → Min-Max normalization
│       ├── denormalize_value()             → Reverse normalization
│       ├── create_sequences()              → Sliding window for ML
│       ├── fmt_pct() / fmt_currency()      → Formatting helpers
│       ├── color_signal() / risk_label()   → UI color mapping
│       ├── send_email_alert()              → Gmail SMTP alerts
│       ├── send_telegram_alert()           → Telegram Bot API alerts
│       └── get_project_root()              → Path resolution
│
└── 📁 data/                          # ── Data Storage ──
    ├── generate_sample.py            # Downloads Yahoo Finance data for all tickers
    └── historical_data.csv           # Saved OHLCV data (auto-generated)
```

---

## 🖥️ Dashboard Sections Explained

The dashboard contains **15+ interactive sections**, each providing a different aspect of stock analysis:

---

### 1. 📡 Data Source Banner
**What it shows:** Transparent information about where all data comes from.

**Sources listed:**
- Stock prices (OHLCV) → **Yahoo Finance API** via `yfinance` (no API key)
- News headlines → **Google News RSS** feed (free)
- Sentiment scoring → **VADER NLP** engine
- ML predictions → **scikit-learn MLP + Random Forest** (trained on Yahoo Finance history)

**Why it matters:** Full transparency — you know exactly where every piece of data originates.

---

### 2. 💰 Market Overview (6 Neon Metric Cards)
**What it shows:** Six key metrics displayed in glowing neon-glass styled cards.

| Card | Content | What the Delta Shows |
|------|---------|---------------------|
| 💵 **Current Price** | Live/delayed price from Yahoo Finance | Daily percentage change |
| 🔮 **Predicted Price** | AI-predicted next-day closing price | Expected percentage move vs current |
| 🚦 **Signal** | BUY 🟢 / SELL 🔴 / HOLD 🟡 badge | Confidence percentage |
| ⚠️ **Risk** | Risk label (Low/Medium/High) | Risk score out of 100 |
| 📐 **Tech Score** | Technical score (−100 to +100) | "Bullish 🟢" or "Bearish 🔴" |
| 🧠 **Sentiment** | Overall sentiment label | VADER compound score |

**How to read:** If Current Price < Predicted Price AND Signal = BUY AND Risk = Low → strongest bullish configuration.

---

### 3. 📋 Raw Data Preview (First 30 Rows)
**What it shows:** The first 30 rows of OHLCV data in a styled, scrollable dataframe with green-highlighted Close prices.

**Information displayed above the table:**
- Ticker symbol and historical period selected
- Total number of rows loaded
- Date range (first date → last date)
- Note about Kaggle CSV compatibility

**Why it's useful:** Lets you verify data quality, check the date range, and see actual price/volume numbers before trusting the analysis.

---

### 4. 📊 Price Chart & AI Prediction (3-Panel Interactive Chart)
**What it shows:** A full-height interactive Plotly chart divided into 3 sub-panels:

**Panel 1 — Price & Moving Averages (55% height):**
- 🕯️ **Candlestick chart** — green candles = up days, red candles = down days
- 🩷 **MA 20** (pink dotted) — short-term trend
- 🟡 **MA 50** (yellow solid) — medium-term trend
- 🔵 **MA 200** (blue solid) — long-term trend
- 🟪 **EMA 12** (purple dashed) — fast exponential MA
- 🩵 **EMA 26** (light blue dashed) — slow exponential MA
- 💜 **Bollinger Bands** (purple dotted upper/lower with shaded fill) — volatility envelope
- ⭐ **AI Prediction Star** — large star marker at the predicted next-day price (green if bullish, red if bearish)

**Panel 2 — RSI (25% height):**
- Magenta RSI line
- Red dashed overbought line at 70
- Green dashed oversold line at 30
- Neutral zone shading between 30-70

**Panel 3 — Volume (20% height):**
- Color-coded bars: green = close ≥ open, red = close < open
- Yellow volume moving average line

**How to read the chart:**
- Price above MA 200 + rising MA 50 + RSI between 40-60 turning up + above-average volume = strong bullish setup
- Price below all MAs + RSI > 70 + declining volume = bearish divergence, potential reversal down
- Price touching Bollinger Lower Band + RSI < 30 = potential oversold bounce opportunity

---

### 5. 📉 MACD Indicator (Separate Chart)
**What it shows:**
- 🔵 **MACD line** (blue) — EMA(12) minus EMA(26)
- 🟠 **Signal line** (orange) — EMA(9) of MACD
- 📊 **Histogram bars** (green/red) — MACD minus Signal

**How to read:**
| Pattern | Interpretation |
|---------|---------------|
| MACD crosses above Signal | 🟢 Bullish crossover — upward momentum starting |
| MACD crosses below Signal | 🔴 Bearish crossover — downward momentum starting |
| Growing green histogram | Strengthening bullish momentum |
| Growing red histogram | Strengthening bearish momentum |
| MACD far above zero | Stock may be overbought |
| MACD far below zero | Stock may be oversold |

---

### 6. 📐 Technical Indicators Grid
**What it shows:** A two-column grid displaying the **latest values** of all 14 computed technical indicators with emoji icons.

**Indicators listed:**
RSI (14), MACD, MACD Signal, MACD Histogram, MA 20, MA 50, MA 200, EMA 12, EMA 26, BB Upper, BB Middle, BB Lower, ATR, Relative Volume

**How to read:** Compare current values against standard thresholds. For example:
- RSI > 70 → overbought territory (potential pullback)
- RSI < 30 → oversold territory (potential bounce)
- Price > BB Upper → extended move (may mean-revert)
- Relative Volume > 1.5 → unusual trading activity

---

### 7. Yesterday Performance Grid
**What it shows:** Day-over-day comparison in a styled two-column grid:

| Metric | What It Tells You |
|--------|-------------------|
| ⬅️ Previous Close | Last trading day's closing price |
| ➡️ Today Open | Today's opening price (gap up/down?) |
| 📊 Daily Change | Dollar difference (previous close → today open) |
| 📈 Daily Change % | Percentage change |
| 🔉 Volume Yesterday | Previous day's trading volume |
| 🔊 Volume Today | Today's trading volume |
| 📶 Volume Change % | Volume increase/decrease percentage |
| 🧭 Trend | **Bullish** (green), **Bearish** (red), or **Neutral** (yellow) |

**Why it matters:** Gap ups/downs and volume changes at the open often set the tone for the entire trading day.

---

### 8. 🧠 News Sentiment Analysis
**What it shows:** Two-column layout analyzing market mood from news.

**Left Column:**
- Large sentiment icon: 🟢 Positive / 🟡 Neutral / 🔴 Negative
- Overall VADER compound score (−1.0 to +1.0)
- Three metric cards: Positive %, Neutral %, Negative %
- **Donut chart** — visual breakdown of sentiment distribution

**Right Column:**
- Up to **8 latest news headlines** about the stock
- Each headline has a color-coded sentiment badge: `[Positive]`, `[Neutral]`, or `[Negative]`
- Source attribution (Google News, etc.)

**How it works:** Fetches 10+ recent articles from Google News RSS → runs each headline through VADER → aggregates scores → displays results with interactive visualization.

---

### 9. ⚠️ Risk Assessment
**What it shows:**
- **4 metric cards:** Risk Score (/100), Volatility %, Max Drawdown %, Trend Instability
- **Color-coded progress bar** showing risk level visually (green → yellow → red gradient)

**Example interpretation:**
- Risk 25/100 (Low Risk, green bar) → relatively stable stock with small drawdowns
- Risk 72/100 (High Risk, red bar) → volatile stock with large historical drawdowns — trade with caution

---

### 10. 🚦 Signal Breakdown
**What it shows:** The individual components that make up the composite BUY/SELL/HOLD signal:

| Component | Weight | Metric Shown |
|-----------|--------|-------------|
| 🔮 **Prediction** | 50% | Expected price change ratio from ML model |
| 📐 **Technical** | 30% | Normalized technical indicator score |
| 🧠 **Sentiment** | 20% | News sentiment VADER score |
| 🎯 **Composite** | — | Final weighted sum (determines BUY/SELL/HOLD) |

Also shows whether the prediction used the **✅ Trained MLP/LSTM Model** or the **⚡ Moving Average Fallback** (when no trained model exists for that ticker).

---

### 11. 🔴 Real-Time Multi-Stock Scanner
**What it shows:** A live dashboard scanning all 6 default tickers simultaneously in a CSS Grid layout:

| Column | Description |
|--------|-------------|
| **Ticker** | Stock symbol (blue, bold) |
| **Price** | Current/delayed price |
| **Change** | Daily change % (green/red colored) |
| **Predicted** | AI-predicted next price |
| **Pred Δ** | Predicted change % (green/red colored) |
| **Signal** | BUY/SELL/HOLD badge (rounded, colored) |
| **Conf.** | Signal confidence percentage |
| **Tech** | Technical score (−100 to +100) |
| **Risk** | Risk label (Low/Medium/High, colored) |
| **Volume** | Current trading volume |

**Special Highlights:**
- 🚀 **Top Pick** (green banner): Best BUY signal with highest confidence — includes price, predicted price, and confidence %
- ⚠️ **Avoid** (red banner): Strongest SELL signal — warns you which stock to stay away from

---

### 12. 💼 Portfolio Allocation Recommendation
**What it shows:** A hypothetical **$10,000 portfolio** allocated across current BUY signals.

| Column | Description |
|--------|-------------|
| **Ticker** | Stock with active BUY signal |
| **Price** | Current price per share |
| **Weight** | Allocation percentage (proportional to signal confidence) |
| **Allocation** | Dollar amount to invest |
| **Shares to Buy** | Integer number of whole shares |
| **Invested** | Actual invested amount (shares × price) |

**How it works:** Stocks with BUY signals are allocated proportionally based on their confidence scores. Higher confidence = larger allocation.

**Important:** This is a **hypothetical suggestion**, not financial advice.

---

### 13. 🔄 Backtesting Engine
**What it shows:** Results of simulating the tech-score-based trading strategy on historical data.

**Strategy Rules:**
```
Initial Capital:    $100,000
Commission:         0.1% per trade
BUY when:           Technical Score > +20 (bullish momentum confirmed)
SELL when:          Technical Score < -20 (bearish momentum confirmed)
Position Sizing:    100% of available capital per trade
```

**Performance Metrics:**
| Metric | What It Means |
|--------|--------------|
| 📈 **Total Return** | Overall strategy profit/loss percentage |
| 🎯 **Win Rate** | Percentage of profitable trades |
| 📉 **Max Drawdown** | Worst peak-to-trough equity decline during backtest |
| ⚡ **Sharpe Ratio** | Risk-adjusted return (> 1.0 is good, > 2.0 is excellent) |
| 🔢 **Total Trades** | Number of completed buy-sell round trips |

**Visual Elements:**
- **Equity Curve** — green line chart showing portfolio value over time with fill
- **Initial Capital Line** — yellow dashed reference line at $100,000
- **Expandable Trade Log** — detailed table of every trade with entry/exit prices, dates, and P&L

---

### 14. 🏆 Stock Rankings (Top 5)
**What it shows:** Stocks ranked by composite ranking score in a gradient-styled dataframe.

**Ranking Score** combines:
- Expected growth % (predicted vs current price)
- Technical score (higher = more bullish)
- Risk score (inverted — lower risk = better rank)
- Sentiment score (higher = more positive)

The table uses a **red-to-green gradient** on the Rank Score column — greener scores indicate better opportunities.

---

### 15. 🏢 Company Information
**What it shows:** 8 fundamental data points about the selected stock:

| Row 1 | Row 2 |
|-------|-------|
| 🏛️ Company Name | 📊 P/E Ratio |
| 🏭 Sector | 📈 52-Week High |
| 🔧 Industry | 📉 52-Week Low |
| 💰 Market Cap | 📍 % from 52-Week High |

**Why it's useful:** Provides fundamental context — is the stock near its all-time high? What sector/industry? How expensive (P/E)?

---

### 16. 📈 Quick Comparison Bar
**What it shows:** All 6 tracked stocks side-by-side in compact cards:
- Ticker name (blue, bold)
- Current price
- Daily change % with arrow (🟢 up / 🔴 down)
- Green background for gainers, red background for losers

**Why it's useful:** Instant at-a-glance view of how all your tracked stocks are performing today.

---

## 🔁 Complete Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          USER (Streamlit Dashboard)                      │
│   Select ticker → Choose period → Toggle sections → View results        │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
              ┌────────────▼────────────────┐
              │   api/real_time_data.py       │  ◄── Yahoo Finance (yfinance)
              │   • get_live_price()          │
              │   • get_historical_data()     │
              │   • get_yesterday_performance │
              │   • get_company_info()        │
              └────────────┬────────────────┘
                           │
          ┌────────────────┼────────────────────────┐
          ▼                ▼                        ▼
┌──────────────────┐ ┌────────────────┐ ┌─────────────────────┐
│   indicators/    │ │   models/      │ │   sentiment/        │
│   technical_     │ │   lstm_model   │ │   news_sentiment    │
│   indicators     │ │   (MLP + RF)   │ │   (VADER + RSS)     │
│                  │ │                │ │                     │
│  14 Indicators   │ │  Predict next  │ │  Google News RSS    │
│  Tech Score      │ │  day close     │ │  → VADER scoring    │
│  (-100 … +100)   │ │  price         │ │  (-1.0 … +1.0)     │
└────────┬─────────┘ └───────┬────────┘ └──────────┬──────────┘
         │                   │                     │
         └───────────────────┼─────────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  strategy/trading_signals     │
              │                              │
              │  Composite Score =           │
              │    50% × prediction_delta    │
              │  + 30% × technical_score     │
              │  + 20% × sentiment_score     │
              │                              │
              │  → BUY / SELL / HOLD signal  │
              │  → Confidence %              │
              │  → Risk Score (0-100)        │
              │  → Stock Rankings            │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  backtesting/backtest.py      │
              │                              │
              │  Simulates strategy on       │
              │  historical data             │
              │  → Equity curve              │
              │  → Win rate, Sharpe ratio    │
              │  → Max drawdown              │
              │  → Trade log                 │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  dashboard/app.py             │
              │                              │
              │  Renders all results in      │
              │  15+ interactive sections    │
              │  with Plotly charts &        │
              │  neon-themed dark UI         │
              │  → CSS Grid layouts          │
              │  → Auto-refresh option       │
              │  → Alert dispatch            │
              └──────────────────────────────┘
```

---

## 🔧 Technology Stack

### Languages & Frameworks
| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Web dashboard framework (reactive, no JavaScript needed) |
| **Plotly** | Interactive charts — candlestick, line, bar, donut |
| **Custom CSS** | Neon-themed dark UI, gradient cards, CSS Grid tables |

### Machine Learning
| Library | Purpose |
|---------|---------|
| **scikit-learn** | MLPRegressor (neural network), RandomForestRegressor |
| **joblib** | Model serialization/deserialization (.pkl files) |
| **numpy** | Numerical computations, array operations |
| **pandas** | Data manipulation, time series handling |

### Data Sources
| Source | Library | API Key? | What It Provides |
|--------|---------|:--------:|-----------------|
| **Yahoo Finance** | `yfinance` | ❌ | OHLCV prices, company info, 52W data |
| **Google News RSS** | `feedparser` | ❌ | News headlines for sentiment analysis |
| **NewsAPI** (optional) | `requests` | ✅ | Additional news source (richer data) |

### NLP & Sentiment
| Library | Purpose |
|---------|---------|
| **vaderSentiment** | VADER compound sentiment scoring (−1 to +1) |
| **nltk** | Natural language processing utilities |

### Technical Analysis
| Library | Purpose |
|---------|---------|
| **ta** | Technical analysis library (RSI, MACD, BB) |
| **pandas** | Moving averages, rolling calculations (fallback) |

### Visualization
| Library | Purpose |
|---------|---------|
| **Plotly** | Candlestick, line, bar, donut/pie charts |
| **matplotlib** | Training metrics visualization |
| **CSS Grid** | Custom grid layouts for scanner, portfolio, indicators tables |

> **Note on CSS Grid:** Streamlit's HTML sanitizer strips `<table>`, `<tr>`, `<td>`, `<th>` tags. AITrade works around this by using `<div>` elements with `display: grid` CSS — achieving the same tabular layout without restricted HTML tags.

---

## ⚙️ Configuration

### Environment Variables (All Optional)
| Variable | Default | Purpose |
|----------|---------|---------|
| `NEWSAPI_KEY` | None | NewsAPI key for additional news sources |
| `EMAIL_USER` | None | Gmail address for email alerts |
| `EMAIL_PASS` | None | Gmail app password for email alerts |
| `ALERT_EMAIL` | None | Recipient email address |
| `TELEGRAM_BOT_TOKEN` | None | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | None | Telegram chat ID for alerts |

Set them before running:
```bash
# Windows
set NEWSAPI_KEY=your_key_here
set TELEGRAM_BOT_TOKEN=your_token_here

# Linux/Mac
export NEWSAPI_KEY=your_key_here
export TELEGRAM_BOT_TOKEN=your_token_here
```

### Key Constants (in `utils/helpers.py`)
| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_TICKERS` | `["AAPL","TSLA","MSFT","RELIANCE.NS","TCS.NS","INFY.NS"]` | Default tracked stocks |
| `SEQUENCE_LENGTH` | `60` | Days of history used for each ML prediction |
| `BUY_THRESHOLD` | `0.02` | Minimum predicted gain for basic signal |
| `SELL_THRESHOLD` | `−0.02` | Maximum predicted loss for basic signal |

---

## 🧪 Backtesting Strategy Details

The built-in backtester simulates a **technical-score-based trading strategy**:

```
┌─────────────────────────────────────────────────┐
│  BACKTESTING STRATEGY                           │
│                                                 │
│  Initial Capital:    $100,000                   │
│  Commission:         0.1% per trade             │
│                                                 │
│  ENTRY (BUY):                                   │
│    Technical Score > +20                        │
│    → Bullish momentum confirmed                 │
│    → Invest 100% of available capital           │
│                                                 │
│  EXIT (SELL):                                   │
│    Technical Score < -20                        │
│    → Bearish momentum confirmed                 │
│    → Sell entire position                       │
│                                                 │
│  OUTPUT:                                        │
│    • Total Return %                             │
│    • Win Rate %                                 │
│    • Max Drawdown %                             │
│    • Sharpe Ratio                               │
│    • Trade Count                                │
│    • Equity Curve                               │
│    • Complete Trade Log                         │
└─────────────────────────────────────────────────┘
```

**Interpreting Results:**
| Metric | Good | Excellent |
|--------|------|-----------|
| Total Return | > 0% | > 20% |
| Win Rate | > 50% | > 65% |
| Max Drawdown | < 20% | < 10% |
| Sharpe Ratio | > 1.0 | > 2.0 |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Ideas for Contributions:
- Add LSTM model support (requires TensorFlow)
- Add more technical indicators (Stochastic, Williams %R, ADX)
- Add portfolio optimization (Markowitz, efficient frontier)
- Add more news sources (Twitter/X API, Reddit)
- Add cryptocurrency support
- Add options analysis
- Mobile-responsive CSS improvements

---

## 📝 License

This project is for **educational and research purposes only**. Not intended for real trading or financial decisions.

---

## 🙏 Acknowledgments

- **[Yahoo Finance](https://finance.yahoo.com)** — Free financial data via `yfinance`
- **[Google News](https://news.google.com)** — Free news RSS feeds
- **[VADER Sentiment](https://github.com/cjhutto/vaderSentiment)** — Hutto & Gilbert, 2014
- **[scikit-learn](https://scikit-learn.org)** — Machine learning library
- **[Streamlit](https://streamlit.io)** — Beautiful data app framework
- **[Plotly](https://plotly.com)** — Interactive visualization library
- **[pandas](https://pandas.pydata.org)** — Data analysis toolkit
- **[NumPy](https://numpy.org)** — Numerical computing

---

<div align="center">

### Built with ❤️ by AITrade | v2.0

📡 Data: Yahoo Finance &nbsp;|&nbsp; 🧠 ML: scikit-learn MLP + Random Forest &nbsp;|&nbsp; 📰 News: Google News RSS &nbsp;|&nbsp; 💻 UI: Streamlit + Plotly

⚠️ *Educational purposes only. Not financial advice.*

</div>

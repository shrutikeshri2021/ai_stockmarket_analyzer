"""
AITrade – Intelligent AI Stock Market Prediction & Advisory System
===================================================================
Colourful, high-contrast Streamlit dashboard.

Run:   cd AITrade  &&  streamlit run dashboard/app.py
"""

import os, sys, time, numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st, joblib

# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import (
    DEFAULT_TICKERS, SEQUENCE_LENGTH, fmt_pct, fmt_currency,
    color_signal, risk_label, risk_color,
    send_email_alert, send_telegram_alert, logger,
)
from api.real_time_data import (
    get_live_price, get_historical_data,
    get_yesterday_performance, get_company_info,
)
from indicators.technical_indicators import calculate_all_indicators, technical_score
from sentiment.news_sentiment import get_stock_sentiment
from strategy.trading_signals import generate_signal, calculate_risk_score, rank_stocks
from backtesting.backtest import run_backtest

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="AITrade – AI Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 🎨 VIBRANT CSS  – high contrast, works on light AND dark themes
# =====================================================================
st.markdown("""
<style>
/* ---- Page background ---- */
.stApp {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 40%, #0d1117 100%) !important;
}
.main .block-container { padding-top: 0.8rem; max-width: 1400px; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0533 0%, #0d1b2a 100%) !important;
}
section[data-testid="stSidebar"] * { color: #e0e0ff !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stCheckbox label { color: #a8d8ff !important; font-weight: 600; }

/* ---- Metric cards – neon glass ---- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%) !important;
    border: 1px solid #00e5ff !important;
    border-radius: 14px !important;
    padding: 18px 16px !important;
    box-shadow: 0 0 18px rgba(0,229,255,0.15), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
div[data-testid="stMetric"] label {
    color: #80deea !important; font-size: 0.82rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important; font-size: 1.35rem !important; font-weight: 700 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.9rem !important; }

/* ---- Section headers ---- */
.sec-hdr {
    background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
    padding: 10px 20px; border-radius: 10px; margin: 18px 0 10px 0;
    color: #ffffff; font-size: 1.15rem; font-weight: 700;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(37,117,252,0.3);
}

/* ---- Data source banner ---- */
.data-src {
    background: linear-gradient(90deg, #1b5e20 0%, #004d40 100%);
    padding: 12px 20px; border-radius: 10px; margin: 6px 0 16px 0;
    color: #c8e6c9; font-size: 0.92rem; font-weight: 500;
    border-left: 4px solid #00e676;
}
.data-src b { color: #69f0ae; }

/* ---- Signal badges ---- */
.signal-badge {
    display: inline-block; padding: 8px 28px; border-radius: 30px;
    font-weight: 800; font-size: 1.5rem; letter-spacing: 1px;
    text-align: center; margin: 4px 0;
}
.signal-BUY  { background: linear-gradient(135deg, #00c853, #69f0ae); color: #003300; }
.signal-SELL { background: linear-gradient(135deg, #ff1744, #ff8a80); color: #4a0000; }
.signal-HOLD { background: linear-gradient(135deg, #ffc107, #ffe082); color: #4a3800; }

/* ---- Risk bar ---- */
.risk-bar-wrap {
    background: #1a1a2e; border-radius: 10px; padding: 4px; margin-top: 6px;
    border: 1px solid #333;
}
.risk-bar-fill {
    height: 22px; border-radius: 8px; text-align: center;
    font-size: 0.78rem; font-weight: 700; line-height: 22px;
    transition: width 0.5s ease;
}

/* ---- Tables ---- */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ---- All text visible ---- */
.stApp, .stApp p, .stApp span, .stApp li, .stApp td, .stApp th,
.stApp .stMarkdown, .stApp label { color: #e0e0e0 !important; }
h1, h2, h3 { color: #ffffff !important; }

/* ---- Expander ---- */
details { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 10px !important; }
details summary { color: #58a6ff !important; font-weight: 600; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #21262d; border-radius: 8px 8px 0 0; color: #c9d1d9 !important;
    font-weight: 600; border: 1px solid #30363d;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6a11cb, #2575fc) !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# MODEL LOADER
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_trained_model(ticker: str):
    saved_dir = os.path.join(PROJECT_ROOT, "models", "saved", ticker)
    params_path = os.path.join(saved_dir, "norm_params.pkl")
    seq_path = os.path.join(saved_dir, "last_sequence.pkl")
    model_path = None
    for fname in ("mlp_model.pkl", "lstm_model.keras", "rf_model.pkl"):
        c = os.path.join(saved_dir, fname)
        if os.path.exists(c):
            model_path = c
            break
    if model_path is None:
        return None, None, None
    try:
        from models.lstm_model import load_model as lm
        model = lm(model_path)
        params = joblib.load(params_path) if os.path.exists(params_path) else None
        seq = joblib.load(seq_path) if os.path.exists(seq_path) else None
        return model, params, seq
    except Exception as exc:
        logger.warning("Model load failed for %s: %s", ticker, exc)
        return None, None, None


def predict_with_model(ticker, df):
    model, params, last_seq = load_trained_model(ticker)
    has_model = model is not None and params is not None and last_seq is not None
    if has_model:
        try:
            from models.lstm_model import predict_next_price
            close_params = params.get("Close", {"min": 0, "max": 1})
            pred = predict_next_price(model, last_seq, close_params)
            return round(pred, 2), True
        except Exception:
            pass
    if len(df) >= 10:
        recent = df["Close"].tail(10).values
        weights = np.arange(1, 11)
        pred = np.average(recent, weights=weights) * (1 + np.random.uniform(-0.005, 0.01))
        return round(pred, 2), False
    return round(float(df["Close"].iloc[-1]), 2), False


# =====================================================================
# 📌 SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0;">
        <span style="font-size:2.5rem;">📈</span><br>
        <span style="font-size:1.6rem; font-weight:800; color:#00e5ff;">AITrade</span><br>
        <span style="font-size:0.85rem; color:#80cbc4;">AI Stock Prediction System</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    selected_ticker = st.selectbox("🔎 Select Stock", DEFAULT_TICKERS, index=0)
    custom_ticker = st.text_input("Or type a custom ticker (e.g. GOOGL)", value="", key="custom_tkr")
    # Only override if user typed a valid-looking ticker (letters, dots, dashes)
    ct = custom_ticker.strip().upper()
    if ct and ct.replace(".", "").replace("-", "").isalpha():
        selected_ticker = ct

    st.markdown("---")
    analysis_period = st.selectbox("📅 Historical Period", ["6mo", "1y", "2y", "5y"], index=1)

    st.markdown("---")
    st.markdown("#### ⚙️ Dashboard Sections")
    show_data_preview = st.checkbox("📋 Show Raw Data (first 30 rows)", value=True)
    show_backtest = st.checkbox("🔄 Show Backtesting", value=True)
    show_ranking = st.checkbox("🏆 Show Stock Rankings", value=True)
    show_realtime_scanner = st.checkbox("🔴 Real-Time Multi-Stock Scanner", value=True)
    auto_refresh = st.checkbox("⏱️ Auto refresh (60 s)", value=False)

    st.markdown("---")
    st.markdown("#### 🔔 Alerts")
    enable_email = st.checkbox("Email alerts", value=False)
    enable_telegram = st.checkbox("Telegram alerts", value=False)

    st.markdown("---")
    st.caption("Built with ❤️ by AITrade | v2.0")

# =====================================================================
# Auto-refresh
# =====================================================================
if auto_refresh:
    if "refresh_counter" not in st.session_state:
        st.session_state.refresh_counter = 0

# =====================================================================
# 🏠 HEADER
# =====================================================================
st.markdown("""
<div style="text-align:center; padding: 10px 0 4px 0;">
    <h1 style="margin:0; font-size:2.3rem;
        background: linear-gradient(90deg, #00e5ff, #6a11cb, #ff6d00);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 900;">
        📈 AITrade – AI Stock Prediction Dashboard
    </h1>
    <p style="color:#8b949e; margin-top:4px; font-size:1rem;">
        Real-time analysis for <b style="color:#58a6ff; font-size:1.15rem;">""" + selected_ticker + """</b>
        &nbsp;|&nbsp; Powered by Yahoo Finance + ML
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 📡 DATA SOURCE BANNER
# =====================================================================
st.markdown("""
<div class="data-src">
    📡 <b>Data Sources:</b> &nbsp;
    Stock prices (OHLCV) → <b>Yahoo Finance API</b> (yfinance, no API key needed) &nbsp;|&nbsp;
    News → <b>Google News RSS</b> (free) &nbsp;|&nbsp;
    Sentiment → <b>VADER NLP</b> &nbsp;|&nbsp;
    ML Model → <b>scikit-learn MLP + Random Forest</b> (trained on Yahoo Finance historical data)
</div>
""", unsafe_allow_html=True)

# =====================================================================
# LOAD DATA
# =====================================================================
@st.cache_data(ttl=300, show_spinner="Fetching market data …")
def load_data(ticker, period):
    live = get_live_price(ticker)
    hist = get_historical_data(ticker, period=period)
    yest = get_yesterday_performance(ticker)
    info = get_company_info(ticker)
    return live, hist, yest, info

with st.spinner(f"🔄 Loading {selected_ticker} data from Yahoo Finance …"):
    live_data, hist_df, yesterday, company = load_data(selected_ticker, analysis_period)

if "error" in live_data or hist_df.empty:
    st.error(f"⚠️ Could not fetch data for **{selected_ticker}**. "
             f"Please select a valid ticker from the sidebar dropdown or type a correct one like `AAPL`, `TSLA`, `RELIANCE.NS`.")
    st.info("💡 **Tip:** The custom ticker box only accepts stock symbols (letters & dots). Numbers like '56' are not valid tickers.")
    st.stop()

# Compute
df_ind = calculate_all_indicators(hist_df)
predicted_price, model_used = predict_with_model(selected_ticker, hist_df)
current_price = live_data["price"]
t_score = technical_score(df_ind)

@st.cache_data(ttl=600, show_spinner="Analysing news sentiment …")
def cached_sentiment(ticker):
    return get_stock_sentiment(ticker, max_articles=10)

sentiment = cached_sentiment(selected_ticker)
sent_score = sentiment["overall_score"]
signal_data = generate_signal(current_price, predicted_price, t_score, sent_score)
risk_data = calculate_risk_score(hist_df)

# =====================================================================
# 💰 ROW 1 – KEY METRICS
# =====================================================================
st.markdown('<div class="sec-hdr">💰 Market Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("💵 Current Price", fmt_currency(current_price), f"{live_data['change_pct']:+.2f}%")
delta_pred = ((predicted_price - current_price) / current_price) * 100
c2.metric("🔮 Predicted Price", fmt_currency(predicted_price), f"{delta_pred:+.2f}%")

sig = signal_data["signal"]
c3.markdown(f'<div class="signal-badge signal-{sig}">{sig}</div>', unsafe_allow_html=True)
c3.markdown(f"**Confidence: {signal_data['confidence']}%**")

c4.metric("⚠️ Risk", risk_data["label"], f"Score: {risk_data['score']}")
c5.metric("📐 Tech Score", f"{t_score:+.1f}", "Bullish 🟢" if t_score > 0 else "Bearish 🔴")
c6.metric("🧠 Sentiment", sentiment["overall_label"], f"{sent_score:+.3f}")

# Alert dispatch
if sig == "BUY":
    msg = (f"🚀 <b>BUY Signal</b> for {selected_ticker}\n"
           f"Price: {current_price} → Predicted: {predicted_price}\n"
           f"Confidence: {signal_data['confidence']}%")
    if enable_email:
        send_email_alert(f"AITrade BUY – {selected_ticker}", msg)
    if enable_telegram:
        send_telegram_alert(msg)

# =====================================================================
# 📋 RAW DATA PREVIEW (first 30 rows)
# =====================================================================
if show_data_preview:
    st.markdown('<div class="sec-hdr">📋 Raw Data Preview – First 30 Rows (from Yahoo Finance)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="data-src">
        📥 This data is fetched <b>live from Yahoo Finance</b> using the <code>yfinance</code> Python library.<br>
        Ticker: <b>{selected_ticker}</b> &nbsp;|&nbsp; Period: <b>{analysis_period}</b> &nbsp;|&nbsp;
        Total rows loaded: <b>{len(hist_df):,}</b> &nbsp;|&nbsp;
        Date range: <b>{hist_df['Date'].iloc[0]}</b> → <b>{hist_df['Date'].iloc[-1]}</b><br>
        🔗 You can also provide a <b>Kaggle CSV</b> with columns (Date, Open, High, Low, Close, Volume) — see README.
    </div>
    """, unsafe_allow_html=True)

    preview_df = hist_df.head(30).copy()
    preview_df.index = range(1, len(preview_df) + 1)
    preview_df.index.name = "#"

    # Colour the Close column
    def color_close(val):
        return "color: #69f0ae; font-weight: 700;"

    styled = preview_df.style.map(color_close, subset=["Close"]).format({
        "Open": "${:,.2f}", "High": "${:,.2f}", "Low": "${:,.2f}",
        "Close": "${:,.2f}", "Volume": "{:,.0f}",
    })
    st.dataframe(styled, width='stretch', height=400)

# =====================================================================
# 📊 PRICE CHART
# =====================================================================
st.markdown('<div class="sec-hdr">📊 Price Chart & AI Prediction</div>', unsafe_allow_html=True)

dates = df_ind["Date"] if "Date" in df_ind.columns else df_ind.index

fig_price = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
    row_heights=[0.55, 0.25, 0.20],
    subplot_titles=("Price & Moving Averages", "RSI (14)", "Volume"),
)

fig_price.add_trace(go.Candlestick(
    x=dates, open=df_ind["Open"], high=df_ind["High"],
    low=df_ind["Low"], close=df_ind["Close"], name="OHLC",
    increasing_line_color="#00e676", decreasing_line_color="#ff1744",
), row=1, col=1)

for ma, clr, dash in [
    ("MA_20", "#ff4081", "dot"), ("MA_50", "#ffab00", "solid"),
    ("MA_200", "#00b0ff", "solid"), ("EMA_12", "#ea80fc", "dash"),
    ("EMA_26", "#80d8ff", "dash"),
]:
    if ma in df_ind.columns:
        fig_price.add_trace(go.Scatter(
            x=dates, y=df_ind[ma], name=ma, line=dict(width=2, color=clr, dash=dash),
        ), row=1, col=1)

if "BB_Upper" in df_ind.columns:
    fig_price.add_trace(go.Scatter(x=dates, y=df_ind["BB_Upper"], name="BB Upper",
        line=dict(width=1, dash="dot", color="#ce93d8")), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=dates, y=df_ind["BB_Lower"], name="BB Lower",
        line=dict(width=1, dash="dot", color="#ce93d8"),
        fill="tonexty", fillcolor="rgba(206,147,216,0.07)"), row=1, col=1)

pred_color = "#00e676" if predicted_price >= current_price else "#ff1744"
fig_price.add_trace(go.Scatter(
    x=[dates.iloc[-1]], y=[predicted_price], mode="markers+text", name="AI Prediction",
    marker=dict(size=16, color=pred_color, symbol="star", line=dict(width=2, color="#fff")),
    text=[f"  ★ Predicted: ${predicted_price}"], textposition="top right",
    textfont=dict(color=pred_color, size=13, family="Arial Black"),
), row=1, col=1)

if "RSI" in df_ind.columns:
    fig_price.add_trace(go.Scatter(x=dates, y=df_ind["RSI"], name="RSI",
        line=dict(color="#e040fb", width=2)), row=2, col=1)
    fig_price.add_hline(y=70, line_dash="dash", line_color="#ff5252", row=2, col=1, annotation_text="Overbought")
    fig_price.add_hline(y=30, line_dash="dash", line_color="#69f0ae", row=2, col=1, annotation_text="Oversold")
    fig_price.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", line_width=0, row=2, col=1)

vol_colors = ["#00e676" if df_ind["Close"].iloc[i] >= df_ind["Open"].iloc[i] else "#ff1744"
              for i in range(len(df_ind))]
fig_price.add_trace(go.Bar(x=dates, y=df_ind["Volume"], name="Volume",
    marker_color=vol_colors, opacity=0.7), row=3, col=1)
if "Volume_MA" in df_ind.columns:
    fig_price.add_trace(go.Scatter(x=dates, y=df_ind["Volume_MA"], name="Vol MA",
        line=dict(color="#ffab00", width=1.5)), row=3, col=1)

fig_price.update_layout(
    template="plotly_dark", height=800,
    margin=dict(l=50, r=30, t=40, b=30),
    legend=dict(orientation="h", y=-0.05, font=dict(size=11)),
    xaxis_rangeslider_visible=False,
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
)
st.plotly_chart(fig_price, key="price_chart")

# =====================================================================
# 📉 MACD
# =====================================================================
if "MACD" in df_ind.columns:
    st.markdown('<div class="sec-hdr">📉 MACD Indicator</div>', unsafe_allow_html=True)
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=dates, y=df_ind["MACD"], name="MACD", line=dict(color="#42a5f5", width=2)))
    fig_macd.add_trace(go.Scatter(x=dates, y=df_ind["MACD_Signal"], name="Signal", line=dict(color="#ff7043", width=2)))
    hist_colors = ["#00e676" if v >= 0 else "#ff1744" for v in df_ind["MACD_Hist"].fillna(0)]
    fig_macd.add_trace(go.Bar(x=dates, y=df_ind["MACD_Hist"], name="Histogram", marker_color=hist_colors))
    fig_macd.update_layout(template="plotly_dark", height=300, margin=dict(l=50, r=30, t=20, b=30),
                           plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
    st.plotly_chart(fig_macd, key="macd_chart")

# =====================================================================
# 📐 TECHNICAL + YESTERDAY
# =====================================================================
st.markdown('<div class="sec-hdr">📐 Technical Indicators & Yesterday Performance</div>', unsafe_allow_html=True)
col_tech, col_yest = st.columns(2)

with col_tech:
    latest = df_ind.iloc[-1]
    ind_rows = [
        ("RSI (14)", latest.get("RSI"), ".2f", "🔵"),
        ("MACD", latest.get("MACD"), ".4f", "🟣"),
        ("MACD Signal", latest.get("MACD_Signal"), ".4f", "🟣"),
        ("MACD Histogram", latest.get("MACD_Hist"), ".4f", "🟣"),
        ("MA 20", latest.get("MA_20"), ",.2f", "🩷"),
        ("MA 50", latest.get("MA_50"), ",.2f", "🟡"),
        ("MA 200", latest.get("MA_200"), ",.2f", "🔵"),
        ("EMA 12", latest.get("EMA_12"), ",.2f", "🟪"),
        ("EMA 26", latest.get("EMA_26"), ",.2f", "🟪"),
        ("BB Upper", latest.get("BB_Upper"), ",.2f", "🟪"),
        ("BB Middle", latest.get("BB_Middle"), ",.2f", "🟪"),
        ("BB Lower", latest.get("BB_Lower"), ",.2f", "🟪"),
        ("ATR", latest.get("ATR"), ".2f", "🔴"),
        ("Rel Volume", latest.get("Relative_Volume"), ".2f", "🟠"),
    ]
    rows_html = ""
    for name, val, fmt, icon in ind_rows:
        v = f"${val:{fmt}}" if "," in fmt else (f"{val:{fmt}}" if pd.notna(val) else "N/A")
        rows_html += (f'<div style="padding:6px 10px; border-bottom:1px solid #30363d;">{icon} {name}</div>'
                      f'<div style="padding:6px 10px; border-bottom:1px solid #30363d; color:#69f0ae; font-weight:700;">{v}</div>')
    st.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; font-size:0.95rem; border-radius:10px; overflow:hidden; border:1px solid #30363d;">
        <div style="padding:8px 10px; background:#1a1a3e; color:#80deea; font-weight:700;">Indicator</div>
        <div style="padding:8px 10px; background:#1a1a3e; color:#80deea; font-weight:700;">Value</div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)

with col_yest:
    if "error" not in yesterday:
        trend_color = "#00e676" if yesterday["trend"] == "Bullish" else ("#ff1744" if yesterday["trend"] == "Bearish" else "#ffc107")
        yest_rows = [
            ("Previous Close", fmt_currency(yesterday["previous_close"]), "⬅️"),
            ("Today Open", fmt_currency(yesterday["today_open"]), "➡️"),
            ("Daily Change", fmt_currency(yesterday["daily_change"]), "📊"),
            ("Daily Change %", fmt_pct(yesterday["daily_change_pct"]), "📈"),
            ("Vol (Yesterday)", f"{yesterday['volume_previous']:,}", "🔉"),
            ("Vol (Today)", f"{yesterday['volume_today']:,}", "🔊"),
            ("Vol Change %", fmt_pct(yesterday["volume_change_pct"]), "📶"),
            ("Trend", f"<span style='color:{trend_color}; font-weight:800;'>{yesterday['trend']}</span>", "🧭"),
        ]
        yrows_html = ""
        for name, val, icon in yest_rows:
            yrows_html += (f'<div style="padding:6px 10px; border-bottom:1px solid #30363d;">{icon} {name}</div>'
                           f'<div style="padding:6px 10px; border-bottom:1px solid #30363d; font-weight:600;">{val}</div>')
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; font-size:0.95rem; border-radius:10px; overflow:hidden; border:1px solid #30363d;">
            <div style="padding:8px 10px; background:#1a1a3e; color:#ffab40; font-weight:700;">Metric</div>
            <div style="padding:8px 10px; background:#1a1a3e; color:#ffab40; font-weight:700;">Value</div>
            {yrows_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Yesterday performance data unavailable.")

# =====================================================================
# 🧠 SENTIMENT
# =====================================================================
st.markdown('<div class="sec-hdr">🧠 News Sentiment Analysis</div>', unsafe_allow_html=True)
col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    sc_map = {"Positive": ("#00e676", "🟢"), "Negative": ("#ff1744", "🔴"), "Neutral": ("#ffc107", "🟡")}
    s_clr, s_icon = sc_map.get(sentiment["overall_label"], ("#fff", "⚪"))
    st.markdown(f"""
    <div style="text-align:center; padding:10px; background:linear-gradient(135deg,#1a1a2e,#0d1b2a);
                border-radius:14px; border:2px solid {s_clr};">
        <div style="font-size:2.8rem;">{s_icon}</div>
        <div style="font-size:1.5rem; font-weight:800; color:{s_clr};">{sentiment['overall_label']}</div>
        <div style="font-size:1.1rem; color:#aaa;">Score: {sentiment['overall_score']:+.3f}</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("🟢 Positive", f"{sentiment['positive_pct']}%")
    sc2.metric("🟡 Neutral", f"{sentiment['neutral_pct']}%")
    sc3.metric("🔴 Negative", f"{sentiment['negative_pct']}%")

    fig_donut = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[sentiment["positive_pct"], sentiment["neutral_pct"], sentiment["negative_pct"]],
        hole=0.6, marker=dict(colors=["#00e676", "#ffc107", "#ff1744"]),
        textfont=dict(color="#fff", size=13),
    ))
    fig_donut.update_layout(template="plotly_dark", height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_donut, key="sentiment_donut")

with col_s2:
    if sentiment["articles"]:
        st.markdown("**📰 Latest News Headlines:**")
        for i, art in enumerate(sentiment["articles"][:8]):
            s = art["sentiment"]
            badge_clr = sc_map.get(s["label"], ("#fff", "⚪"))[0]
            st.markdown(
                f'<div style="padding:6px 10px; margin:4px 0; background:#161b22; border-radius:8px; '
                f'border-left:4px solid {badge_clr};">'
                f'<span style="color:{badge_clr}; font-weight:700;">[{s["label"]}]</span> '
                f'{art["title"]} — <i style="color:#8b949e;">{art.get("source", "")}</i></div>',
                unsafe_allow_html=True)
    else:
        st.info("No news articles found.")

# =====================================================================
# ⚠️ RISK
# =====================================================================
st.markdown('<div class="sec-hdr">⚠️ Risk Assessment</div>', unsafe_allow_html=True)
rc1, rc2, rc3, rc4 = st.columns(4)
comps = risk_data.get("components", {})
rc1.metric("🎯 Risk Score", f"{risk_data['score']}/100")
rc2.metric("📊 Volatility", f"{comps.get('volatility_pct', 0):.1f}%")
rc3.metric("📉 Max Drawdown", f"{comps.get('max_drawdown_pct', 0):.1f}%")
rc4.metric("🌊 Trend Instability", f"{comps.get('trend_instability', 0):.1f}")

r_clr = risk_color(risk_data["label"])
st.markdown(f"""
<div class="risk-bar-wrap">
    <div class="risk-bar-fill" style="width:{max(risk_data['score'], 5)}%; background:linear-gradient(90deg,{r_clr},{r_clr}aa);">
        {risk_data['label']}
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 🚦 SIGNAL BREAKDOWN
# =====================================================================
st.markdown('<div class="sec-hdr">🚦 Signal Breakdown</div>', unsafe_allow_html=True)
sg = st.columns(4)
sg[0].metric("🔮 Prediction", f"{signal_data['components'].get('prediction', 0):.2f}")
sg[1].metric("📐 Technical", f"{signal_data['components'].get('technical', 0):.2f}")
sg[2].metric("🧠 Sentiment", f"{signal_data['components'].get('sentiment', 0):.2f}")
sg[3].metric("🎯 Composite", f"{signal_data['composite_score']:.4f}")

model_label = "✅ MLP/LSTM Trained Model" if model_used else "⚡ Moving Average Forecast (no trained model)"
st.markdown(f"<div style='color:#80cbc4; font-size:0.9rem;'>Prediction source: <b>{model_label}</b></div>",
            unsafe_allow_html=True)

# =====================================================================
# 🔴 REAL-TIME MULTI-STOCK SCANNER
# =====================================================================
if show_realtime_scanner:
    st.markdown('<div class="sec-hdr">🔴 Real-Time Multi-Stock Scanner – Live Prices & Signals</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="data-src">
        🔴 <b>Live data from Yahoo Finance</b> — each stock below is fetched in real-time.
        The system predicts next-day price, computes technical score, and generates a BUY/SELL/HOLD signal for every stock.
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data(ttl=120, show_spinner="Scanning all stocks …")
    def scan_all_stocks():
        rows = []
        for tkr in DEFAULT_TICKERS:
            try:
                lp = get_live_price(tkr)
                if "error" in lp:
                    continue
                hdf = get_historical_data(tkr, period="1y")
                if hdf.empty:
                    continue
                p_price, _ = predict_with_model(tkr, hdf)
                hdf_ind = calculate_all_indicators(hdf)
                ts = technical_score(hdf_ind)
                risk = calculate_risk_score(hdf)
                sig = generate_signal(lp["price"], p_price, ts, 0)

                delta = ((p_price - lp["price"]) / lp["price"]) * 100
                rows.append({
                    "Ticker": tkr,
                    "Price": lp["price"],
                    "Change %": lp["change_pct"],
                    "Predicted": p_price,
                    "Pred Δ%": round(delta, 2),
                    "Signal": sig["signal"],
                    "Confidence": sig["confidence"],
                    "Tech Score": ts,
                    "Risk": risk["label"],
                    "Volume": lp["volume"],
                })
            except Exception:
                continue
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    scanner_df = scan_all_stocks()
    if not scanner_df.empty:
        # Show as div-based grid (Streamlit strips <table> tags)
        _GC = "grid-template-columns: 1.2fr 1fr 0.9fr 1fr 0.9fr 0.9fr 0.7fr 0.7fr 1fr 1.2fr"
        html_rows = ""
        for _, r in scanner_df.iterrows():
            sig = r["Signal"]
            sig_clr = {"BUY": "#00e676", "SELL": "#ff1744", "HOLD": "#ffc107"}.get(sig, "#fff")
            chg_clr = "#00e676" if r["Change %"] >= 0 else "#ff1744"
            pred_clr = "#00e676" if r["Pred Δ%"] >= 0 else "#ff1744"
            risk_clr = {"Low Risk": "#00e676", "Medium Risk": "#ffc107", "High Risk": "#ff1744"}.get(r["Risk"], "#fff")
            html_rows += (
                f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; font-weight:800; color:#58a6ff;">{r["Ticker"]}</div>'
                f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; font-weight:700;">${r["Price"]:,.2f}</div>'
                f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; color:{chg_clr}; font-weight:700;">{r["Change %"]:+.2f}%</div>'
                f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; font-weight:700;">${r["Predicted"]:,.2f}</div>'
                f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; color:{pred_clr}; font-weight:700;">{r["Pred Δ%"]:+.2f}%</div>'
                f'<div style="padding:8px 10px; border-bottom:1px solid #30363d;"><span style="background:{sig_clr}; color:#000; padding:3px 14px; border-radius:20px; font-weight:800; font-size:0.85rem;">{sig}</span></div>'
                f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; font-weight:600;">{r["Confidence"]}%</div>'
                f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; font-weight:600;">{r["Tech Score"]:+.1f}</div>'
                f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; color:{risk_clr}; font-weight:700;">{r["Risk"]}</div>'
                f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; color:#aaa;">{r["Volume"]:,}</div>'
            )
        _hdr_style = "padding:10px; font-weight:700;"
        st.markdown(f"""
        <div style="display:grid; {_GC}; font-size:0.92rem; border-radius:10px; overflow:hidden; border:1px solid #30363d;">
            <div style="{_hdr_style} background:linear-gradient(90deg,#6a11cb,#2575fc);">Ticker</div>
            <div style="{_hdr_style} background:linear-gradient(90deg,#6a11cb,#2575fc);">Price</div>
            <div style="{_hdr_style} background:linear-gradient(90deg,#6a11cb,#2575fc);">Change</div>
            <div style="{_hdr_style} background:linear-gradient(90deg,#6a11cb,#2575fc);">Predicted</div>
            <div style="{_hdr_style} background:linear-gradient(90deg,#6a11cb,#2575fc);">Pred Δ</div>
            <div style="{_hdr_style} background:linear-gradient(90deg,#6a11cb,#2575fc);">Signal</div>
            <div style="{_hdr_style} background:linear-gradient(90deg,#6a11cb,#2575fc);">Conf.</div>
            <div style="{_hdr_style} background:linear-gradient(90deg,#6a11cb,#2575fc);">Tech</div>
            <div style="{_hdr_style} background:linear-gradient(90deg,#6a11cb,#2575fc);">Risk</div>
            <div style="{_hdr_style} background:linear-gradient(90deg,#6a11cb,#2575fc);">Volume</div>
            {html_rows}
        </div>
        """, unsafe_allow_html=True)

        # Recommendation
        buys = scanner_df[scanner_df["Signal"] == "BUY"].sort_values("Confidence", ascending=False)
        if not buys.empty:
            top = buys.iloc[0]
            st.markdown(f"""
            <div style="margin-top:12px; padding:14px 20px; background:linear-gradient(135deg,#1b5e20,#004d40);
                        border-radius:12px; border:2px solid #00e676;">
                <span style="font-size:1.3rem; font-weight:800; color:#69f0ae;">
                    🚀 Top Pick: {top['Ticker']}</span>
                <span style="color:#c8e6c9;"> — BUY with {top['Confidence']}% confidence
                    | Price ${top['Price']:,.2f} → Predicted ${top['Predicted']:,.2f}
                    ({top['Pred Δ%']:+.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        sells = scanner_df[scanner_df["Signal"] == "SELL"].sort_values("Confidence", ascending=False)
        if not sells.empty:
            top_sell = sells.iloc[0]
            st.markdown(f"""
            <div style="margin-top:8px; padding:14px 20px; background:linear-gradient(135deg,#b71c1c,#880e4f);
                        border-radius:12px; border:2px solid #ff1744;">
                <span style="font-size:1.3rem; font-weight:800; color:#ff8a80;">
                    ⚠️ Avoid: {top_sell['Ticker']}</span>
                <span style="color:#ffcdd2;"> — SELL signal with {top_sell['Confidence']}% confidence</span>
            </div>
            """, unsafe_allow_html=True)

        # Portfolio allocation recommendation ($10,000 hypothetical)
        st.markdown("""<div class="sec-hdr" style="background:linear-gradient(90deg,#004d40,#00695c);
            box-shadow:0 4px 15px rgba(0,150,136,0.3);">💼 Portfolio Allocation (Hypothetical $10,000)</div>""",
            unsafe_allow_html=True)
        PORTFOLIO_VALUE = 10_000
        buy_stocks = scanner_df[scanner_df["Signal"] == "BUY"].copy()
        if not buy_stocks.empty:
            total_conf = buy_stocks["Confidence"].sum()
            alloc_rows = ""
            for _, row in buy_stocks.iterrows():
                weight = row["Confidence"] / total_conf if total_conf > 0 else 1 / len(buy_stocks)
                alloc_amount = PORTFOLIO_VALUE * weight
                shares = int(alloc_amount // row["Price"]) if row["Price"] > 0 else 0
                alloc_rows += (
                    f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; font-weight:800; color:#58a6ff;">{row["Ticker"]}</div>'
                    f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; font-weight:700;">${row["Price"]:,.2f}</div>'
                    f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; color:#ffc107; font-weight:700;">{weight*100:.1f}%</div>'
                    f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; color:#69f0ae; font-weight:700;">${alloc_amount:,.0f}</div>'
                    f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; color:#00e5ff; font-weight:800; font-size:1.1rem;">{shares}</div>'
                    f'<div style="padding:8px 10px; border-bottom:1px solid #30363d; color:#aaa;">${shares * row["Price"]:,.2f}</div>'
                )
            _ahdr = "padding:10px; font-weight:700; background:linear-gradient(90deg,#004d40,#00695c);"
            st.markdown(f"""
            <div class="data-src">💡 Based on BUY signals — allocate proportionally by confidence. This is <b>not financial advice</b>.</div>
            <div style="display:grid; grid-template-columns:1.2fr 1fr 0.8fr 1fr 1fr 1fr; font-size:0.92rem; border-radius:10px; overflow:hidden; border:1px solid #30363d;">
                <div style="{_ahdr}">Ticker</div>
                <div style="{_ahdr}">Price</div>
                <div style="{_ahdr}">Weight</div>
                <div style="{_ahdr}">Allocation</div>
                <div style="{_ahdr}">Shares to Buy</div>
                <div style="{_ahdr}">Invested</div>
                {alloc_rows}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No BUY signals currently — hold cash or wait for opportunities.")
    else:
        st.info("Scanner data unavailable.")

# =====================================================================
# 🔄 BACKTESTING
# =====================================================================
if show_backtest:
    st.markdown('<div class="sec-hdr">🔄 Backtesting Engine</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=600, show_spinner="Running backtest …")
    def cached_backtest(ticker, period):
        df = get_historical_data(ticker, period=period)
        return run_backtest(df) if not df.empty else None

    bt = cached_backtest(selected_ticker, analysis_period)
    if bt:
        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        ret_clr = "normal" if bt["total_return_pct"] >= 0 else "inverse"
        bc1.metric("📈 Total Return", fmt_pct(bt["total_return_pct"]))
        bc2.metric("🎯 Win Rate", fmt_pct(bt["win_rate_pct"]))
        bc3.metric("📉 Max Drawdown", fmt_pct(bt["max_drawdown_pct"]))
        bc4.metric("⚡ Sharpe", f"{bt['sharpe_ratio']:.2f}")
        bc5.metric("🔢 Trades", bt["total_trades"])

        if bt["equity_curve"]:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=bt["equity_curve"], mode="lines", name="Equity",
                line=dict(color="#00e676", width=2.5),
                fill="tozeroy", fillcolor="rgba(0,230,118,0.1)"))
            fig_eq.add_hline(y=bt["initial_capital"], line_dash="dash", line_color="#ffc107",
                             annotation_text="Initial Capital")
            fig_eq.update_layout(template="plotly_dark", height=320,
                margin=dict(l=50, r=30, t=20, b=30), yaxis_title="Portfolio Value ($)",
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
            st.plotly_chart(fig_eq, key="equity_chart")

        if bt["trades"]:
            with st.expander("📋 View Trade Log"):
                st.dataframe(pd.DataFrame(bt["trades"]), width='stretch')
    else:
        st.info("Not enough data for backtesting.")

# =====================================================================
# 🏆 RANKINGS
# =====================================================================
if show_ranking:
    st.markdown('<div class="sec-hdr">🏆 Stock Rankings – Top 5 Recommended</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=600, show_spinner="Ranking stocks …")
    def cached_ranking():
        return rank_stocks(DEFAULT_TICKERS)

    ranking_df = cached_ranking()
    if not ranking_df.empty:
        st.dataframe(
            ranking_df.head(5).style.background_gradient(
                subset=["Rank Score"], cmap="RdYlGn",
            ).format({
                "Price": "${:.2f}", "Predicted": "${:.2f}",
                "Growth %": "{:+.2f}%", "Risk Score": "{:.1f}",
                "Sentiment": "{:+.3f}", "Rank Score": "{:.2f}",
            }),
            width='stretch',
        )
    else:
        st.info("Ranking data unavailable.")

# =====================================================================
# 🏢 COMPANY INFO
# =====================================================================
st.markdown('<div class="sec-hdr">🏢 Company Information</div>', unsafe_allow_html=True)
if company:
    ci1 = st.columns(4)
    ci1[0].metric("🏛️ Company", company.get("name", selected_ticker))
    ci1[1].metric("🏭 Sector", company.get("sector", "N/A"))
    ci1[2].metric("🔧 Industry", company.get("industry", "N/A"))
    mc = company.get("market_cap")
    ci1[3].metric("💰 Market Cap", f"${mc:,.0f}" if mc else "N/A")
    ci2 = st.columns(4)
    pe = company.get("pe_ratio")
    ci2[0].metric("📊 P/E Ratio", f"{pe:.2f}" if pe else "N/A")
    w52h = company.get("52w_high")
    ci2[1].metric("📈 52W High", f"${w52h:,.2f}" if w52h else "N/A")
    w52l = company.get("52w_low")
    ci2[2].metric("📉 52W Low", f"${w52l:,.2f}" if w52l else "N/A")
    if w52h and w52l and current_price:
        pct_from_high = ((current_price - w52h) / w52h) * 100
        ci2[3].metric("📍 From 52W High", f"{pct_from_high:+.1f}%")

# =====================================================================
# 📈 QUICK COMPARISON
# =====================================================================
st.markdown('<div class="sec-hdr">📈 Quick Comparison – All Tracked Stocks</div>', unsafe_allow_html=True)
comp_cols = st.columns(len(DEFAULT_TICKERS))
for idx, tkr in enumerate(DEFAULT_TICKERS):
    with comp_cols[idx]:
        lp = get_live_price(tkr)
        if "error" not in lp:
            chg = lp.get("change_pct", 0)
            arrow = "🟢" if chg > 0 else ("🔴" if chg < 0 else "⚪")
            bg = "#0d2818" if chg > 0 else ("#2d0a0a" if chg < 0 else "#1a1a2e")
            st.markdown(f"""
            <div style="background:{bg}; padding:10px; border-radius:10px; text-align:center;
                        border:1px solid #30363d; margin:2px;">
                <div style="font-weight:800; color:#58a6ff; font-size:0.9rem;">{tkr}</div>
                <div style="font-size:1.15rem; font-weight:700; color:#fff;">${lp['price']:,.2f}</div>
                <div style="font-weight:700; color:{'#00e676' if chg>0 else '#ff1744'}; font-size:0.9rem;">
                    {arrow} {chg:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#1a1a2e; padding:10px; border-radius:10px; text-align:center;
                        border:1px solid #30363d;"><b>{tkr}</b><br>N/A</div>
            """, unsafe_allow_html=True)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:10px;">
    <p style="color:#8b949e; font-size:0.85rem;">
        <b style="color:#58a6ff;">AITrade v2.0</b> © 2026 — AI Stock Prediction & Advisory System<br>
        📡 Data: Yahoo Finance | 🧠 ML: scikit-learn MLP + Random Forest | 📰 News: Google News RSS<br>
        ⚠️ <i>Educational purposes only. Not financial advice.</i>
    </p>
</div>
""", unsafe_allow_html=True)

if auto_refresh:
    time.sleep(60)
    st.rerun()

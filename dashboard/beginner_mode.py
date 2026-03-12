"""
AITrade – Beginner Learning Mode  (18 sections)
=================================================
A complete learning platform + simulation environment that teaches
users stock market concepts from beginner to advanced level.

Sections
--------
 1. Introduction to the Stock Market
 2. Key Terms Glossary
 3. Interactive Stock Market Lessons
 4. Chart Learning Playground
 5. Technical Indicator Learning
 6. AI Prediction Explained
 7. Fake Stock Market Simulator
 8. Paper Trading Simulator
 9. AI Prediction Playground
10. Market Scenario Simulator
11. Trading Strategy Lab
12. Risk Management Training
13. Quiz System
14. Beginner Portfolio Builder
15. Performance Analyzer
16. AI Trading Coach
17. Progress Tracker
18. Graduation to Advanced Mode
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.learning_content import (
    GLOSSARY_TERMS, LESSONS, MARKET_SCENARIOS,
    TRADING_STRATEGIES, RISK_CONCEPTS, SECTOR_INFO, BEGINNER_TIPS,
)
from utils.fake_market_generator import (
    FAKE_STOCKS, generate_fake_ohlcv, apply_scenario,
    compute_indicators, generate_fake_prediction,
)
from utils.paper_trading_engine import (
    init_state, buy, sell, get_cash, get_holdings,
    get_history, get_portfolio_stats, reset,
)
from utils.quiz_engine import (
    QUIZ_QUESTIONS, get_questions, check_answer,
    calculate_score, get_categories,
)


# =====================================================================
# STYLE HELPERS
# =====================================================================
def _card(title, body, border_color="#30363d", bg="linear-gradient(135deg,#0d1b2e,#1a1a2e)"):
    """Render a styled card."""
    st.markdown(f"""
    <div style="background:{bg}; padding:18px 22px; border-radius:12px;
                border:1px solid {border_color}; margin-bottom:12px;">
        <div style="font-size:1.08rem; font-weight:700; color:#80deea; margin-bottom:8px;">
            {title}
        </div>
        <div style="color:#c9d1d9; font-size:0.95rem; line-height:1.75;">
            {body}
        </div>
    </div>""", unsafe_allow_html=True)


def _section_hdr(title, subtitle=""):
    st.markdown(f"""
    <div style="margin:10px 0 6px 0;">
        <h2 style="color:#00e5ff; font-size:1.45rem; margin:0;">{title}</h2>
        {"<p style='color:#8b949e; font-size:0.92rem; margin:4px 0 10px 0;'>" + subtitle + "</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)


def _metric_box(label, value, color="#00e5ff"):
    st.markdown(f"""
    <div style="background:#0d1b2e; padding:14px; border-radius:10px;
                text-align:center; border:1px solid #30363d;">
        <div style="color:#8b949e; font-size:0.82rem;">{label}</div>
        <div style="color:{color}; font-size:1.4rem; font-weight:800;">{value}</div>
    </div>""", unsafe_allow_html=True)


def _badge(text, colour="#58a6ff"):
    return (f'<span style="background:{colour};color:#fff;padding:4px 12px;'
            f'border-radius:8px;font-weight:700;font-size:0.88rem;">{text}</span>')


# =====================================================================
# SECTION NAVIGATION ITEMS
# =====================================================================
SECTION_LIST = [
    "📖 Introduction to the Stock Market",
    "📚 Key Terms Glossary",
    "🎓 Interactive Stock Market Lessons",
    "📊 Chart Learning Playground",
    "📐 Technical Indicator Learning",
    "🤖 AI Prediction Explained",
    "🎮 Fake Stock Market Simulator",
    "💰 Paper Trading Simulator",
    "🔮 AI Prediction Playground",
    "🌪️ Market Scenario Simulator",
    "🧪 Trading Strategy Lab",
    "🛡️ Risk Management Training",
    "❓ Quiz System",
    "📁 Beginner Portfolio Builder",
    "📈 Performance Analyzer",
    "🧠 AI Trading Coach",
    "🏆 Progress Tracker",
    "🎓 Graduation to Advanced Mode",
]


# =====================================================================
#  CACHED FAKE DATA LOADER
# =====================================================================
@st.cache_data(show_spinner=False)
def _load_fake(ticker, days=252):
    df = generate_fake_ohlcv(ticker, days=days)
    df = compute_indicators(df)
    return df


# =====================================================================
# S1 – INTRODUCTION
# =====================================================================
def _s_introduction():
    _section_hdr("📖 Introduction to the Stock Market",
                 "Learn the fundamentals before you invest a single dollar.")

    _card("What is a Stock?",
          "A <b>stock</b> (or <b>share</b>) is a tiny piece of ownership in a company. "
          "When you buy a stock, you become a part-owner. If the company earns more profit, "
          "your share becomes more valuable.<br><br>"
          "<b>Example:</b> A pizza company 🍕 is worth $1,000 and has 100 shares. "
          "Each share costs $10. If you buy 5 shares ($50), you own 5 % of the company.")

    c1, c2 = st.columns(2)
    with c1:
        _card("Why Do Companies Issue Stock?",
              "Companies need money to build factories, hire people, and develop products. "
              "Instead of borrowing from a bank, they can sell shares to the public — "
              "this is called an <b>IPO</b> (Initial Public Offering). The company gets "
              "cash to grow; investors get a chance to profit if the company succeeds.")
    with c2:
        _card("Why Do Prices Change?",
              "Prices move based on <b>supply and demand</b>.<br>"
              "• More buyers → price goes <b style='color:#00c853;'>UP</b><br>"
              "• More sellers → price goes <b style='color:#ff1744;'>DOWN</b><br><br>"
              "News, earnings reports, economic data, and even social media can shift "
              "demand instantly.")

    st.markdown("##### 💡 Simple Profit Example")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        _metric_box("Buy Price", "$100", "#ffd600")
    with ex2:
        _metric_box("Sell Price", "$120", "#00c853")
    with ex3:
        _metric_box("Your Profit", "$20 (+20%)", "#76ff03")

    st.info("💡 **Key takeaway:** If you buy a stock at $100 and sell at $120, you make a $20 profit (20 % return). "
            "But if the price drops to $80, you'd lose $20. That's why learning is important!")

    # track progress
    if "bm_progress_intro" not in st.session_state:
        st.session_state["bm_progress_intro"] = False
    if st.button("✅ I understand — mark as complete", key="intro_done"):
        st.session_state["bm_progress_intro"] = True
        st.success("Section marked complete!")


# =====================================================================
# S2 – GLOSSARY
# =====================================================================
def _s_glossary():
    _section_hdr("📚 Key Terms Glossary",
                 "Click any term to see a beginner-friendly explanation.")

    for term_info in GLOSSARY_TERMS:
        with st.expander(f"{term_info['icon']} {term_info['term']} — {term_info['short']}"):
            st.markdown(f"<p style='color:#c9d1d9; font-size:0.97rem; line-height:1.75;'>"
                        f"{term_info['detail']}</p>", unsafe_allow_html=True)

    if "bm_progress_glossary" not in st.session_state:
        st.session_state["bm_progress_glossary"] = False
    if st.button("✅ I've read the glossary", key="glossary_done"):
        st.session_state["bm_progress_glossary"] = True
        st.success("Section marked complete!")


# =====================================================================
# S3 – INTERACTIVE LESSONS
# =====================================================================
def _s_lessons():
    _section_hdr("🎓 Interactive Stock Market Lessons",
                 "Progress through 7 lessons step-by-step.")

    if "bm_lesson_idx" not in st.session_state:
        st.session_state["bm_lesson_idx"] = 0
    if "bm_completed_lessons" not in st.session_state:
        st.session_state["bm_completed_lessons"] = set()

    idx = st.session_state["bm_lesson_idx"]

    # lesson selector
    cols_top = st.columns(len(LESSONS))
    for i, lesson in enumerate(LESSONS):
        is_done = lesson["id"] in st.session_state["bm_completed_lessons"]
        is_current = i == idx
        label = f"{'✅' if is_done else lesson['icon']} L{lesson['id']}"
        with cols_top[i]:
            if st.button(label, key=f"lesson_nav_{i}",
                         type="primary" if is_current else "secondary",
                         use_container_width=True):
                st.session_state["bm_lesson_idx"] = i
                st.rerun()

    lesson = LESSONS[idx]
    st.markdown(f"### {lesson['icon']} Lesson {lesson['id']}: {lesson['title']}")
    st.markdown(lesson["content"])

    st.markdown("**🔑 Key Points:**")
    for kp in lesson["key_points"]:
        st.markdown(f"- {kp}")

    bc1, bc2, bc3 = st.columns([1, 1, 1])
    with bc1:
        if idx > 0:
            if st.button("⬅️ Previous Lesson", key="lesson_prev"):
                st.session_state["bm_lesson_idx"] = idx - 1
                st.rerun()
    with bc2:
        if st.button("✅ Mark Complete", key="lesson_complete", type="primary"):
            st.session_state["bm_completed_lessons"].add(lesson["id"])
            st.success(f"Lesson {lesson['id']} complete!")
    with bc3:
        if idx < len(LESSONS) - 1:
            if st.button("Next Lesson ➡️", key="lesson_next"):
                st.session_state["bm_lesson_idx"] = idx + 1
                st.rerun()

    st.progress(len(st.session_state["bm_completed_lessons"]) / len(LESSONS),
                text=f"{len(st.session_state['bm_completed_lessons'])}/{len(LESSONS)} lessons completed")


# =====================================================================
# S4 – CHART LEARNING PLAYGROUND
# =====================================================================
def _s_chart_playground(hist_df, selected_ticker):
    _section_hdr("📊 Chart Learning Playground",
                 "Explore line and candlestick charts. Learn to spot trends.")

    chart_type = st.radio("Chart Type", ["Line Chart", "Candlestick Chart"],
                          horizontal=True, key="chart_pg_type")

    fig = go.Figure()
    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df["Close"],
            mode="lines", name="Close",
            line=dict(color="#00e5ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,229,255,0.07)",
        ))
        st.caption("📌 **Line chart** connects daily closing prices — great for seeing the overall trend at a glance.")
    else:
        fig.add_trace(go.Candlestick(
            x=hist_df.index,
            open=hist_df["Open"], high=hist_df["High"],
            low=hist_df["Low"], close=hist_df["Close"],
            increasing_line_color="#00c853", decreasing_line_color="#ff1744",
            name="OHLC",
        ))
        st.caption("📌 **Candlestick chart** shows Open, High, Low, Close for each day. "
                    "Green = price went up, Red = price went down.")

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,46,0.7)", height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Date", gridcolor="#21262d"),
        yaxis=dict(title="Price ($)", gridcolor="#21262d"),
    )
    st.plotly_chart(fig, use_container_width=True, key="chart_pg_fig")

    # Trend explanation
    st.markdown("##### 📐 Understanding Trends")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        _card("📈 Uptrend", "Prices make <b>higher highs</b> and <b>higher lows</b>. "
              "The overall direction is upward. Good time to consider buying.",
              border_color="#00c853")
    with tc2:
        _card("📉 Downtrend", "Prices make <b>lower highs</b> and <b>lower lows</b>. "
              "The overall direction is downward. Be cautious or consider selling.",
              border_color="#ff1744")
    with tc3:
        _card("➡️ Sideways", "Prices bounce between a <b>support</b> (floor) and "
              "<b>resistance</b> (ceiling). No clear direction — traders wait for a breakout.",
              border_color="#ffc107")

    if "bm_progress_charts" not in st.session_state:
        st.session_state["bm_progress_charts"] = False
    if st.button("✅ I understand charts", key="charts_done"):
        st.session_state["bm_progress_charts"] = True
        st.success("Section marked complete!")


# =====================================================================
# S5 – TECHNICAL INDICATOR LEARNING
# =====================================================================
def _s_indicator_learning(hist_df, selected_ticker):
    _section_hdr("📐 Technical Indicator Learning",
                 "See how each indicator works on real chart data.")

    indicator = st.selectbox("Choose an indicator to learn", [
        "RSI (Relative Strength Index)",
        "MACD (Moving Average Convergence Divergence)",
        "Moving Averages (SMA 20 & 50)",
        "Bollinger Bands",
        "Volume",
    ], key="ind_learn_sel")

    # We need indicators computed on hist_df
    from indicators.technical_indicators import calculate_all_indicators
    df = calculate_all_indicators(hist_df)

    if "RSI" in indicator:
        _card("📈 RSI Explained",
              "RSI ranges from 0 to 100. <b>Above 70</b> → overbought (may drop). "
              "<b>Below 30</b> → oversold (may bounce). It measures the speed of recent "
              "price changes over 14 periods.")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                            vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                                 line=dict(color="#00e5ff")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                 line=dict(color="#ffd600")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff1744", row=2, col=1,
                      annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="#00c853", row=2, col=1,
                      annotation_text="Oversold (30)")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(13,27,46,0.7)", height=500,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, key="ind_rsi_fig")
        st.markdown("**How to read:** When RSI dips below 30 (green zone), the stock might be "
                    "a buying opportunity. When it rises above 70 (red zone), it might be overvalued.")

    elif "MACD" in indicator:
        _card("〰️ MACD Explained",
              "MACD = 12-day EMA − 26-day EMA. The <b>signal line</b> is a 9-day EMA of MACD. "
              "When MACD crosses <b>above</b> the signal → bullish. <b>Below</b> → bearish.")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                            vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                                 line=dict(color="#00e5ff")), row=1, col=1)
        if "MACD" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                                     line=dict(color="#e040fb")), row=2, col=1)
        if "MACD_Signal" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                                     line=dict(color="#ffd600")), row=2, col=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(13,27,46,0.7)", height=500,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, key="ind_macd_fig")
        st.markdown("**How to read:** A bullish crossover (MACD goes above signal) can be a buy signal. "
                    "A bearish crossover (MACD goes below signal) can be a sell signal.")

    elif "Moving" in indicator:
        _card("📉 Moving Averages Explained",
              "A Moving Average smooths out noise. <b>SMA 20</b> (short-term) and "
              "<b>SMA 50</b> (medium-term). When SMA 20 crosses above SMA 50 → "
              "'Golden Cross' (bullish). When below → 'Death Cross' (bearish).")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                                 line=dict(color="#00e5ff")))
        if "SMA_20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20",
                                     line=dict(color="#ffd600", dash="dash")))
        if "SMA_50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50",
                                     line=dict(color="#e040fb", dash="dash")))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(13,27,46,0.7)", height=420,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, key="ind_ma_fig")
        st.markdown("**How to read:** Price above the MA = uptrend. Price below = downtrend. "
                    "The 'Golden Cross' (SMA 20 crosses above SMA 50) is a classic buy signal.")

    elif "Bollinger" in indicator:
        _card("🎸 Bollinger Bands Explained",
              "Bollinger Bands = 20-day SMA ± 2 standard deviations. They expand when "
              "volatility is high and squeeze when volatility is low. Price near the upper "
              "band → overbought. Near the lower band → oversold.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                                 line=dict(color="#00e5ff")))
        if "BB_Upper" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="Upper Band",
                                     line=dict(color="#ff1744", dash="dot")))
        if "BB_Lower" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="Lower Band",
                                     line=dict(color="#00c853", dash="dot"),
                                     fill="tonexty", fillcolor="rgba(0,200,83,0.05)"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(13,27,46,0.7)", height=420,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, key="ind_bb_fig")
        st.markdown("**How to read:** A 'Bollinger Squeeze' (bands tighten) often precedes "
                    "a big price move. Price touching the upper band may signal overbought conditions.")

    else:  # Volume
        _card("📊 Volume Explained",
              "Volume is the number of shares traded. High volume confirms price moves — "
              "a price increase on high volume is more reliable than one on low volume.")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                            vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                                 line=dict(color="#00e5ff")), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color="#6a11cb"), row=2, col=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(13,27,46,0.7)", height=500,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, key="ind_vol_fig")
        st.markdown("**How to read:** Big green candles + high volume = strong buying. "
                    "Big red candles + high volume = strong selling.")

    if "bm_progress_indicators" not in st.session_state:
        st.session_state["bm_progress_indicators"] = False
    if st.button("✅ I understand indicators", key="indicators_done"):
        st.session_state["bm_progress_indicators"] = True
        st.success("Section marked complete!")


# =====================================================================
# S6 – AI PREDICTION EXPLAINED
# =====================================================================
def _s_ai_explained():
    _section_hdr("🤖 How AI Predicts Stock Prices",
                 "Understand the machine learning pipeline behind AITrade.")

    # Pipeline diagram
    steps = [
        ("1️⃣", "Collect Data", "5 years of daily Open, High, Low, Close, Volume from Yahoo Finance."),
        ("2️⃣", "Normalise", "Scale all values to 0-1 range so the model treats them equally."),
        ("3️⃣", "Create Sequences", "Build 60-day windows — the model learns patterns in 60-day chunks."),
        ("4️⃣", "Train Models", "MLP Neural Network (primary) + Random Forest (backup) learn from thousands of sequences."),
        ("5️⃣", "Predict", "Feed the latest 60-day sequence → model outputs a predicted next-day price."),
        ("6️⃣", "Combine", "50 % prediction + 30 % technical score + 20 % sentiment = composite signal."),
    ]

    for emoji, title, desc in steps:
        st.markdown(f"""
        <div style="background:#0d1b2e; padding:12px 18px; border-radius:10px;
                    border-left:4px solid #6a11cb; margin-bottom:8px;">
            <span style="font-size:1.1rem;">{emoji}</span>
            <b style="color:#e040fb;"> {title}</b>
            <p style="color:#c9d1d9; font-size:0.92rem; margin:4px 0 0 28px;">{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.warning("⚠️ **Important:** AI predictions are probabilities — they show the *most likely* "
               "outcome, not a certainty. Always combine AI with your own research and risk management.")

    with st.expander("🔍 What is Machine Learning?"):
        st.markdown(
            "Machine Learning is a branch of AI where computers learn patterns from data "
            "instead of being explicitly programmed. Think of it like teaching a child to "
            "recognise dogs by showing thousands of dog photos — eventually, the child can "
            "identify dogs it has never seen before. Similarly, the model 'sees' thousands "
            "of historical price patterns and learns to predict what comes next."
        )

    with st.expander("🧠 MLP Neural Network vs Random Forest"):
        st.markdown(
            "**MLP (Multi-Layer Perceptron):** An artificial neural network with layers of "
            "neurons. Each neuron takes inputs, applies weights, and passes the result to "
            "the next layer. AITrade uses layers of 128 → 64 → 32 neurons.\n\n"
            "**Random Forest:** An ensemble of 200 decision trees. Each tree makes a prediction, "
            "and the forest averages them. It's robust and handles noisy data well."
        )

    if "bm_progress_ai" not in st.session_state:
        st.session_state["bm_progress_ai"] = False
    if st.button("✅ I understand AI predictions", key="ai_done"):
        st.session_state["bm_progress_ai"] = True
        st.success("Section marked complete!")


# =====================================================================
# S7 – FAKE STOCK MARKET SIMULATOR
# =====================================================================
def _s_fake_simulator():
    _section_hdr("🎮 Fake Stock Market Simulator",
                 "Practice trading with simulated stocks — no real money involved.")

    init_state("sim", 10_000.0)

    # Stock selector
    sel = st.selectbox("Select a Fake Stock", list(FAKE_STOCKS.keys()),
                       format_func=lambda t: f"{t} — {FAKE_STOCKS[t]['name']} ({FAKE_STOCKS[t]['sector']})",
                       key="sim_stock_sel")

    cfg = FAKE_STOCKS[sel]
    df = _load_fake(sel)
    latest_price = round(float(df["Close"].iloc[-1]), 2)

    # Mini chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines",
                             line=dict(color="#00e5ff", width=2), name=sel))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(13,27,46,0.7)", height=280,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, key="sim_chart")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        _metric_box("Current Price", f"${latest_price:,.2f}", "#00e5ff")
    with mc2:
        _metric_box("Sector", cfg["sector"], "#ffd600")
    with mc3:
        _metric_box("Cash Balance", f"${get_cash('sim'):,.2f}", "#76ff03")

    # Trading
    st.markdown("##### 🛒 Trade")
    tc1, tc2 = st.columns(2)
    with tc1:
        buy_qty = st.number_input("Shares to buy", 0, 10000, 1, key="sim_buy_qty")
        if st.button(f"🟢 Buy {buy_qty} × {sel}", key="sim_buy_btn"):
            ok, msg = buy("sim", sel, buy_qty, latest_price)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()
    with tc2:
        holdings = get_holdings("sim")
        max_sell = holdings.get(sel, {}).get("shares", 0)
        sell_qty = st.number_input("Shares to sell", 0, max(max_sell, 0), 0, key="sim_sell_qty")
        if st.button(f"🔴 Sell {sell_qty} × {sel}", key="sim_sell_btn"):
            ok, msg = sell("sim", sel, sell_qty, latest_price)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()

    # Holdings table
    if holdings:
        st.markdown("##### 📊 Your Holdings")
        rows = []
        for tkr, info in holdings.items():
            fp = round(float(_load_fake(tkr)["Close"].iloc[-1]), 2)
            mkt_val = info["shares"] * fp
            pnl = (fp - info["avg_cost"]) * info["shares"]
            rows.append({"Ticker": tkr, "Shares": info["shares"],
                         "Avg Cost": f"${info['avg_cost']:,.2f}",
                         "Market Value": f"${mkt_val:,.2f}",
                         "P&L": f"${pnl:+,.2f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if st.button("🔄 Reset Simulator", key="sim_reset"):
        reset("sim", 10_000.0)
        st.rerun()


# =====================================================================
# S8 – PAPER TRADING SIMULATOR
# =====================================================================
def _s_paper_trading(selected_ticker, current_price):
    _section_hdr("💰 Paper Trading Simulator",
                 f"Practice with real stock data for {selected_ticker}. Demo balance: $10,000.")

    init_state("pt", 10_000.0)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        _metric_box(f"{selected_ticker} Price", f"${current_price:,.2f}", "#00e5ff")
    with mc2:
        _metric_box("Cash Balance", f"${get_cash('pt'):,.2f}", "#76ff03")
    with mc3:
        holdings = get_holdings("pt")
        shares_held = holdings.get(selected_ticker, {}).get("shares", 0)
        _metric_box("Shares Held", str(shares_held), "#ffd600")

    st.markdown("##### 🛒 Trade")
    tc1, tc2 = st.columns(2)
    with tc1:
        buy_qty = st.number_input("Shares to buy", 0, 10000, 1, key="pt_buy_qty")
        cost = buy_qty * current_price
        st.caption(f"Total cost: ${cost:,.2f}")
        if st.button(f"🟢 Buy {buy_qty} × {selected_ticker}", key="pt_buy_btn"):
            ok, msg = buy("pt", selected_ticker, buy_qty, current_price)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()
    with tc2:
        max_sell = shares_held
        sell_qty = st.number_input("Shares to sell", 0, max(max_sell, 0), 0, key="pt_sell_qty")
        proceeds = sell_qty * current_price
        st.caption(f"Proceeds: ${proceeds:,.2f}")
        if st.button(f"🔴 Sell {sell_qty} × {selected_ticker}", key="pt_sell_btn"):
            ok, msg = sell("pt", selected_ticker, sell_qty, current_price)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()

    # Trade history
    hist = get_history("pt")
    if hist:
        with st.expander(f"📜 Trade History ({len(hist)} trades)"):
            st.dataframe(pd.DataFrame(hist), hide_index=True, use_container_width=True)

    if st.button("🔄 Reset Paper Portfolio", key="pt_reset"):
        reset("pt", 10_000.0)
        st.rerun()


# =====================================================================
# S9 – AI PREDICTION PLAYGROUND
# =====================================================================
def _s_ai_playground():
    _section_hdr("🔮 AI Prediction Playground",
                 "Select a fake stock and see how the AI analyses it.")

    sel = st.selectbox("Pick a fake stock", list(FAKE_STOCKS.keys()),
                       format_func=lambda t: f"{t} — {FAKE_STOCKS[t]['name']}",
                       key="ai_pg_sel")

    df = _load_fake(sel)
    latest_price = round(float(df["Close"].iloc[-1]), 2)
    pred = generate_fake_prediction(df)

    # Show price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines",
                             name="Price", line=dict(color="#00e5ff", width=2)))
    fig.add_trace(go.Scatter(x=[df.index[-1]], y=[pred], mode="markers+text",
                             name="AI Prediction", marker=dict(color="#76ff03", size=14, symbol="star"),
                             text=[f"${pred:,.2f}"], textposition="top center",
                             textfont=dict(color="#76ff03")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(13,27,46,0.7)", height=350,
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True, key="ai_pg_chart")

    # Metrics
    change_pct = ((pred - latest_price) / latest_price) * 100
    direction = "📈 UP" if change_pct > 0 else ("📉 DOWN" if change_pct < 0 else "➡️ FLAT")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        _metric_box("Current Price", f"${latest_price:,.2f}", "#00e5ff")
    with mc2:
        _metric_box("AI Prediction", f"${pred:,.2f}", "#76ff03")
    with mc3:
        _metric_box("Expected Move", f"{change_pct:+.2f}%", "#ffd600")
    with mc4:
        _metric_box("Direction", direction, "#e040fb")

    # RSI
    rsi_val = df["RSI"].dropna().iloc[-1] if "RSI" in df.columns and not df["RSI"].dropna().empty else 50
    rsi_label = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")

    st.markdown("##### 🤖 Why does AI think this?")
    reasons = []
    if change_pct > 0.5:
        reasons.append("• Recent price momentum is **upward** — the weighted average of the last 10 days is rising.")
    elif change_pct < -0.5:
        reasons.append("• Recent price momentum is **downward** — prices have been declining.")
    else:
        reasons.append("• Price is relatively **stable** with no strong directional movement.")

    if rsi_val < 30:
        reasons.append(f"• RSI is **{rsi_val:.1f}** (oversold) — suggests potential upward bounce.")
    elif rsi_val > 70:
        reasons.append(f"• RSI is **{rsi_val:.1f}** (overbought) — suggests the stock may cool off.")
    else:
        reasons.append(f"• RSI is **{rsi_val:.1f}** (neutral range) — no extreme condition.")

    reasons.append("• The prediction blends momentum, recent patterns, and a small random factor to simulate uncertainty.")
    st.markdown("\n".join(reasons))


# =====================================================================
# S10 – MARKET SCENARIO SIMULATOR
# =====================================================================
def _s_scenario_simulator():
    _section_hdr("🌪️ Market Scenario Simulator",
                 "See how different market events affect stock prices.")

    sel_stock = st.selectbox("Fake Stock", list(FAKE_STOCKS.keys()),
                             format_func=lambda t: f"{t} — {FAKE_STOCKS[t]['name']}",
                             key="scn_stock")
    sel_scenario = st.selectbox("Scenario", [s["name"] for s in MARKET_SCENARIOS], key="scn_sel")

    scenario = next(s for s in MARKET_SCENARIOS if s["name"] == sel_scenario)

    _card(f"Scenario: {scenario['name']}", scenario["description"],
          border_color="#e040fb")
    st.caption(f"Price effect: ×{scenario['price_effect']:.2f}  |  "
               f"Volatility: ×{scenario['volatility_mult']:.1f}  |  "
               f"Affected sectors: {', '.join(scenario['affected_sectors'])}")

    base_df = _load_fake(sel_stock)
    mod_df = apply_scenario(base_df, scenario["price_effect"], scenario["volatility_mult"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base_df.index, y=base_df["Close"], name="Normal",
                             line=dict(color="#8b949e", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=mod_df.index, y=mod_df["Close"], name=f"After {sel_scenario}",
                             line=dict(color="#ff6d00", width=2.5)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(13,27,46,0.7)", height=400,
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True, key="scn_chart")

    normal_final = float(base_df["Close"].iloc[-1])
    scenario_final = float(mod_df["Close"].iloc[-1])
    impact = ((scenario_final - normal_final) / normal_final) * 100

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        _metric_box("Normal Price", f"${normal_final:,.2f}", "#8b949e")
    with sc2:
        _metric_box("Scenario Price", f"${scenario_final:,.2f}", "#ff6d00")
    with sc3:
        _metric_box("Impact", f"{impact:+.1f}%", "#00c853" if impact > 0 else "#ff1744")


# =====================================================================
# S11 – TRADING STRATEGY LAB
# =====================================================================
def _s_strategy_lab():
    _section_hdr("🧪 Trading Strategy Lab",
                 "Learn and test different trading strategies on fake data.")

    sel_strat = st.selectbox("Choose a Strategy",
                             [s["name"] for s in TRADING_STRATEGIES],
                             key="strat_sel")
    strat = next(s for s in TRADING_STRATEGIES if s["name"] == sel_strat)

    _card(f"{strat['icon']} {strat['name']}", strat["description"],
          border_color="#64ffda")

    st.markdown("**📋 Rules:**")
    for r in strat["rules"]:
        st.markdown(f"- {r}")

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**✅ Pros:**")
        for p in strat["pros"]:
            st.markdown(f"- {p}")
    with pc2:
        st.markdown("**❌ Cons:**")
        for c in strat["cons"]:
            st.markdown(f"- {c}")

    # Backtest simulation on fake data
    st.markdown("---")
    st.markdown("##### 🔬 Strategy Backtest on TECHX")
    df = _load_fake("TECHX")

    signals, cash, holdings_qty, portfolio_vals = [], 10_000.0, 0, []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = float(row["Close"])
        rsi = float(row.get("RSI", 50))
        sma20 = float(row.get("SMA_20", price))
        sma50 = float(row.get("SMA_50", price))

        signal = "HOLD"
        if sel_strat == "Trend Following":
            if price > sma50 and rsi < 70:
                signal = "BUY"
            elif price < sma50:
                signal = "SELL"
        elif sel_strat == "Mean Reversion":
            if rsi < 30:
                signal = "BUY"
            elif rsi > 70:
                signal = "SELL"
        elif sel_strat == "Momentum Trading":
            macd = float(row.get("MACD", 0))
            macd_sig = float(row.get("MACD_Signal", 0))
            if macd > macd_sig:
                signal = "BUY"
            elif macd < macd_sig:
                signal = "SELL"
        else:  # Long-Term
            if price < sma50 * 0.95:
                signal = "BUY"

        # Execute
        if signal == "BUY" and cash >= price:
            qty = int(cash * 0.1 / price)
            if qty > 0:
                cash -= qty * price
                holdings_qty += qty
        elif signal == "SELL" and holdings_qty > 0:
            sell_q = max(1, holdings_qty // 2)
            cash += sell_q * price
            holdings_qty -= sell_q

        portfolio_vals.append(cash + holdings_qty * price)

    if portfolio_vals:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index[50:], y=portfolio_vals,
            mode="lines", name="Portfolio Value",
            line=dict(color="#76ff03", width=2),
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(13,27,46,0.7)", height=300,
                          margin=dict(l=10, r=10, t=20, b=10),
                          yaxis_title="Portfolio ($)")
        st.plotly_chart(fig, use_container_width=True, key="strat_bt_chart")

        final_val = portfolio_vals[-1]
        ret = ((final_val / 10_000) - 1) * 100
        st.markdown(f"**Result:** Starting $10,000 → Final **${final_val:,.2f}** "
                    f"(**{ret:+.1f}%** return)")


# =====================================================================
# S12 – RISK MANAGEMENT TRAINING
# =====================================================================
def _s_risk_training():
    _section_hdr("🛡️ Risk Management Training",
                 "Learn to protect your capital.")

    for concept in RISK_CONCEPTS:
        with st.expander(f"{concept['icon']} {concept['name']}"):
            st.markdown(concept["description"])
            st.markdown(f"**📝 Example:** {concept['example']}")
            st.info(f"💡 **Tip:** {concept['tip']}")

    # Visual: diversification pie
    st.markdown("##### 🥧 Diversification Example")
    labels = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]
    values = [25, 20, 20, 15, 20]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values,
                                  marker_colors=["#00e5ff", "#76ff03", "#ffd600",
                                                  "#ff6d00", "#e040fb"],
                                  hole=0.4)])
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, key="risk_pie")
    st.caption("A well-diversified portfolio spreads risk across multiple sectors.")

    if "bm_progress_risk" not in st.session_state:
        st.session_state["bm_progress_risk"] = False
    if st.button("✅ I understand risk management", key="risk_done"):
        st.session_state["bm_progress_risk"] = True
        st.success("Section marked complete!")


# =====================================================================
# S13 – QUIZ SYSTEM
# =====================================================================
def _s_quiz():
    _section_hdr("❓ Stock Market Quiz",
                 "Test your knowledge! Answer the questions below.")

    if "bm_quiz_answers" not in st.session_state:
        st.session_state["bm_quiz_answers"] = {}
    if "bm_quiz_submitted" not in st.session_state:
        st.session_state["bm_quiz_submitted"] = False

    questions = get_questions()

    if not st.session_state["bm_quiz_submitted"]:
        for q in questions:
            st.markdown(f"**Q{q['id']}.** {q['question']}")
            ans = st.radio("Select your answer:", q["options"],
                           key=f"quiz_q_{q['id']}", index=None)
            if ans is not None:
                st.session_state["bm_quiz_answers"][q["id"]] = q["options"].index(ans)
            st.markdown("---")

        if st.button("📝 Submit Quiz", key="quiz_submit", type="primary"):
            if len(st.session_state["bm_quiz_answers"]) < len(questions):
                st.warning("Please answer all questions before submitting.")
            else:
                st.session_state["bm_quiz_submitted"] = True
                st.rerun()
    else:
        correct, total, pct = calculate_score(st.session_state["bm_quiz_answers"])

        # Score banner
        color = "#00c853" if pct >= 70 else ("#ffc107" if pct >= 50 else "#ff1744")
        st.markdown(f"""
        <div style="background:#0d1b2e; padding:20px; border-radius:14px;
                    border:2px solid {color}; text-align:center; margin-bottom:16px;">
            <div style="font-size:2rem; font-weight:900; color:{color};">
                {correct}/{total} ({pct}%)
            </div>
            <div style="color:#8b949e; font-size:1rem; margin-top:4px;">
                {"🎉 Excellent!" if pct >= 80 else ("👍 Good job!" if pct >= 60 else "📚 Keep learning!")}
            </div>
        </div>""", unsafe_allow_html=True)

        # Show answers
        for q in questions:
            user_ans = st.session_state["bm_quiz_answers"].get(q["id"])
            is_correct, explanation = check_answer(q["id"], user_ans) if user_ans is not None else (False, "")
            icon = "✅" if is_correct else "❌"
            st.markdown(f"**{icon} Q{q['id']}.** {q['question']}")
            if user_ans is not None:
                st.markdown(f"Your answer: **{q['options'][user_ans]}**")
            st.markdown(f"Correct answer: **{q['options'][q['correct']]}**")
            st.caption(explanation)
            st.markdown("---")

        if st.button("🔄 Retake Quiz", key="quiz_retry"):
            st.session_state["bm_quiz_answers"] = {}
            st.session_state["bm_quiz_submitted"] = False
            st.rerun()


# =====================================================================
# S14 – BEGINNER PORTFOLIO BUILDER
# =====================================================================
def _s_portfolio_builder():
    _section_hdr("📁 Beginner Portfolio Builder",
                 "Build a diversified portfolio by allocating across sectors.")

    if "bm_pf_alloc" not in st.session_state:
        st.session_state["bm_pf_alloc"] = {s: 0 for s in SECTOR_INFO}

    st.markdown("Allocate your $10,000 budget across sectors (as %):")

    alloc = {}
    cols = st.columns(len(SECTOR_INFO))
    for i, (sector, info) in enumerate(SECTOR_INFO.items()):
        with cols[i]:
            val = st.number_input(
                f"{info['icon']} {sector}",
                min_value=0, max_value=100,
                value=st.session_state["bm_pf_alloc"].get(sector, 0),
                step=5, key=f"pf_alloc_{sector}",
            )
            alloc[sector] = val

    total_alloc = sum(alloc.values())
    st.session_state["bm_pf_alloc"] = alloc

    if total_alloc != 100:
        st.warning(f"⚠️ Total allocation is **{total_alloc}%** — it must be exactly **100%**.")
    else:
        st.success("✅ Portfolio is balanced at 100%!")

        # Pie chart
        labels = [s for s, v in alloc.items() if v > 0]
        values = [v for v in alloc.values() if v > 0]
        colors = [SECTOR_INFO[s]["color"] for s in labels]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values,
                                      marker_colors=colors, hole=0.4)])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, key="pf_pie")

        # Dollar amounts
        st.markdown("##### 💵 Dollar Allocation")
        for sector, pct in alloc.items():
            if pct > 0:
                amt = 10_000 * pct / 100
                info = SECTOR_INFO[sector]
                st.markdown(f"{info['icon']} **{sector}**: ${amt:,.0f} ({pct}%) — e.g. {info['example']}")


# =====================================================================
# S15 – PERFORMANCE ANALYZER
# =====================================================================
def _s_performance():
    _section_hdr("📈 Performance Analyzer",
                 "Analyse your simulated trading performance.")

    tab_sim, tab_paper = st.tabs(["🎮 Fake Simulator", "💰 Paper Trading"])

    with tab_sim:
        init_state("sim", 10_000.0)
        # Build current prices dict
        prices = {}
        for tkr in FAKE_STOCKS:
            df = _load_fake(tkr)
            prices[tkr] = round(float(df["Close"].iloc[-1]), 2)

        stats = get_portfolio_stats("sim", prices)
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            _metric_box("Total Value", f"${stats['total_value']:,.2f}", "#00e5ff")
        with sc2:
            col = "#00c853" if stats["total_pnl"] >= 0 else "#ff1744"
            _metric_box("P&L", f"${stats['total_pnl']:+,.2f}", col)
        with sc3:
            col = "#00c853" if stats["total_return"] >= 0 else "#ff1744"
            _metric_box("Return", f"{stats['total_return']:+.1f}%", col)
        with sc4:
            _metric_box("Trades", str(stats["num_trades"]), "#ffd600")

        hist = get_history("sim")
        if hist:
            st.dataframe(pd.DataFrame(hist), hide_index=True, use_container_width=True)
        else:
            st.info("No trades yet — go to the Fake Simulator to start trading!")

    with tab_paper:
        init_state("pt", 10_000.0)
        stats_pt = get_portfolio_stats("pt", {}, 10_000.0)
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            _metric_box("Total Value", f"${stats_pt['total_value']:,.2f}", "#00e5ff")
        with sc2:
            col = "#00c853" if stats_pt["total_pnl"] >= 0 else "#ff1744"
            _metric_box("P&L", f"${stats_pt['total_pnl']:+,.2f}", col)
        with sc3:
            col = "#00c853" if stats_pt["total_return"] >= 0 else "#ff1744"
            _metric_box("Return", f"{stats_pt['total_return']:+.1f}%", col)
        with sc4:
            _metric_box("Win Rate", f"{stats_pt['win_rate']:.0f}%", "#ffd600")

        hist_pt = get_history("pt")
        if hist_pt:
            st.dataframe(pd.DataFrame(hist_pt), hide_index=True, use_container_width=True)
        else:
            st.info("No trades yet — go to Paper Trading to start!")


# =====================================================================
# S16 – AI TRADING COACH
# =====================================================================
def _s_ai_coach(selected_ticker, current_price, predicted_price,
                signal_label, sentiment):
    _section_hdr("🧠 AI Trading Coach",
                 f"Understanding AI's analysis for {selected_ticker}.")

    sig_colors = {"BUY": "#00c853", "SELL": "#ff1744", "HOLD": "#ffc107"}
    sig_col = sig_colors.get(signal_label.upper(), "#ffc107")

    st.markdown(f"""
    <div style="background:#0d1b2e; padding:20px 24px; border-radius:14px;
                border:2px solid {sig_col}; margin-bottom:16px;">
        <div style="font-size:1.3rem; font-weight:800; color:{sig_col}; margin-bottom:10px;">
            🤖 AI Signal: {signal_label.upper()}
        </div>
        <div style="color:#c9d1d9; font-size:1rem; line-height:1.8;">
            The AI recommends <b style="color:{sig_col};">{signal_label.upper()}</b>
            for <b style="color:#58a6ff;">{selected_ticker}</b>. Here's why:
        </div>
    </div>""", unsafe_allow_html=True)

    # Build reasons
    change_pct = ((predicted_price - current_price) / current_price * 100) if current_price else 0
    sent_score = sentiment["overall_score"]
    sent_label = sentiment.get("overall_label", "Neutral")

    reasons = []

    # Price prediction
    if change_pct > 0.5:
        reasons.append(f"📈 **Price prediction is positive** — AI predicts ${predicted_price:,.2f} "
                       f"({change_pct:+.1f}% from current ${current_price:,.2f}).")
    elif change_pct < -0.5:
        reasons.append(f"📉 **Price prediction is negative** — AI predicts ${predicted_price:,.2f} "
                       f"({change_pct:+.1f}% from current ${current_price:,.2f}).")
    else:
        reasons.append(f"➡️ **Price prediction is neutral** — AI predicts ${predicted_price:,.2f} "
                       f"(~{change_pct:+.1f}% change).")

    # Sentiment
    if sent_score > 0.05:
        reasons.append(f"📰 **News sentiment is {sent_label}** (score: {sent_score:+.3f}) — "
                       "recent news is mostly positive, supporting upward pressure.")
    elif sent_score < -0.05:
        reasons.append(f"📰 **News sentiment is {sent_label}** (score: {sent_score:+.3f}) — "
                       "recent news is mostly negative, suggesting caution.")
    else:
        reasons.append(f"📰 **News sentiment is {sent_label}** (score: {sent_score:+.3f}) — "
                       "no strong positive or negative bias in recent news.")

    # Signal explanation
    if signal_label.upper() == "BUY":
        reasons.append("🟢 **Composite signal is BUY** — the combination of price prediction, "
                       "technical analysis, and sentiment exceeds the buy threshold (+0.15).")
    elif signal_label.upper() == "SELL":
        reasons.append("🔴 **Composite signal is SELL** — the combination falls below the sell "
                       "threshold (−0.15), suggesting downward pressure.")
    else:
        reasons.append("🟡 **Composite signal is HOLD** — no strong directional conviction. "
                       "The score is between −0.15 and +0.15.")

    for r in reasons:
        st.markdown(r)

    st.markdown("---")
    st.info("💡 **Remember:** AI signals are tools to *assist* your decisions, not replace them. "
            "Always consider your own research, risk tolerance, and financial goals.")


# =====================================================================
# S17 – PROGRESS TRACKER
# =====================================================================
def _s_progress():
    _section_hdr("🏆 Progress Tracker",
                 "See how far you've come in your learning journey.")

    # Section completion
    sections_status = {
        "Introduction": st.session_state.get("bm_progress_intro", False),
        "Glossary": st.session_state.get("bm_progress_glossary", False),
        "Charts": st.session_state.get("bm_progress_charts", False),
        "Indicators": st.session_state.get("bm_progress_indicators", False),
        "AI Explained": st.session_state.get("bm_progress_ai", False),
        "Risk Management": st.session_state.get("bm_progress_risk", False),
    }

    completed = sum(1 for v in sections_status.values() if v)
    total_sections = len(sections_status)

    st.progress(completed / total_sections,
                text=f"Learning Progress: {completed}/{total_sections} sections completed")

    for name, done in sections_status.items():
        icon = "✅" if done else "⬜"
        st.markdown(f"{icon} {name}")

    # Lessons progress
    st.markdown("---")
    lessons_done = len(st.session_state.get("bm_completed_lessons", set()))
    total_lessons = len(LESSONS)
    st.progress(lessons_done / total_lessons if total_lessons else 0,
                text=f"Lessons: {lessons_done}/{total_lessons} completed")

    # Quiz score
    st.markdown("---")
    if st.session_state.get("bm_quiz_submitted", False):
        correct, total, pct = calculate_score(st.session_state.get("bm_quiz_answers", {}))
        st.markdown(f"**Quiz Score:** {correct}/{total} ({pct}%)")
    else:
        st.markdown("**Quiz:** Not yet attempted")

    # Trading stats
    st.markdown("---")
    st.markdown("##### 📊 Trading Activity")
    sim_hist = get_history("sim") if f"bm_sim_history" in st.session_state else []
    pt_hist = get_history("pt") if f"bm_pt_history" in st.session_state else []
    st.markdown(f"- Fake Simulator trades: **{len(sim_hist)}**")
    st.markdown(f"- Paper Trading trades: **{len(pt_hist)}**")


# =====================================================================
# S18 – GRADUATION
# =====================================================================
def _s_graduation():
    _section_hdr("🎓 Graduation to Advanced Mode")

    # Check completion
    sections_done = sum([
        st.session_state.get("bm_progress_intro", False),
        st.session_state.get("bm_progress_glossary", False),
        st.session_state.get("bm_progress_charts", False),
        st.session_state.get("bm_progress_indicators", False),
        st.session_state.get("bm_progress_ai", False),
        st.session_state.get("bm_progress_risk", False),
    ])
    lessons_done = len(st.session_state.get("bm_completed_lessons", set()))
    quiz_done = st.session_state.get("bm_quiz_submitted", False)
    quiz_score = 0
    if quiz_done:
        _, _, quiz_score = calculate_score(st.session_state.get("bm_quiz_answers", {}))

    total_progress = sections_done + lessons_done + (1 if quiz_done else 0)
    max_progress = 6 + len(LESSONS) + 1  # 6 sections + 7 lessons + quiz

    readiness = total_progress / max_progress * 100

    if readiness >= 75:
        st.balloons()
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#004d40,#1b5e20); padding:30px;
                    border-radius:16px; border:2px solid #00c853; text-align:center;">
            <div style="font-size:3rem; margin-bottom:10px;">🎓🎉</div>
            <h2 style="color:#76ff03; margin:0;">Congratulations!</h2>
            <p style="color:#c9d1d9; font-size:1.1rem; margin-top:10px;">
                You have completed <b>{readiness:.0f}%</b> of the Beginner Learning Program!
            </p>
            <p style="color:#76ff03; font-size:1.2rem; font-weight:700; margin-top:16px;">
                🚀 You are ready to explore <b>Advanced Trading Mode</b>!
            </p>
            <p style="color:#c9d1d9; font-size:0.95rem; margin-top:10px;">
                Switch to the <b>📈 Advanced Trading Mode</b> tab to access real stock analysis,
                AI predictions, backtesting, and more.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1a2e,#0d1b2e); padding:30px;
                    border-radius:16px; border:2px solid #ffc107; text-align:center;">
            <div style="font-size:3rem; margin-bottom:10px;">📚</div>
            <h2 style="color:#ffc107; margin:0;">Keep Learning!</h2>
            <p style="color:#c9d1d9; font-size:1.1rem; margin-top:10px;">
                You've completed <b>{readiness:.0f}%</b> of the learning program.
            </p>
            <p style="color:#ffc107; font-size:1rem; margin-top:10px;">
                Complete more lessons, quizzes, and practice sections to unlock graduation!
            </p>
        </div>""", unsafe_allow_html=True)

        st.progress(readiness / 100, text=f"Progress: {readiness:.0f}%")

        st.markdown("##### 📋 What's left:")
        if sections_done < 6:
            st.markdown(f"- Complete {6 - sections_done} more learning sections")
        if lessons_done < len(LESSONS):
            st.markdown(f"- Complete {len(LESSONS) - lessons_done} more lessons")
        if not quiz_done:
            st.markdown("- Take the quiz")
        elif quiz_score < 60:
            st.markdown("- Retake the quiz and score at least 60%")

    st.info("💡 You can always come back to Beginner Mode to review concepts anytime!")


# =====================================================================
# ▶ MAIN ENTRY POINT
# =====================================================================
def render_beginner_dashboard(selected_ticker, current_price, predicted_price,
                               signal_label, hist_df, sentiment):
    """Render the full Beginner Learning Mode tab."""

    # ── Header ──
    st.markdown("""
    <div style="text-align:center; padding:14px 0 4px 0;">
        <h1 style="margin:0; font-size:2.2rem;
            background:linear-gradient(90deg,#00e5ff,#76ff03);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            font-weight:900;">
            🎓 Beginner Learning Mode
        </h1>
        <p style="color:#8b949e; margin-top:4px; font-size:0.95rem;">
            Learn → Practice → Simulate → Analyse → Become an Advanced Trader
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ──
    if "bm_nav" not in st.session_state:
        st.session_state["bm_nav"] = SECTION_LIST[0]

    selected = st.selectbox("📍 Navigate to Section", SECTION_LIST,
                            index=SECTION_LIST.index(st.session_state["bm_nav"]),
                            key="bm_nav_select")
    st.session_state["bm_nav"] = selected

    st.markdown("---")

    # ── Route to section ──
    idx = SECTION_LIST.index(selected)

    if idx == 0:
        _s_introduction()
    elif idx == 1:
        _s_glossary()
    elif idx == 2:
        _s_lessons()
    elif idx == 3:
        _s_chart_playground(hist_df, selected_ticker)
    elif idx == 4:
        _s_indicator_learning(hist_df, selected_ticker)
    elif idx == 5:
        _s_ai_explained()
    elif idx == 6:
        _s_fake_simulator()
    elif idx == 7:
        _s_paper_trading(selected_ticker, current_price)
    elif idx == 8:
        _s_ai_playground()
    elif idx == 9:
        _s_scenario_simulator()
    elif idx == 10:
        _s_strategy_lab()
    elif idx == 11:
        _s_risk_training()
    elif idx == 12:
        _s_quiz()
    elif idx == 13:
        _s_portfolio_builder()
    elif idx == 14:
        _s_performance()
    elif idx == 15:
        _s_ai_coach(selected_ticker, current_price, predicted_price,
                     signal_label, sentiment)
    elif idx == 16:
        _s_progress()
    elif idx == 17:
        _s_graduation()

    # ── Footer ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:8px;">
        <p style="color:#8b949e; font-size:0.82rem;">
            <b style="color:#76ff03;">🎓 Beginner Learning Mode</b> by
            <b style="color:#58a6ff;">AITrade v2.0</b><br>
            ⚠️ <i>Educational purposes only. Not financial advice.</i>
        </p>
    </div>
    """, unsafe_allow_html=True)

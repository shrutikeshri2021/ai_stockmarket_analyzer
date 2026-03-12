"""
AITrade – Learning Content
============================
Pure-data module: glossary, lessons, scenarios, strategies, risk concepts.
No Streamlit or UI code – consumed by dashboard/beginner_mode.py.
"""

# =====================================================================
# GLOSSARY
# =====================================================================
GLOSSARY_TERMS = [
    {
        "term": "Stock / Share",
        "icon": "💲",
        "short": "A small piece of ownership in a company.",
        "detail": (
            "When you buy a stock (also called a share), you own a tiny fraction of "
            "that company. If the company does well, the stock price usually goes up; "
            "if it struggles, the price can fall. Stocks are traded on exchanges like "
            "NYSE, NASDAQ, BSE, and NSE."
        ),
    },
    {
        "term": "Price",
        "icon": "💵",
        "short": "The current cost to buy one share.",
        "detail": (
            "The stock price changes every second during market hours based on supply "
            "and demand. If more people want to buy than sell, the price rises. "
            "The 'Close' price is the last traded price when the market closes for the day."
        ),
    },
    {
        "term": "Volume",
        "icon": "📊",
        "short": "How many shares were traded in a given period.",
        "detail": (
            "High volume means lots of people are buying and selling – the stock is "
            "actively traded. Low volume can mean less interest. A sudden spike in "
            "volume often signals important news or a big price move."
        ),
    },
    {
        "term": "Market Capitalization",
        "icon": "🏢",
        "short": "Total market value of a company's shares.",
        "detail": (
            "Market Cap = Share Price × Total Shares Outstanding. A company trading "
            "at $100 with 1 billion shares has a market cap of $100 billion. Companies "
            "are grouped as Large-cap (>$10B), Mid-cap ($2-10B), or Small-cap (<$2B)."
        ),
    },
    {
        "term": "Prediction",
        "icon": "🔮",
        "short": "An AI-generated estimate of where the price might go next.",
        "detail": (
            "AITrade's ML model analyses historical prices, technical indicators, and "
            "news sentiment to forecast the next closing price. Think of it as a "
            "weather forecast for stocks – helpful, but never 100% accurate."
        ),
    },
    {
        "term": "Technical Indicators",
        "icon": "📐",
        "short": "Math formulas applied to price/volume data to spot trends.",
        "detail": (
            "Indicators like RSI, MACD, and Moving Averages help traders understand "
            "whether a stock is trending up, down, or sideways. They don't predict "
            "the future alone, but they provide useful signals when combined."
        ),
    },
    {
        "term": "RSI (Relative Strength Index)",
        "icon": "📈",
        "short": "Measures if a stock is overbought or oversold (0-100 scale).",
        "detail": (
            "RSI above 70 → the stock may be overbought (could dip soon). "
            "RSI below 30 → the stock may be oversold (could bounce up). "
            "It compares recent gains to recent losses over 14 periods."
        ),
    },
    {
        "term": "MACD",
        "icon": "〰️",
        "short": "Shows the relationship between two moving averages.",
        "detail": (
            "MACD = 12-day EMA minus 26-day EMA. When the MACD line crosses above "
            "the signal line, it's a bullish signal; when it crosses below, it's "
            "bearish. The histogram shows the gap between the two lines."
        ),
    },
    {
        "term": "Moving Average (MA)",
        "icon": "📉",
        "short": "The average closing price over a set number of days.",
        "detail": (
            "A 50-day MA smooths out daily noise to show the medium-term trend. "
            "If the price is above its MA the trend is generally up; below it, "
            "the trend is generally down. Common periods: 20, 50, 100, 200 days."
        ),
    },
    {
        "term": "Bollinger Bands",
        "icon": "🎸",
        "short": "A volatility envelope around the moving average.",
        "detail": (
            "Bollinger Bands are a 20-day MA ± 2 standard deviations. When the bands "
            "squeeze together, low volatility often precedes a big move. Price touching "
            "the upper band can signal overbought; touching the lower band, oversold."
        ),
    },
    {
        "term": "Sentiment",
        "icon": "📰",
        "short": "The overall mood of recent news about a stock.",
        "detail": (
            "AITrade scans Google News headlines and scores them from −1 (very negative) "
            "to +1 (very positive) using NLP. Positive sentiment can push prices up; "
            "negative sentiment can push them down."
        ),
    },
    {
        "term": "Risk Score",
        "icon": "⚠️",
        "short": "How volatile (jumpy) a stock's price is.",
        "detail": (
            "Higher risk = bigger potential gains but also bigger potential losses. "
            "AITrade calculates risk from historical volatility, average true range, "
            "and max drawdown. Low risk stocks are steadier; high risk stocks swing more."
        ),
    },
    {
        "term": "BUY / HOLD / SELL Signal",
        "icon": "🚦",
        "short": "AITrade's composite recommendation.",
        "detail": (
            "The signal combines: 50% price prediction + 30% technical score + 20% "
            "sentiment. If the composite score is above +0.15 → BUY; below −0.15 → SELL; "
            "in between → HOLD. It's a suggestion, not a guarantee."
        ),
    },
]

# =====================================================================
# LESSONS
# =====================================================================
LESSONS = [
    {
        "id": 1,
        "title": "What is the Stock Market?",
        "icon": "🏛️",
        "content": (
            "The stock market is a marketplace where buyers and sellers trade shares of "
            "publicly listed companies. Think of it like a farmers' market — but instead "
            "of fruits and vegetables, people buy and sell tiny pieces of companies.\n\n"
            "**Why does it exist?** Companies need money to grow. Instead of borrowing "
            "from a bank, they can sell shares to the public. In return, shareholders get "
            "a piece of the company's future profits.\n\n"
            "**Major exchanges:** NYSE & NASDAQ (USA), BSE & NSE (India), LSE (UK), "
            "TSE (Japan)."
        ),
        "key_points": [
            "A stock = a tiny piece of ownership in a company",
            "Companies sell shares to raise money for growth",
            "Prices change based on supply and demand",
            "If you buy at $100 and sell at $120, you make $20 profit",
        ],
    },
    {
        "id": 2,
        "title": "Understanding Price Charts",
        "icon": "📊",
        "content": (
            "Price charts are the trader's most basic tool. They show how a stock's price "
            "has moved over time.\n\n"
            "**Line chart:** Connects closing prices — simple and clean.\n\n"
            "**Candlestick chart:** Each 'candle' shows the Open, High, Low, and Close "
            "for one period. A green candle means the price went up; red means it went down.\n\n"
            "**Trends:** An *uptrend* has higher highs and higher lows. A *downtrend* has "
            "lower highs and lower lows. *Sideways* means the price bounces in a range."
        ),
        "key_points": [
            "Line charts show closing prices over time",
            "Candlestick charts show Open, High, Low, Close",
            "Green candle = price went up, Red candle = price went down",
            "Identifying the trend is the first step in analysis",
        ],
    },
    {
        "id": 3,
        "title": "Candlestick Patterns",
        "icon": "🕯️",
        "content": (
            "Certain candlestick shapes repeat and can hint at what might happen next.\n\n"
            "**Doji:** Open and Close are almost equal — the market is undecided.\n\n"
            "**Hammer:** Small body at the top, long lower wick — can signal a reversal "
            "upward after a downtrend.\n\n"
            "**Engulfing:** A large candle completely covers the previous one. Bullish "
            "engulfing (green covers red) can signal buying pressure; bearish engulfing "
            "(red covers green) can signal selling pressure."
        ),
        "key_points": [
            "Doji = market indecision",
            "Hammer = potential reversal after a downtrend",
            "Bullish engulfing = strong buying pressure",
            "Patterns are hints, not guarantees",
        ],
    },
    {
        "id": 4,
        "title": "Technical Indicators",
        "icon": "📐",
        "content": (
            "Indicators are math formulas applied to price and volume data. They help "
            "you measure trend strength, momentum, and volatility.\n\n"
            "**RSI (Relative Strength Index):** 0-100 scale. Above 70 = overbought, "
            "below 30 = oversold.\n\n"
            "**MACD:** Compares short-term vs long-term momentum. Crossovers signal "
            "potential trend changes.\n\n"
            "**Moving Average:** Smooths out noise. Price above MA = bullish, below = bearish.\n\n"
            "**Bollinger Bands:** Show volatility. Tight bands = calm, wide bands = volatile."
        ),
        "key_points": [
            "RSI measures overbought / oversold conditions",
            "MACD shows momentum via moving average crossovers",
            "Moving averages reveal the underlying trend",
            "Bollinger Bands measure volatility",
        ],
    },
    {
        "id": 5,
        "title": "News & Sentiment Analysis",
        "icon": "📰",
        "content": (
            "Stock prices are driven not just by numbers but by human emotions — fear "
            "and greed. News headlines influence how investors feel.\n\n"
            "**Positive news** (strong earnings, new product launch, partnerships) → "
            "investors buy → price goes up.\n\n"
            "**Negative news** (lawsuits, earnings miss, CEO resignation) → investors sell "
            "→ price goes down.\n\n"
            "**NLP (Natural Language Processing)** lets computers read thousands of news "
            "articles and score the overall mood. AITrade uses VADER sentiment analysis "
            "on Google News RSS feeds."
        ),
        "key_points": [
            "News affects investor emotions → affects price",
            "Sentiment score ranges from −1 (negative) to +1 (positive)",
            "AITrade scans Google News automatically",
            "Combine sentiment with technical analysis for best results",
        ],
    },
    {
        "id": 6,
        "title": "Risk Management",
        "icon": "🛡️",
        "content": (
            "The #1 rule of investing: **don't lose money**. Risk management protects "
            "your capital.\n\n"
            "**Diversification:** Don't put all eggs in one basket. Spread across sectors.\n\n"
            "**Position sizing:** Never risk more than 1-2% of your portfolio on a single trade.\n\n"
            "**Stop-loss:** An automatic sell order if the price drops below a set level — "
            "limits your downside.\n\n"
            "**Take-profit:** Lock in gains by selling when a target price is reached."
        ),
        "key_points": [
            "Diversify across sectors and asset types",
            "Never risk more than 1-2% per trade",
            "Use stop-loss orders to limit losses",
            "Preserving capital is more important than maximising gains",
        ],
    },
    {
        "id": 7,
        "title": "AI & Machine Learning in Trading",
        "icon": "🤖",
        "content": (
            "AI can process millions of data points in seconds — far more than any human.\n\n"
            "**How AITrade works:**\n"
            "1. Collects 5 years of historical OHLCV data from Yahoo Finance.\n"
            "2. Normalises data and creates 60-day sequences.\n"
            "3. Trains an MLP Neural Network + Random Forest on the sequences.\n"
            "4. The model learns patterns between past prices and future prices.\n"
            "5. For prediction, it feeds the latest 60-day sequence into the model.\n"
            "6. The output is a predicted next-day closing price.\n\n"
            "**Important:** AI predictions are probabilistic — they show the most likely "
            "outcome, not a certainty. Always combine AI with your own research."
        ),
        "key_points": [
            "AI analyses patterns in large historical datasets",
            "AITrade uses MLP Neural Network + Random Forest",
            "Predictions are probabilities, not certainties",
            "Combine AI signals with technical and sentiment analysis",
        ],
    },
]

# =====================================================================
# MARKET SCENARIOS
# =====================================================================
MARKET_SCENARIOS = [
    {
        "id": "tech_boom",
        "name": "🚀 Tech Boom",
        "description": (
            "A wave of positive earnings reports from major tech companies triggers "
            "a broad tech rally. AI, cloud, and semiconductor stocks surge."
        ),
        "price_effect": 1.25,
        "volatility_mult": 1.3,
        "affected_sectors": ["Technology", "Finance"],
    },
    {
        "id": "market_crash",
        "name": "📉 Market Crash",
        "description": (
            "Global recession fears, rising unemployment, and bank failures cause "
            "a market-wide sell-off. Panic selling drives prices down sharply."
        ),
        "price_effect": 0.70,
        "volatility_mult": 2.0,
        "affected_sectors": ["Technology", "Finance", "Automotive", "Healthcare", "Energy"],
    },
    {
        "id": "rate_hike",
        "name": "🏦 Interest Rate Hike",
        "description": (
            "The central bank raises interest rates to combat inflation. Borrowing "
            "becomes more expensive, hurting growth stocks but helping banks."
        ),
        "price_effect": 0.90,
        "volatility_mult": 1.4,
        "affected_sectors": ["Technology", "Automotive"],
    },
    {
        "id": "earnings_beat",
        "name": "📈 Positive Earnings Surprise",
        "description": (
            "Companies across multiple sectors report earnings well above analyst "
            "expectations. Investor confidence rises and money flows into equities."
        ),
        "price_effect": 1.15,
        "volatility_mult": 1.1,
        "affected_sectors": ["Technology", "Healthcare", "Finance", "Energy"],
    },
]

# =====================================================================
# TRADING STRATEGIES
# =====================================================================
TRADING_STRATEGIES = [
    {
        "name": "Trend Following",
        "icon": "📈",
        "description": (
            "Buy stocks in an uptrend; sell (or avoid) stocks in a downtrend. "
            "The idea: trends tend to continue for a while."
        ),
        "rules": [
            "Buy when price crosses above the 50-day Moving Average",
            "Sell when price crosses below the 50-day Moving Average",
            "Use RSI to avoid buying overbought stocks (RSI > 70)",
        ],
        "pros": ["Simple to understand", "Works well in strong trends", "Clear entry/exit rules"],
        "cons": ["Whipsaws in sideways markets", "Late entries and exits", "Requires patience"],
    },
    {
        "name": "Mean Reversion",
        "icon": "🔄",
        "description": (
            "Prices tend to return to their average over time. Buy when a stock is "
            "unusually cheap; sell when it's unusually expensive."
        ),
        "rules": [
            "Buy when RSI drops below 30 (oversold)",
            "Sell when RSI rises above 70 (overbought)",
            "Confirm with Bollinger Bands: buy near lower band, sell near upper band",
        ],
        "pros": ["Works well in range-bound markets", "Clear overbought/oversold signals"],
        "cons": ["Dangerous in strong trends (catching falling knives)", "Requires quick execution"],
    },
    {
        "name": "Momentum Trading",
        "icon": "⚡",
        "description": (
            "Buy stocks that are rising fast; sell stocks that are falling fast. "
            "Momentum traders ride the wave of strong price movements."
        ),
        "rules": [
            "Buy when MACD crosses above the signal line",
            "Look for increasing volume to confirm momentum",
            "Set a trailing stop-loss to lock in profits",
        ],
        "pros": ["Can capture large gains quickly", "Works in volatile markets"],
        "cons": ["High risk if momentum reverses", "Requires constant monitoring", "Higher transaction costs"],
    },
    {
        "name": "Long-Term Investing",
        "icon": "🏦",
        "description": (
            "Buy quality companies and hold for years. Focus on fundamentals "
            "(revenue, earnings, market position) rather than short-term price swings."
        ),
        "rules": [
            "Research the company's financial health and competitive advantage",
            "Buy at a reasonable price relative to earnings (P/E ratio)",
            "Hold through short-term dips — focus on the long-term trend",
            "Reinvest dividends for compounding growth",
        ],
        "pros": ["Lower stress", "Tax-efficient (long-term capital gains)", "Historically strong returns"],
        "cons": ["Capital locked up for years", "Requires patience during downturns"],
    },
]

# =====================================================================
# RISK MANAGEMENT CONCEPTS
# =====================================================================
RISK_CONCEPTS = [
    {
        "name": "Diversification",
        "icon": "🎯",
        "description": (
            "Spread your investments across different sectors, companies, and asset types. "
            "If one stock drops 50%, the others can cushion your portfolio."
        ),
        "example": (
            "Instead of putting $10,000 into one tech stock, split it: $2,000 each into "
            "Technology, Healthcare, Finance, Energy, and Consumer Goods."
        ),
        "tip": "A well-diversified portfolio has 15-30 stocks across at least 5 sectors.",
    },
    {
        "name": "Stop-Loss Orders",
        "icon": "🛑",
        "description": (
            "A stop-loss automatically sells your stock if it drops to a certain price, "
            "limiting your loss on any single trade."
        ),
        "example": (
            "You buy a stock at $100 and set a stop-loss at $90. If the price drops to "
            "$90, it sells automatically — you lose only 10% instead of potentially more."
        ),
        "tip": "A common stop-loss is 5-10% below your purchase price.",
    },
    {
        "name": "Position Sizing",
        "icon": "📏",
        "description": (
            "Decide how much of your total portfolio to put into each trade. "
            "Never bet everything on one stock."
        ),
        "example": (
            "With a $10,000 portfolio and a 2% risk rule, you risk at most $200 per trade. "
            "If your stop-loss is 10% below entry, you can buy up to $2,000 of that stock."
        ),
        "tip": "The 1-2% rule: never risk more than 1-2% of your total capital on a single trade.",
    },
    {
        "name": "Portfolio Allocation",
        "icon": "🥧",
        "description": (
            "Decide what percentage of your money goes into different asset categories "
            "(stocks, bonds, cash). Younger investors can take more risk (more stocks); "
            "older investors often prefer stability (more bonds)."
        ),
        "example": (
            "Aggressive: 80% stocks, 15% bonds, 5% cash.\n"
            "Moderate: 60% stocks, 30% bonds, 10% cash.\n"
            "Conservative: 40% stocks, 40% bonds, 20% cash."
        ),
        "tip": "Re-balance your portfolio every 6-12 months to maintain target allocations.",
    },
]

# =====================================================================
# SECTOR INFO (for portfolio builder)
# =====================================================================
SECTOR_INFO = {
    "Technology": {"icon": "💻", "color": "#00e5ff", "example": "Apple, Microsoft, Google"},
    "Healthcare": {"icon": "🏥", "color": "#76ff03", "example": "Johnson & Johnson, Pfizer"},
    "Finance": {"icon": "🏦", "color": "#ffd600", "example": "JPMorgan, Goldman Sachs"},
    "Energy": {"icon": "⚡", "color": "#ff6d00", "example": "ExxonMobil, Chevron"},
    "Consumer Goods": {"icon": "🛒", "color": "#e040fb", "example": "Procter & Gamble, Coca-Cola"},
    "Automotive": {"icon": "🚗", "color": "#ff1744", "example": "Tesla, Toyota"},
    "Real Estate": {"icon": "🏠", "color": "#40c4ff", "example": "Prologis, American Tower"},
}

# =====================================================================
# BEGINNER TIPS
# =====================================================================
BEGINNER_TIPS = [
    ("🚫 Don't invest money you can't afford to lose",
     "Only use savings that you won't need for at least 3-5 years."),
    ("📚 Keep learning",
     "Read books like 'The Intelligent Investor' or free online courses on investing."),
    ("🧘 Stay calm during dips",
     "Markets go up and down. Panicking and selling at a loss is the #1 beginner mistake."),
    ("🎯 Diversify",
     "Don't put all your money into one stock. Spread it across different sectors."),
    ("🤖 Use AI as a tool, not a crystal ball",
     "AITrade's predictions help you decide, but no AI is right 100% of the time."),
    ("📅 Think long-term",
     "Historically the market returns ~10% per year on average. Time in the market beats timing the market."),
    ("💰 Start with small amounts",
     "Many brokers let you buy fractional shares. You can start with as little as $1!"),
    ("🔍 Do your own research",
     "Never buy a stock just because someone on social media recommended it."),
]

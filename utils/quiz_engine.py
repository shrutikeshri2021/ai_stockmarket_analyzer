"""
AITrade – Quiz Engine
=======================
Question bank + answer checking for the Beginner Mode quiz system.
Pure logic — no Streamlit UI code.
"""

# =====================================================================
# QUESTION BANK
# =====================================================================
QUIZ_QUESTIONS = [
    {
        "id": 1,
        "category": "Basics",
        "question": "What is a stock?",
        "options": [
            "A loan you give to a company",
            "A small piece of ownership in a company",
            "A type of savings account",
            "A government bond",
        ],
        "correct": 1,
        "explanation": "A stock (share) represents a small fraction of ownership in a company.",
    },
    {
        "id": 2,
        "category": "Basics",
        "question": "Why do stock prices change?",
        "options": [
            "The government sets prices every day",
            "Prices are random and unpredictable",
            "Supply and demand from buyers and sellers",
            "The company CEO decides the price",
        ],
        "correct": 2,
        "explanation": "Prices change based on supply and demand — more buyers push prices up, more sellers push them down.",
    },
    {
        "id": 3,
        "category": "Basics",
        "question": "What does 'Market Capitalization' mean?",
        "options": [
            "The total number of employees",
            "The company's annual revenue",
            "Share price × total shares outstanding",
            "The amount of cash the company has",
        ],
        "correct": 2,
        "explanation": "Market Cap = Share Price × Total Shares Outstanding. It represents the total market value of a company.",
    },
    {
        "id": 4,
        "category": "Technical",
        "question": "What does RSI measure?",
        "options": [
            "Company revenue",
            "Price momentum (overbought/oversold)",
            "Market capitalization",
            "Trading volume only",
        ],
        "correct": 1,
        "explanation": "RSI (Relative Strength Index) measures price momentum on a 0-100 scale. Above 70 = overbought, below 30 = oversold.",
    },
    {
        "id": 5,
        "category": "Technical",
        "question": "An RSI value of 25 suggests the stock is:",
        "options": [
            "Overbought — might drop soon",
            "Oversold — might bounce up",
            "Perfectly priced",
            "About to be delisted",
        ],
        "correct": 1,
        "explanation": "RSI below 30 indicates oversold conditions — the stock may be due for a bounce upward.",
    },
    {
        "id": 6,
        "category": "Technical",
        "question": "What does a Moving Average do?",
        "options": [
            "Predicts exact future prices",
            "Smooths price data to show the trend",
            "Shows company earnings",
            "Measures trading volume",
        ],
        "correct": 1,
        "explanation": "A Moving Average smooths daily price fluctuations to reveal the underlying trend direction.",
    },
    {
        "id": 7,
        "category": "Technical",
        "question": "When MACD crosses above the signal line, it's generally:",
        "options": [
            "A bearish (sell) signal",
            "A bullish (buy) signal",
            "Meaningless noise",
            "A sign to close your account",
        ],
        "correct": 1,
        "explanation": "A MACD crossover above the signal line indicates bullish (upward) momentum.",
    },
    {
        "id": 8,
        "category": "Charts",
        "question": "On a candlestick chart, a green candle means:",
        "options": [
            "The stock lost value that day",
            "The stock is owned by the government",
            "The closing price was higher than the opening price",
            "The volume was very high",
        ],
        "correct": 2,
        "explanation": "Green (bullish) candle = Close > Open — the price went up during that period.",
    },
    {
        "id": 9,
        "category": "Charts",
        "question": "What is an 'uptrend'?",
        "options": [
            "Prices moving mostly downward",
            "Prices staying flat",
            "Prices making higher highs and higher lows",
            "Very high trading volume",
        ],
        "correct": 2,
        "explanation": "An uptrend is defined by a series of higher highs and higher lows on the chart.",
    },
    {
        "id": 10,
        "category": "Sentiment",
        "question": "A sentiment score of +0.8 means:",
        "options": [
            "News is very negative",
            "News is neutral",
            "News is very positive",
            "There is no news available",
        ],
        "correct": 2,
        "explanation": "Sentiment ranges from −1 (very negative) to +1 (very positive). +0.8 is strongly positive.",
    },
    {
        "id": 11,
        "category": "Risk",
        "question": "What is diversification?",
        "options": [
            "Buying only one stock you believe in",
            "Spreading investments across different assets",
            "Selling all your stocks at once",
            "Investing only in bonds",
        ],
        "correct": 1,
        "explanation": "Diversification means spreading your money across different stocks, sectors, or asset types to reduce risk.",
    },
    {
        "id": 12,
        "category": "Risk",
        "question": "What is a stop-loss order?",
        "options": [
            "An order to buy more stock",
            "An automatic sell if price drops to a set level",
            "A request to the CEO to raise prices",
            "A type of dividend payment",
        ],
        "correct": 1,
        "explanation": "A stop-loss order automatically sells your stock when it falls to a specified price, limiting your loss.",
    },
    {
        "id": 13,
        "category": "Risk",
        "question": "The '1-2% rule' in position sizing means:",
        "options": [
            "Only invest 1-2% of your time in research",
            "Never risk more than 1-2% of your portfolio on one trade",
            "Only buy stocks priced between $1 and $2",
            "Sell after a 1-2% gain",
        ],
        "correct": 1,
        "explanation": "The 1-2% rule means you should never risk more than 1-2% of your total portfolio value on any single trade.",
    },
    {
        "id": 14,
        "category": "AI",
        "question": "How does AITrade generate its composite signal?",
        "options": [
            "It randomly picks BUY, HOLD, or SELL",
            "50% price prediction + 30% technical + 20% sentiment",
            "100% based on news headlines",
            "It copies other traders' decisions",
        ],
        "correct": 1,
        "explanation": "AITrade's composite signal = 50% ML price prediction + 30% technical score + 20% sentiment score.",
    },
    {
        "id": 15,
        "category": "AI",
        "question": "Why should you NOT blindly follow AI predictions?",
        "options": [
            "AI is always wrong",
            "AI predictions are probabilities, not certainties",
            "AI only works for crypto",
            "The government bans AI trading",
        ],
        "correct": 1,
        "explanation": "AI predictions are based on historical patterns and probabilities. Markets can behave unpredictably due to unforeseen events.",
    },
    {
        "id": 16,
        "category": "Strategy",
        "question": "In 'Trend Following', you should:",
        "options": [
            "Buy when a stock is in a downtrend",
            "Buy when a stock is in an uptrend",
            "Never buy any stocks",
            "Only buy penny stocks",
        ],
        "correct": 1,
        "explanation": "Trend following means buying stocks moving upward and avoiding or selling stocks moving downward.",
    },
    {
        "id": 17,
        "category": "Strategy",
        "question": "'Mean Reversion' assumes that:",
        "options": [
            "Prices always go up",
            "Prices tend to return to their average over time",
            "The market is always efficient",
            "Stocks never recover from crashes",
        ],
        "correct": 1,
        "explanation": "Mean reversion is the idea that prices and returns eventually move back toward their historical average.",
    },
    {
        "id": 18,
        "category": "Basics",
        "question": "What is the best single piece of advice for a beginner investor?",
        "options": [
            "Put all your money in one stock for maximum returns",
            "Day-trade every day to stay active",
            "Start small, diversify, and think long-term",
            "Only invest in the most expensive stocks",
        ],
        "correct": 2,
        "explanation": "Starting small, diversifying, and thinking long-term is the foundation of successful investing.",
    },
]


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def get_questions(category: str | None = None) -> list:
    """Return questions, optionally filtered by category."""
    if category is None:
        return QUIZ_QUESTIONS
    return [q for q in QUIZ_QUESTIONS if q["category"] == category]


def check_answer(question_id: int, selected_idx: int) -> tuple[bool, str]:
    """Check if *selected_idx* is correct for the given *question_id*."""
    for q in QUIZ_QUESTIONS:
        if q["id"] == question_id:
            is_correct = selected_idx == q["correct"]
            return is_correct, q["explanation"]
    return False, "Question not found."


def calculate_score(answers: dict) -> tuple[int, int, float]:
    """
    Given {question_id: selected_idx}, return (correct, total, percentage).
    """
    correct = 0
    total = len(answers)
    for qid, sel in answers.items():
        for q in QUIZ_QUESTIONS:
            if q["id"] == qid and sel == q["correct"]:
                correct += 1
                break
    pct = (correct / total * 100) if total > 0 else 0.0
    return correct, total, round(pct, 1)


def get_categories() -> list[str]:
    """Return sorted unique categories."""
    return sorted(set(q["category"] for q in QUIZ_QUESTIONS))

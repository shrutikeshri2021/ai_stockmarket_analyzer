"""
AITrade – News Sentiment Analysis Module
==========================================
Fetches latest news for a stock ticker and scores sentiment using VADER.
Supports Google News RSS (no API key required) and NewsAPI (key optional).
"""

import os
import re
import feedparser
import requests
import pandas as pd
from datetime import datetime

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_OK = True
except ImportError:
    _VADER_OK = False

from utils.helpers import logger

# ---------------------------------------------------------------------------
# Sentiment analyser singleton
# ---------------------------------------------------------------------------
_analyzer = SentimentIntensityAnalyzer() if _VADER_OK else None


# ---------------------------------------------------------------------------
# News fetchers
# ---------------------------------------------------------------------------

def _fetch_google_news(query: str, max_results: int = 15) -> list[dict]:
    """
    Fetch news headlines from Google News RSS.  No API key needed.
    """
    from urllib.parse import quote
    safe_query = quote(query, safe="")
    url = f"https://news.google.com/rss/search?q={safe_query}&hl=en&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_results]:
            articles.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "source": entry.get("source", {}).get("title", "Google News"),
            })
        return articles
    except Exception as exc:
        logger.error("Google News fetch failed: %s", exc)
        return []


def _fetch_newsapi(query: str, max_results: int = 15) -> list[dict]:
    """
    Fetch news from NewsAPI.  Requires NEWSAPI_KEY env var.
    """
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "apiKey": api_key,
        "language": "en",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for art in data.get("articles", []):
            articles.append({
                "title": art.get("title", ""),
                "link": art.get("url", ""),
                "published": art.get("publishedAt", ""),
                "source": art.get("source", {}).get("name", ""),
            })
        return articles
    except Exception as exc:
        logger.error("NewsAPI fetch failed: %s", exc)
        return []


def fetch_news(ticker_symbol: str, max_results: int = 15) -> list[dict]:
    """
    Fetch news for a stock ticker.  Tries NewsAPI first, falls back to
    Google News RSS.
    """
    # Build a cleaner search query (strip exchange suffix)
    clean = re.sub(r"\.\w+$", "", ticker_symbol)
    query = f"{clean} stock"

    articles = _fetch_newsapi(query, max_results)
    if not articles:
        articles = _fetch_google_news(query, max_results)
    return articles


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------

def analyse_sentiment(text: str) -> dict:
    """
    Return VADER sentiment scores for a piece of text.

    Keys: neg, neu, pos, compound, label
    """
    if not _VADER_OK or _analyzer is None:
        return {"neg": 0, "neu": 1, "pos": 0, "compound": 0, "label": "Neutral"}

    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "neg": round(scores["neg"], 3),
        "neu": round(scores["neu"], 3),
        "pos": round(scores["pos"], 3),
        "compound": round(compound, 3),
        "label": label,
    }


def get_stock_sentiment(ticker_symbol: str, max_articles: int = 15) -> dict:
    """
    End-to-end: fetch news → score each headline → return aggregate.

    Returns:
        overall_score   : average compound (-1 … +1)
        overall_label   : Positive / Neutral / Negative
        positive_pct    : % of positive headlines
        negative_pct    : % of negative headlines
        neutral_pct     : % of neutral headlines
        articles        : list of dicts with title, source, sentiment
    """
    articles = fetch_news(ticker_symbol, max_articles)
    if not articles:
        return {
            "overall_score": 0,
            "overall_label": "Neutral",
            "positive_pct": 0,
            "negative_pct": 0,
            "neutral_pct": 100,
            "article_count": 0,
            "articles": [],
        }

    scored = []
    for art in articles:
        sent = analyse_sentiment(art["title"])
        scored.append({**art, "sentiment": sent})

    compounds = [a["sentiment"]["compound"] for a in scored]
    labels = [a["sentiment"]["label"] for a in scored]
    total = len(labels)

    avg_compound = sum(compounds) / total if total else 0
    pos_pct = round(labels.count("Positive") / total * 100, 1)
    neg_pct = round(labels.count("Negative") / total * 100, 1)
    neu_pct = round(labels.count("Neutral") / total * 100, 1)

    if avg_compound >= 0.05:
        overall = "Positive"
    elif avg_compound <= -0.05:
        overall = "Negative"
    else:
        overall = "Neutral"

    return {
        "overall_score": round(avg_compound, 3),
        "overall_label": overall,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "neutral_pct": neu_pct,
        "article_count": total,
        "articles": scored,
    }

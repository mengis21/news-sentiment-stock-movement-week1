"""Sentiment scoring helpers (TextBlob + VADER)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Iterable, cast

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

LOGGER = logging.getLogger(__name__)


def compute_headline_sentiment(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """Attach TextBlob polarity and subjectivity scores to a DataFrame."""

    def _score(text: str) -> tuple[float, float]:
        blob = TextBlob(str(text))
        sentiment = cast(Any, blob.sentiment)
        return sentiment.polarity, sentiment.subjectivity

    df = df.copy()
    if df.empty:
        df["polarity"] = pd.Series(dtype=float)
        df["subjectivity"] = pd.Series(dtype=float)
        return df

    polarity, subjectivity = zip(*df[text_col].apply(_score))
    df["polarity"] = list(polarity)
    df["subjectivity"] = list(subjectivity)
    return df


@lru_cache(maxsize=1)
def _vader_analyzer() -> SentimentIntensityAnalyzer:
    """Ensure the VADER lexicon is available and return a shared analyzer."""

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        LOGGER.info("Downloading VADER lexicon...")
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()


def compute_vader_sentiment(
    df: pd.DataFrame,
    text_col: str = "headline",
    prefix: str = "vader",
) -> pd.DataFrame:
    """Attach VADER polarity scores (neg/neu/pos/compound)."""

    analyzer = _vader_analyzer()
    df = df.copy()
    scores = df[text_col].astype(str).apply(analyzer.polarity_scores)
    vader_df = pd.DataFrame(scores.tolist(), index=df.index)
    df[[f"{prefix}_{col}" for col in vader_df.columns]] = vader_df
    return df


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_col: str = "date",
    stock_col: str = "stock",
    columns: Iterable[str] = ("polarity", "subjectivity"),
) -> pd.DataFrame:
    """Average sentiment columns per ticker/day to align with stock returns."""

    if df.empty:
        base_cols = ["date", stock_col] + [f"avg_{col}" for col in columns]
        return pd.DataFrame(columns=base_cols)

    normalized_dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    working = df.assign(day=normalized_dates.dt.date)
    group_cols = ["day"]
    if stock_col in working.columns:
        group_cols.append(stock_col)

    grouped = working.groupby(group_cols)[list(columns)].mean().reset_index()
    rename_map = {col: f"avg_{col}" for col in columns}
    grouped = grouped.rename(columns={"day": "date", **rename_map})
    return grouped

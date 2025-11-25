"""Simple TextBlob-based sentiment scoring."""

from __future__ import annotations

from textblob import TextBlob
import pandas as pd


def compute_headline_sentiment(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """Attach polarity and subjectivity scores to a DataFrame."""
    def _score(text: str) -> tuple[float, float]:
        blob = TextBlob(str(text))
        sentiment = blob.sentiment
        return sentiment.polarity, sentiment.subjectivity

    scores = df[text_col].apply(_score)
    df = df.copy()
    df["polarity"], df["subjectivity"] = zip(*scores)
    return df


def aggregate_daily_sentiment(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Average polarity per day to align with stock returns."""
    daily = (
        df.assign(day=df[date_col].dt.date)
        .groupby("day")[["polarity", "subjectivity"]]
        .mean()
        .reset_index()
        .rename(columns={"day": "date", "polarity": "avg_polarity", "subjectivity": "avg_subjectivity"})
    )
    return daily

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

    df = df.copy()
    scores = list(df[text_col].apply(_score)) if not df.empty else []
    if scores:
        polarity, subjectivity = zip(*scores)
        df["polarity"] = list(polarity)
        df["subjectivity"] = list(subjectivity)
    else:
        df["polarity"] = pd.Series(dtype=float)
        df["subjectivity"] = pd.Series(dtype=float)
    return df


def aggregate_daily_sentiment(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Average polarity per day to align with stock returns."""

    if df.empty:
        return pd.DataFrame(columns=["date", "avg_polarity", "avg_subjectivity"])

    normalized_dates = pd.to_datetime(df[date_col], errors="coerce")

    daily = (
        df.assign(day=normalized_dates.dt.date)
        .groupby("day")[["polarity", "subjectivity"]]
        .mean()
        .reset_index()
        .rename(columns={"day": "date", "polarity": "avg_polarity", "subjectivity": "avg_subjectivity"})
    )
    return daily

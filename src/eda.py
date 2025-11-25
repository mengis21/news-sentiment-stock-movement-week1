"""Exploratory helpers for the interim analysis."""

from __future__ import annotations

import pandas as pd


def headline_length_stats(df: pd.DataFrame, text_col: str = "headline") -> dict:
    """Return simple stats for headline lengths."""
    lengths = df[text_col].astype(str).str.len()
    stats = lengths.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    return {k: float(v) for k, v in stats.items()}


def publisher_activity(df: pd.DataFrame, publisher_col: str = "publisher", top_n: int = 10) -> pd.DataFrame:
    """Count how many stories each publisher contributed."""
    counts = df[publisher_col].fillna("unknown").value_counts().head(top_n)
    return counts.rename_axis("publisher").reset_index(name="article_count")


def daily_article_counts(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Track how many headlines land per calendar day."""
    daily = df.groupby(df[date_col].dt.date).size().rename("article_count")
    return daily.reset_index(names=["date"])

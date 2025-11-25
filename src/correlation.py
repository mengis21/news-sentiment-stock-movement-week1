"""Helpers to line up sentiment with price moves."""

from __future__ import annotations

import pandas as pd

try:  # pragma: no cover - prefer package relative import
    from .technical import compute_daily_returns
except ImportError:  # pragma: no cover - allow running as script
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.technical import compute_daily_returns


def align_sentiment_with_returns(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    prices = compute_daily_returns(price_df)
    prices = prices.assign(trade_date=prices["date"].dt.date)
    merged = prices.merge(
        sentiment_df,
        left_on="trade_date",
        right_on="date",
        how="left",
        suffixes=("_price", "_sentiment"),
    )
    merged = merged.drop(columns=["date_sentiment"], errors="ignore").rename(columns={"trade_date": "date"})
    return merged


def correlation_between_sentiment_and_returns(
    df: pd.DataFrame,
    sentiment_col: str = "avg_polarity",
    return_col: str = "daily_return",
) -> float:
    clean = df[[sentiment_col, return_col]].dropna()
    if clean.empty:
        return float("nan")
    corr_matrix = clean.corr(method="pearson")
    return float(corr_matrix.loc[sentiment_col, return_col])

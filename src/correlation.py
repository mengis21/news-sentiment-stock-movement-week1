"""Helpers to line up sentiment with price moves."""

from __future__ import annotations

import pandas as pd
from numbers import Number
from typing import Sequence

try:  # pragma: no cover - prefer package relative import
    from .technical import compute_daily_returns
except ImportError:  # pragma: no cover - allow running as script
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.technical import compute_daily_returns


def align_sentiment_with_returns(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
    price_col: str = "close",
    price_ticker_col: str = "ticker",
    sentiment_ticker_col: str = "stock",
    sentiment_shift_days: int = 0,
) -> pd.DataFrame:
    """Merge aggregated sentiment with price-based daily returns."""

    if "date" not in price_df.columns:
        raise ValueError("price_df must contain a 'date' column")
    if "date" not in sentiment_df.columns:
        raise ValueError("sentiment_df must contain a 'date' column")

    prices = compute_daily_returns(price_df, price_col=price_col, ticker_col=price_ticker_col)
    price_dates = pd.to_datetime(prices["date"], errors="coerce", utc=True)
    prices = prices.assign(trade_date=price_dates.dt.date)

    sentiment = sentiment_df.copy()
    sentiment_dates = pd.to_datetime(sentiment["date"], errors="coerce", utc=True)
    if sentiment_shift_days:
        sentiment_dates = sentiment_dates + pd.to_timedelta(sentiment_shift_days, unit="D")
    sentiment = sentiment.assign(merge_date=sentiment_dates.dt.date)

    join_left = ["trade_date"]
    join_right = ["merge_date"]
    price_has_ticker = price_ticker_col in prices.columns
    sentiment_has_ticker = sentiment_ticker_col in sentiment.columns

    if price_has_ticker and sentiment_has_ticker:
        if price_ticker_col != sentiment_ticker_col:
            sentiment = sentiment.rename(columns={sentiment_ticker_col: price_ticker_col})
        join_left.append(price_ticker_col)
        join_right.append(price_ticker_col)

    merged = prices.merge(
        sentiment,
        left_on=join_left,
        right_on=join_right,
        how="left",
        suffixes=("_price", "_sentiment"),
    )
    merged = merged.drop(columns=["merge_date"], errors="ignore")
    merged = merged.rename(columns={"trade_date": "date"})
    if "date_price" in merged.columns:
        merged = merged.rename(columns={"date_price": "price_timestamp"})
    merged = merged.drop(columns=["date_sentiment"], errors="ignore")
    return merged


def correlation_between_sentiment_and_returns(
    df: pd.DataFrame,
    sentiment_col: str = "avg_polarity",
    return_col: str = "daily_return",
) -> float:
    clean = df[[sentiment_col, return_col]].dropna()
    if clean.empty:
        # No overlapping data to correlate; return NaN to indicate missing result.
        return float("nan")
    corr_matrix = clean.corr(method="pearson")
    raw = corr_matrix.loc[sentiment_col, return_col]
    # If it's a plain Python number, convert directly.
    if isinstance(raw, Number):
        return float(raw)
    # If it's a numpy/pandas scalar with .item(), use that.
    item = getattr(raw, "item", None)
    if callable(item):
        return float(item())
    # Fallback to float() which will raise if conversion is invalid.
    return float(raw)


def correlations_by_ticker(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
    sentiment_columns: Sequence[str],
    price_col: str = "close",
    price_ticker_col: str = "ticker",
    sentiment_ticker_col: str = "stock",
    sentiment_shift_days: int = 0,
) -> pd.DataFrame:
    """Return ticker-level correlations for multiple sentiment columns."""

    aligned = align_sentiment_with_returns(
        sentiment_df,
        price_df,
        price_col=price_col,
        price_ticker_col=price_ticker_col,
        sentiment_ticker_col=sentiment_ticker_col,
        sentiment_shift_days=sentiment_shift_days,
    )

    if aligned.empty:
        return pd.DataFrame(columns=["ticker", "sentiment_metric", "correlation", "pair_count"])

    ticker_col = price_ticker_col if price_ticker_col in aligned.columns else None
    grouped = aligned.groupby(ticker_col) if ticker_col else [("ALL", aligned)]

    rows: list[dict[str, object]] = []
    for ticker, group in grouped:
        for metric in sentiment_columns:
            if metric not in group.columns:
                continue
            corr_value = correlation_between_sentiment_and_returns(group, sentiment_col=metric)
            valid_pairs = group[[metric, "daily_return"]].dropna().shape[0]
            rows.append(
                {
                    "ticker": ticker,
                    "sentiment_metric": metric,
                    "correlation": float(corr_value) if not pd.isna(corr_value) else float("nan"),
                    "pair_count": int(valid_pairs),
                }
            )

    return pd.DataFrame(rows)


def rolling_sentiment_return_corr(
    aligned_df: pd.DataFrame,
    sentiment_col: str = "avg_polarity",
    return_col: str = "daily_return",
    ticker_col: str = "ticker",
    window: int = 5,
) -> pd.DataFrame:
    """Compute rolling correlations to surface regime shifts for a given sentiment column."""

    if aligned_df.empty:
        return aligned_df.copy()

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        ordered = group.sort_values("date")
        rolling_corr = ordered[sentiment_col].rolling(window=window, min_periods=window).corr(ordered[return_col])
        col_name = f"rolling_corr_{sentiment_col}_{return_col}_{window}"
        return ordered.assign(**{col_name: rolling_corr})

    if ticker_col in aligned_df.columns:
        return aligned_df.groupby(ticker_col, group_keys=False).apply(_apply)
    return _apply(aligned_df)

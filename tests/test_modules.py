"""Smoke tests for interim helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.correlation import align_sentiment_with_returns, correlation_between_sentiment_and_returns, correlations_by_ticker
from src.data_io import load_news_data, load_stock_data
from src.eda import headline_length_stats
from src.sentiment import aggregate_daily_sentiment, compute_headline_sentiment
from src.technical import add_bollinger_bands, add_macd, add_moving_average, add_rsi, add_volatility


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_loaders_read_sample_files() -> None:
    news = load_news_data(DATA_DIR / "sample_news.csv")
    prices = load_stock_data(DATA_DIR / "sample_prices.csv")
    assert not news.empty
    assert not prices.empty
    assert pd.api.types.is_datetime64_any_dtype(news["date"])
    assert pd.api.types.is_datetime64_any_dtype(prices["date"])


def test_headline_stats_returns_numbers() -> None:
    news = load_news_data(DATA_DIR / "sample_news.csv")
    stats = headline_length_stats(news)
    assert set(["mean", "max", "min"]).issubset(stats)


def test_sentiment_and_indicators_columns() -> None:
    news = load_news_data(DATA_DIR / "sample_news.csv")
    prices = load_stock_data(DATA_DIR / "sample_prices.csv")

    scored_news = compute_headline_sentiment(news)
    assert {"polarity", "subjectivity"}.issubset(scored_news.columns)

    enriched = add_moving_average(prices, window=3)
    enriched = add_rsi(enriched, window=3)
    assert any(col.startswith("ma_") for col in enriched.columns)
    assert any(col.startswith("rsi_") for col in enriched.columns)


def test_advanced_indicators_attach_columns() -> None:
    prices = load_stock_data(DATA_DIR / "sample_prices.csv")
    enriched = add_macd(prices)
    enriched = add_bollinger_bands(enriched, window=3)
    enriched = add_volatility(enriched, window=3)
    assert {"macd", "macd_signal", "macd_hist"}.issubset(enriched.columns)
    assert any(col.startswith("bb_high_") for col in enriched.columns)
    assert any(col.startswith("volatility_") for col in enriched.columns)


def test_sentiment_helpers_handle_empty_frames() -> None:
    empty_news = pd.DataFrame({"headline": [], "date": []})
    scored = compute_headline_sentiment(empty_news)
    assert {"polarity", "subjectivity"}.issubset(scored.columns)

    aggregated = aggregate_daily_sentiment(scored)
    assert list(aggregated.columns) == ["date", "stock", "avg_polarity", "avg_subjectivity"]
    assert aggregated.empty


def test_align_sentiment_with_returns_respects_tickers() -> None:
    price_df = pd.DataFrame(
        {
            "date": pd.to_datetime([
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ], utc=True),
            "close": [10, 11, 12, 20, 19, 18],
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
        }
    )
    sentiment_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"], utc=True),
            "stock": ["AAA", "BBB"],
            "avg_polarity": [0.2, -0.1],
        }
    )

    aligned = align_sentiment_with_returns(sentiment_df, price_df)
    assert set(aligned["ticker"]) == {"AAA", "BBB"}
    aaa_rows = aligned[aligned["ticker"] == "AAA"]
    bbb_rows = aligned[aligned["ticker"] == "BBB"]
    assert 0.2 in aaa_rows["avg_polarity"].values
    assert -0.1 in bbb_rows["avg_polarity"].values


def test_correlations_by_ticker_outputs_expected_rows() -> None:
    price_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
            "close": [100, 102, 101],
            "ticker": ["TEST"] * 3,
        }
    )
    sentiment_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "stock": ["TEST", "TEST"],
            "avg_polarity": [0.4, -0.2],
        }
    )

    corr_df = correlations_by_ticker(
        sentiment_df,
        price_df,
        sentiment_columns=["avg_polarity"],
        sentiment_shift_days=1,
    )
    assert not corr_df.empty
    row = corr_df.iloc[0]
    assert row["ticker"] == "TEST"
    assert row["pair_count"] == 2
    assert isinstance(row["correlation"], float)

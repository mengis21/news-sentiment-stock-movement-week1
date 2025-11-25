"""Smoke tests for interim helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_io import load_news_data, load_stock_data
from src.eda import headline_length_stats
from src.sentiment import compute_headline_sentiment
from src.technical import add_moving_average, add_rsi


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

"""Command-line helper that reproduces the interim workflow on sample data."""

from __future__ import annotations

import argparse
import json

from src.correlation import align_sentiment_with_returns, correlation_between_sentiment_and_returns
from src.data_io import load_news_data, load_stock_data
from src.eda import daily_article_counts, headline_length_stats, publisher_activity
from src.sentiment import aggregate_daily_sentiment, compute_headline_sentiment
from src.technical import add_moving_average, add_rsi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interim analysis workflow")
    parser.add_argument("--news", default="data/raw/sample_news.csv", help="Path to the news CSV")
    parser.add_argument("--prices", default="data/raw/sample_prices.csv", help="Path to the price CSV")
    parser.add_argument("--ma-window", type=int, default=5, help="Moving average window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    news_df = load_news_data(args.news)
    price_df = load_stock_data(args.prices)

    news_df = compute_headline_sentiment(news_df)
    sentiment_daily = aggregate_daily_sentiment(news_df)

    enriched_prices = add_moving_average(price_df, args.ma_window)
    enriched_prices = add_rsi(enriched_prices, window=7)

    aligned = align_sentiment_with_returns(sentiment_daily, enriched_prices)
    corr_value = correlation_between_sentiment_and_returns(aligned)

    summary = {
        "headline_length_stats": headline_length_stats(news_df),
        "top_publishers": publisher_activity(news_df).to_dict(orient="records"),
        "daily_articles": daily_article_counts(news_df).tail().to_dict(orient="records"),
        "correlation": corr_value,
    }

    print(json.dumps(summary, indent=2, default=str))
if __name__ == "__main__":
    main()

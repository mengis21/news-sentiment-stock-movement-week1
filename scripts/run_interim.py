"""Command-line helper with richer Task 2 / 3 analytics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.correlation import correlations_by_ticker
from src.data_io import load_news_data, load_stock_data, load_stock_directory
from src.eda import (
    daily_article_counts,
    headline_length_stats,
    lda_topics,
    publisher_activity,
    publisher_domain_breakdown,
    publishing_hour_distribution,
    tfidf_top_phrases,
)
from src.sentiment import aggregate_daily_sentiment, compute_headline_sentiment, compute_vader_sentiment
from src.technical import (
    add_bollinger_bands,
    add_macd,
    add_moving_average,
    add_rsi,
    add_volatility,
    apply_pynance_bollinger,
    pynance_growth_metrics,
)

LOGGER = logging.getLogger(__name__)
TECH_PREFIXES = ("ma_", "rsi_", "macd", "bb_", "volatility_", "pn_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sentiment + technical analytics across tickers")
    parser.add_argument("--news", default="data/raw/news_headlines.csv", help="Path to the news CSV")
    parser.add_argument("--prices", help="Path to a single OHLCV CSV (overrides --price-dir)")
    parser.add_argument("--price-dir", default="data/raw", help="Directory containing ticker-level OHLCV CSVs")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA"],
        help="Ticker symbols to load from the price directory",
    )
    parser.add_argument("--single-ticker-label", default="SAMPLE", help="Ticker label for --prices CSVs")
    parser.add_argument("--ma-windows", nargs="+", type=int, default=[5, 20], help="Moving average windows")
    parser.add_argument("--rsi-window", type=int, default=14, help="RSI lookback window")
    parser.add_argument("--boll-window", type=int, default=20, help="Bollinger Band window")
    parser.add_argument("--vol-window", type=int, default=20, help="Rolling volatility window")
    parser.add_argument("--tfidf-top-k", type=int, default=25, help="Number of TF-IDF phrases to return")
    parser.add_argument("--topics", type=int, default=5, help="Number of LDA topics to model")
    parser.add_argument("--topic-words", type=int, default=10, help="Words per topic to display")
    parser.add_argument("--max-news-rows", type=int, help="Optional cap on number of news rows to process")
    parser.add_argument("--skip-tfidf", action="store_true", help="Skip TF-IDF keyword extraction")
    parser.add_argument("--skip-topics", action="store_true", help="Skip topic modeling")
    parser.add_argument("--disable-pynance", action="store_true", help="Skip PyNance-specific indicators")
    parser.add_argument(
        "--sentiment-shift-days",
        type=int,
        default=1,
        help="Days to shift sentiment forward when aligning with returns",
    )
    parser.add_argument("--output", help="Optional path to save the JSON summary instead of stdout")
    return parser.parse_args()


def _load_prices(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """Load either a single CSV or an entire directory of ticker files."""

    if args.prices:
        df = load_stock_data(args.prices)
        ticker_label = args.single_ticker_label.upper()
        df["ticker"] = ticker_label
        return df, [ticker_label]

    tickers = [t.upper() for t in args.tickers]
    df = load_stock_directory(args.price_dir, tickers=tickers)
    return df, tickers


def _apply_indicator_suite(
    df: pd.DataFrame,
    ma_windows: Iterable[int],
    rsi_window: int,
    boll_window: int,
    vol_window: int,
    enable_pynance: bool,
) -> pd.DataFrame:
    ordered = df.sort_values("date").copy()
    for window in ma_windows:
        ordered = add_moving_average(ordered, window=window)
    ordered = add_rsi(ordered, window=rsi_window)
    ordered = add_macd(ordered)
    ordered = add_bollinger_bands(ordered, window=boll_window)
    ordered = add_volatility(ordered, window=vol_window)
    if enable_pynance:
        try:
            ordered = apply_pynance_bollinger(ordered, window=boll_window)
        except ImportError:
            LOGGER.debug("PyNance not installed; skipping pn-specific Bollinger columns")
        except Exception as exc:  # pragma: no cover - PyNance runtime variance
            LOGGER.warning("PyNance Bollinger failed: %s", exc)
    return ordered


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    working = df if limit is None else df.head(limit)
    return working.to_dict(orient="records")


def _safe_tfidf(texts: pd.Series, top_k: int) -> pd.DataFrame:
    try:
        return tfidf_top_phrases(texts, top_k=top_k)
    except ValueError as exc:
        LOGGER.warning("TF-IDF failed: %s", exc)
        return pd.DataFrame(columns=["term", "score"])


def _safe_topics(texts: pd.Series, n_topics: int, n_top_words: int) -> list[list[str]]:
    try:
        return lda_topics(texts, n_topics=n_topics, n_top_words=n_top_words)
    except ValueError as exc:
        LOGGER.warning("LDA failed: %s", exc)
        return []


def _indicator_snapshot(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    columns = ["ticker", "date"] + [col for col in df.columns if col.startswith(TECH_PREFIXES)]
    columns = [col for col in columns if col in df.columns]
    snapshot = df.sort_values("date").groupby("ticker").tail(1)[columns]
    return _records(snapshot)


def _pynance_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    try:
        for ticker, group in df.groupby("ticker"):
            key = str(ticker)
            summary[key] = pynance_growth_metrics(group)
    except ImportError:
        LOGGER.info("PyNance not installed; skipping growth metrics")
        return {}
    except Exception as exc:
        LOGGER.warning("PyNance growth metrics failed: %s", exc)
    return summary


def _clean_corr(value: float | int | None) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    news_df = load_news_data(args.news)
    prices_df, tickers = _load_prices(args)

    if args.max_news_rows and len(news_df) > args.max_news_rows:
        LOGGER.info("Subsampling news from %s to %s rows", len(news_df), args.max_news_rows)
        news_df = news_df.head(args.max_news_rows)

    news_df = compute_headline_sentiment(news_df)
    news_df = compute_vader_sentiment(news_df)
    base_sentiment_cols = [col for col in ("polarity", "subjectivity", "vader_compound") if col in news_df.columns]
    sentiment_daily = aggregate_daily_sentiment(news_df, columns=base_sentiment_cols)
    aggregated_metrics = [f"avg_{col}" for col in base_sentiment_cols]

    enriched_frames: list[pd.DataFrame] = []
    for ticker, group in prices_df.groupby("ticker"):
        enriched = _apply_indicator_suite(
            group,
            ma_windows=args.ma_windows,
            rsi_window=args.rsi_window,
            boll_window=args.boll_window,
            vol_window=args.vol_window,
            enable_pynance=not args.disable_pynance,
        )
        enriched["ticker"] = ticker
        enriched_frames.append(enriched)

    indicator_df = pd.concat(enriched_frames, ignore_index=True) if enriched_frames else pd.DataFrame()

    correlation_df = correlations_by_ticker(
        sentiment_daily,
        indicator_df,
        sentiment_columns=aggregated_metrics,
        sentiment_shift_days=args.sentiment_shift_days,
    )
    if not correlation_df.empty:
        correlation_df["correlation"] = correlation_df["correlation"].apply(_clean_corr)
    corr_values = correlation_df["correlation"].dropna().tolist() if not correlation_df.empty else []
    overall_corr = float(pd.Series(corr_values).mean()) if corr_values else None

    tfidf_terms = (
        pd.DataFrame(columns=["term", "score"])
        if args.skip_tfidf
        else _safe_tfidf(news_df["headline"], args.tfidf_top_k)
    )
    topics = [] if args.skip_topics or args.topics <= 0 else _safe_topics(news_df["headline"], args.topics, args.topic_words)

    if args.disable_pynance:
        pynance_metrics = {}
    else:
        pynance_metrics = _pynance_summary(indicator_df)

    summary = {
        "tickers_loaded": tickers,
        "headline_length_stats": headline_length_stats(news_df),
        "top_publishers": _records(publisher_activity(news_df)),
        "domain_breakdown": _records(publisher_domain_breakdown(news_df)),
        "hourly_distribution": _records(publishing_hour_distribution(news_df)),
        "daily_articles": _records(daily_article_counts(news_df).tail(30)),
        "tfidf_terms": _records(tfidf_terms),
        "lda_topics": topics,
        "sentiment_daily_tail": _records(sentiment_daily.tail(10)),
        "indicator_snapshot": _indicator_snapshot(indicator_df),
        "pynance_growth_metrics": pynance_metrics,
        "ticker_correlations": _records(correlation_df),
        "overall_correlation": overall_corr,
    }

    payload = json.dumps(summary, indent=2, default=str)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(payload)
        print(f"Saved summary to {output_path}")
    else:
        print(payload)


if __name__ == "__main__":
    main()

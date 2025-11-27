"""Exploratory helpers for the interim analysis."""

from __future__ import annotations

from urllib.parse import urlparse

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def headline_length_stats(df: pd.DataFrame, text_col: str = "headline") -> dict:
    """Return simple stats for headline lengths."""
    lengths = df[text_col].astype(str).str.len()
    stats = lengths.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    return {k: float(v) for k, v in stats.items()}


def publisher_activity(df: pd.DataFrame, publisher_col: str = "publisher", top_n: int = 10) -> pd.DataFrame:
    """Count how many stories each publisher contributed."""
    counts = df[publisher_col].fillna("unknown").value_counts().head(top_n)
    return counts.rename_axis("publisher").reset_index(name="article_count")


def publisher_domain_breakdown(df: pd.DataFrame, url_col: str = "url", top_n: int = 10) -> pd.DataFrame:
    """Extract domains from URLs to understand source concentration."""

    def extract_domain(url: str) -> str:
        hostname = urlparse(str(url)).hostname or "unknown"
        return hostname.replace("www.", "")

    domains = df[url_col].apply(extract_domain)
    counts = domains.value_counts().head(top_n)
    return counts.rename_axis("domain").reset_index(name="article_count")


def daily_article_counts(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Track how many headlines land per calendar day."""
    daily = (
        df.assign(day=df[date_col].dt.date)
        .groupby("day")
        .size()
        .reset_index(name="article_count")
        .rename(columns={"day": "date"})
    )
    return daily


def publishing_hour_distribution(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Summarize article counts by hour of day."""
    timestamps = pd.to_datetime(df[date_col], utc=True)
    hours = timestamps.dt.hour
    counts = hours.value_counts().sort_index()
    return counts.rename_axis("hour").reset_index(name="article_count")


def tfidf_top_phrases(
    texts: pd.Series,
    ngram_range: tuple[int, int] = (1, 2),
    top_k: int = 20,
) -> pd.DataFrame:
    """Return the highest TF-IDF phrases to satisfy topic/keyword KPIs."""

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=ngram_range, max_features=5000)
    matrix = vectorizer.fit_transform(texts.astype(str))
    matrix_sum = cast(Any, matrix).sum(axis=0)
    scores = np.asarray(matrix_sum).ravel()
    terms = vectorizer.get_feature_names_out()
    ranking = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return pd.DataFrame(ranking, columns=["term", "score"])


def lda_topics(texts: pd.Series, n_topics: int = 5, n_top_words: int = 10) -> list[list[str]]:
    """Simple LDA topic modeling over headlines."""

    vectorizer = CountVectorizer(stop_words="english")
    term_matrix = vectorizer.fit_transform(texts.astype(str))
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method="online", random_state=42)
    lda.fit(term_matrix)
    vocab = vectorizer.get_feature_names_out()
    topics: list[list[str]] = []
    for topic in lda.components_:
        top_indices = topic.argsort()[::-1][:n_top_words]
        topic_terms = [str(vocab[int(i)]) for i in top_indices]
        topics.append(topic_terms)
    return topics


def rolling_publisher_mix(df: pd.DataFrame, date_col: str = "date", publisher_col: str = "publisher", window: int = 30) -> pd.DataFrame:
    """Calculate rolling share of top publishers to show concentration shifts."""

    df_sorted = df.sort_values(date_col).copy()
    timestamps = pd.to_datetime(df_sorted[date_col], utc=True)
    df_sorted["publisher"] = df_sorted[publisher_col].fillna("unknown")
    daily_counts = (
        df_sorted.assign(day=timestamps.dt.date)
        .groupby(["day", "publisher"])
        .size()
        .reset_index(name="article_count")
    )
    daily_totals = daily_counts.groupby("day")["article_count"].transform("sum")
    daily_counts["share"] = daily_counts["article_count"] / daily_totals
    daily_counts["rolling_share"] = (
        daily_counts.groupby("publisher")["share"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return daily_counts

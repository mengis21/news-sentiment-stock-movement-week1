"""Small helpers for loading news and stock CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing file: {resolved}")
    return resolved


def load_news_data(
    path: str | Path,
    tickers: Iterable[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load the news dataset with parsed timestamps and optional filters."""

    file_path = _resolve_path(path)
    df = pd.read_csv(file_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["headline"] = df["headline"].fillna("")
    df["publisher"] = df["publisher"].fillna("unknown")
    if "stock" in df.columns:
        df["stock"] = df["stock"].astype(str).str.upper().str.strip()
        if tickers:
            ticker_set = {t.upper() for t in tickers}
            df = df[df["stock"].isin(ticker_set)]

    if start:
        df = df[df["date"] >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end, utc=True)]

    return df.reset_index(drop=True)


def load_stock_data(path: str | Path) -> pd.DataFrame:
    """Load OHLCV stock data and enforce a DateTime index."""
    file_path = _resolve_path(path)
    df = pd.read_csv(file_path)
    date_col = "date" if "date" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.rename(columns={date_col: "date"})
    price_cols: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    df.columns = [c.lower() for c in df.columns]
    for col in price_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in stock data")
    return df.sort_values("date").reset_index(drop=True)


def load_stock_directory(path: str | Path, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Load multiple OHLCV CSVs into one tidy DataFrame."""

    directory = _resolve_path(path)
    data_frames: list[pd.DataFrame] = []
    selected = {t.upper() for t in tickers} if tickers else None

    for csv_file in sorted(directory.glob("*.csv")):
        ticker = csv_file.stem.upper()
        if selected and ticker not in selected:
            continue
        frame = load_stock_data(csv_file)
        frame["ticker"] = ticker
        data_frames.append(frame)

    if not data_frames:
        raise FileNotFoundError("No stock CSV files found for the requested tickers.")

    return pd.concat(data_frames, ignore_index=True)


def load_config(config: dict[str, Any], key: str, fallback: Any) -> Any:
    """Tiny helper so scripts can read optional config entries."""
    return config.get(key, fallback)


def load_analyst_ratings(path: str | Path) -> pd.DataFrame:
    """Load analyst ratings dataset if available."""

    file_path = _resolve_path(path)
    df = pd.read_csv(file_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df

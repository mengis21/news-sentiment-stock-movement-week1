"""Small helpers for loading news and stock CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing file: {resolved}")
    return resolved


def load_news_data(path: str | Path) -> pd.DataFrame:
    """Load the news dataset with parsed timestamps."""
    file_path = _resolve_path(path)
    df = pd.read_csv(file_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["headline"] = df["headline"].fillna("")
    df["publisher"] = df["publisher"].fillna("unknown")
    return df


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


def load_config(config: dict[str, Any], key: str, fallback: Any) -> Any:
    """Tiny helper so scripts can read optional config entries."""
    return config.get(key, fallback)

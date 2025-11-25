"""Lightweight technical indicator helpers without TA-Lib dependency."""

from __future__ import annotations

import pandas as pd


def add_moving_average(df: pd.DataFrame, window: int = 5, price_col: str = "close") -> pd.DataFrame:
    data = df.copy()
    data[f"ma_{window}"] = data[price_col].rolling(window=window, min_periods=1).mean()
    return data


def add_rsi(df: pd.DataFrame, window: int = 14, price_col: str = "close") -> pd.DataFrame:
    data = df.copy()
    delta = data[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    data[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return data


def compute_daily_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    data = df.copy()
    data["daily_return"] = data[price_col].pct_change()
    return data

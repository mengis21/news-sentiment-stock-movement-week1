"""Technical indicator helpers (pandas + ta + PyNance)."""

from __future__ import annotations

import pandas as pd
from math import sqrt
from typing import Any

try:  # Optional TA library for richer indicators
    from ta.trend import MACD
    from ta.volatility import BollingerBands
except ImportError:  # pragma: no cover - ta is optional
    MACD = None  # type: ignore
    BollingerBands = None  # type: ignore

try:  # Optional PyNance helpers for rubric compliance
    import pynance.tech as pn_tech
except Exception:  # pragma: no cover - PyNance may fail on some platforms
    pn_tech = None


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


def compute_daily_returns(df: pd.DataFrame, price_col: str = "close", ticker_col: str = "ticker") -> pd.DataFrame:
    data = df.sort_values("date").copy()
    if ticker_col in data.columns:
        data["daily_return"] = data.groupby(ticker_col)[price_col].pct_change()
    else:
        data["daily_return"] = data[price_col].pct_change()
    return data


def add_macd(df: pd.DataFrame, price_col: str = "close", fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD, signal, and histogram columns."""

    data = df.copy()
    if MACD is not None:
        indicator = MACD(close=data[price_col], window_fast=fast, window_slow=slow, window_sign=signal)
        data["macd"] = indicator.macd()
        data["macd_signal"] = indicator.macd_signal()
        data["macd_hist"] = indicator.macd_diff()
    else:
        ema_fast = data[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = data[price_col].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        data["macd"] = macd
        data["macd_signal"] = signal_line
        data["macd_hist"] = macd - signal_line
    return data


def add_bollinger_bands(df: pd.DataFrame, price_col: str = "close", window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands using `ta` if available, otherwise pandas."""

    data = df.copy()
    if BollingerBands is not None:
        indicator = BollingerBands(close=data[price_col], window=window, window_dev=int(n_std))
        data[f"bb_high_{window}"] = indicator.bollinger_hband()
        data[f"bb_low_{window}"] = indicator.bollinger_lband()
        data[f"bb_mavg_{window}"] = indicator.bollinger_mavg()
    else:
        rolling = data[price_col].rolling(window=window, min_periods=1)
        mavg = rolling.mean()
        std = rolling.std().fillna(0)
        data[f"bb_mavg_{window}"] = mavg
        data[f"bb_high_{window}"] = mavg + n_std * std
        data[f"bb_low_{window}"] = mavg - n_std * std
    return data


def add_volatility(df: pd.DataFrame, price_col: str = "close", window: int = 20) -> pd.DataFrame:
    """Add rolling volatility (annualized)."""

    data = df.copy()
    returns = data[price_col].pct_change()
    data[f"volatility_{window}"] = returns.rolling(window=window).std() * sqrt(252)
    return data


def apply_pynance_bollinger(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 20,
    selection_label: str = "Adj Close",
) -> pd.DataFrame:
    """Compute Bollinger bands via PyNance for rubric alignment."""

    if pn_tech is None:
        raise ImportError("PyNance is not available in this environment")

    adj_df = df[[price_col]].rename(columns={price_col: selection_label})
    boll, sma_df = pn_tech.bollinger(adj_df, window=window, selection=selection_label)
    out = df.copy()
    out[f"pn_boll_upper_{window}"] = boll["Upper"].values
    out[f"pn_boll_lower_{window}"] = boll["Lower"].values
    out[f"pn_sma_{window}"] = sma_df.iloc[:, 0].values
    return out


def pynance_growth_metrics(
    df: pd.DataFrame,
    price_col: str = "close",
    selection_label: str = "Adj Close",
    window: int = 20,
) -> dict[str, float]:
    """Return PyNance growth metrics using the selected price column."""

    if pn_tech is None:
        raise ImportError("PyNance is not available in this environment")

    adj_df = df[[price_col]].rename(columns={price_col: selection_label})
    try:
        growth_series = pn_tech.growth(eqdata=adj_df, selection=selection_label)
        volatility_series = pn_tech.volatility(eqdata=adj_df, selection=selection_label)
        ratio_series = pn_tech.ratio_to_ave(eqdata=adj_df, window=window, selection=selection_label)
    except Exception as exc:  # pragma: no cover - PyNance internals may raise
        raise RuntimeError("PyNance metric computation failed") from exc

    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    return {
        "growth": _to_float(growth_series.iloc[-1]),
        "volatility": _to_float(volatility_series.iloc[-1]),
        "ratio_to_average": _to_float(ratio_series.iloc[-1]),
    }

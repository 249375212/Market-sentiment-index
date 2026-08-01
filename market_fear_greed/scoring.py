from __future__ import annotations

import numpy as np
import pandas as pd


def clamp_score(value: float, neutral: float = 50.0) -> float:
    """把分数限制在 0-100；缺失或异常时返回中性值。"""
    try:
        numeric = float(value)
    except Exception:
        return neutral
    if not np.isfinite(numeric):
        return neutral
    return float(np.clip(numeric, 0.0, 100.0))


def safe_score(value: float, neutral: float = 50.0) -> float:
    """中性化的安全分数入口，便于语义阅读。"""
    return clamp_score(value, neutral=neutral)


def _clean_history(series: pd.Series, lookback_window: int = 252) -> pd.Series:
    clean = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if lookback_window and lookback_window > 0:
        clean = clean.tail(lookback_window)
    return clean


def percentile_score(
    series: pd.Series,
    value: float,
    reverse: bool = False,
    min_periods: int = 60,
    lookback_window: int = 252,
) -> float:
    """
    计算 value 在历史序列中的分位数得分。

    指标缺失或历史样本不足时返回 50；reverse=True 用于波动率等恐惧方向指标。
    """
    clean = _clean_history(series, lookback_window=lookback_window)
    if len(clean) < min_periods:
        return 50.0
    try:
        numeric_value = float(value)
    except Exception:
        return 50.0
    if not np.isfinite(numeric_value):
        return 50.0
    score = float((clean <= numeric_value).mean() * 100.0)
    if reverse:
        score = 100.0 - score
    return clamp_score(score)


def rolling_percentile_score(
    series: pd.Series,
    reverse: bool = False,
    min_periods: int = 60,
    lookback_window: int = 252,
) -> pd.Series:
    """对时间序列逐日滚动计算分位数得分。"""
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = []
    for idx in range(len(numeric)):
        history = numeric.iloc[max(0, idx - lookback_window + 1) : idx + 1]
        values.append(percentile_score(history, numeric.iloc[idx], reverse=reverse, min_periods=min_periods, lookback_window=lookback_window))
    return pd.Series(values, index=series.index, dtype="float64")

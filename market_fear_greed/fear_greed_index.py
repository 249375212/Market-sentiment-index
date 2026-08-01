from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .data_sources import INDEX_CODES, load_index_daily, load_market_fear_greed_source_data
from .indicators import build_raw_indicator_timeseries
from .scoring import rolling_percentile_score, safe_score


SCORE_COLUMNS = [
    "volatility_score",
    "volume_score",
    "price_strength_score",
    "risk_appetite_score",
]

RAW_COLUMNS = [
    "raw_market_volatility",
    "raw_leverage_ratio",
    "raw_amount_ratio",
    "raw_futures_basis",
    "raw_new_high_ratio",
    "raw_risk_appetite",
]


def classify_fear_greed_state(score: float) -> str:
    """AFGI 五档状态分类。"""
    try:
        value = float(score)
    except Exception:
        return "--"
    if not np.isfinite(value):
        return "--"
    if value < 20:
        return "极度恐惧"
    if value < 40:
        return "恐惧"
    if value < 60:
        return "中性"
    if value < 80:
        return "贪婪"
    return "极度贪婪"


def calculate_afgi_from_scores(scores: Dict[str, float] | pd.Series) -> float:
    """核心分项等权平均；缺失分项按 50 处理。"""
    values = [safe_score(scores.get(col, 50.0)) for col in SCORE_COLUMNS]
    return safe_score(float(np.mean(values)))


def _apply_status_fallback(df: pd.DataFrame, score_col: str, status_col: str) -> pd.Series:
    score = df[score_col] if score_col in df.columns else pd.Series(50.0, index=df.index)
    if status_col not in df.columns:
        df[status_col] = "missing"
    status = df[status_col].fillna("missing").astype(str).str.lower()
    valid_mask = status.eq("ok") & pd.to_numeric(score, errors="coerce").notna()
    last_valid_score = pd.to_numeric(score, errors="coerce").where(valid_mask).ffill()
    stale_mask = ~valid_mask & last_valid_score.notna()

    source_date_col = f"{score_col}_source_date"
    source_dates = df.get("trade_date", pd.Series("", index=df.index)).astype(str).where(valid_mask).ffill()
    df[source_date_col] = source_dates
    df.loc[stale_mask, status_col] = "stale"

    message_col = status_col.replace("_status", "_message")
    if message_col not in df.columns:
        df[message_col] = ""
    stale_dates = source_dates.loc[stale_mask].astype(str).str.replace(r"\.0$", "", regex=True)
    df.loc[stale_mask, message_col] = "当日数据源暂不可用，沿用最近有效交易日 " + stale_dates

    return pd.to_numeric(score, errors="coerce").where(valid_mask, last_valid_score).fillna(50.0).map(safe_score)


def score_raw_fear_greed_indicators(raw_df: pd.DataFrame) -> pd.DataFrame:
    """把 5 个入选指标统一标准化为 0-100，分数越高越贪婪。"""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)

    df["volatility_score"] = rolling_percentile_score(df.get("raw_market_volatility", pd.Series(np.nan, index=df.index)), reverse=True)
    df["volume_score"] = rolling_percentile_score(df.get("raw_amount_ratio", pd.Series(np.nan, index=df.index)))
    df["futures_basis_score"] = rolling_percentile_score(df.get("raw_futures_basis", pd.Series(np.nan, index=df.index)))
    df["price_strength_score"] = rolling_percentile_score(df.get("raw_new_high_ratio", pd.Series(np.nan, index=df.index)))
    df["risk_appetite_score"] = rolling_percentile_score(df.get("raw_risk_appetite", pd.Series(np.nan, index=df.index)))

    fallback_pairs = [
        ("volatility_score", "volatility_status"),
        ("volume_score", "volume_status"),
        ("futures_basis_score", "futures_basis_status"),
        ("price_strength_score", "price_strength_status"),
        ("risk_appetite_score", "risk_appetite_status"),
    ]
    for score_col, status_col in fallback_pairs:
        df[score_col] = _apply_status_fallback(df, score_col, status_col)

    df["afgi"] = df.apply(calculate_afgi_from_scores, axis=1)
    df["afgi_state"] = df["afgi"].apply(classify_fear_greed_state)
    df["afgi_change"] = df["afgi"].diff().fillna(0.0)
    df["afgi_ma5"] = df["afgi"].rolling(5, min_periods=1).mean()
    df["afgi_ma20"] = df["afgi"].rolling(20, min_periods=1).mean()
    return df


def _attach_index_closes(token: str, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """为叠加图补充主要指数收盘价；失败时保持 AFGI 本体可用。"""
    out = df.copy()
    for key, code in INDEX_CODES.items():
        if f"{key}_close" in out.columns and out[f"{key}_close"].notna().any():
            continue
        idx = load_index_daily(token, code, start_date, end_date)
        if idx.empty:
            out[f"{key}_close"] = np.nan
            continue
        close = idx[["date", "close"]].rename(columns={"close": f"{key}_close"})
        out = out.merge(close, on="date", how="left")
    return out


def build_fear_greed_timeseries(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """不经过缓存，直接从数据源构建 AFGI 历史序列。"""
    source = load_market_fear_greed_source_data(token, start_date, end_date)
    raw_df = build_raw_indicator_timeseries(source)
    scored = score_raw_fear_greed_indicators(raw_df)
    return _attach_index_closes(token, scored, start_date, end_date) if not scored.empty else scored


def calculate_daily_fear_greed_index(token: str, trade_date: str) -> Dict[str, object]:
    """
    计算单日 AFGI。

    为了保证分位数有足够历史样本，内部会向前预取约 430 个自然日。
    """
    end = pd.Timestamp(trade_date)
    start = end - pd.Timedelta(days=430)
    df = build_fear_greed_timeseries(token, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    if df.empty:
        data_quality = {
            "volatility": "fallback_neutral",
            "volume": "fallback_neutral",
            "futures_basis": "fallback_neutral",
            "price_strength": "fallback_neutral",
            "risk_appetite": "fallback_neutral",
        }
        return {
            "trade_date": pd.Timestamp(trade_date).strftime("%Y%m%d"),
            **{col: 50.0 for col in SCORE_COLUMNS},
            "afgi": 50.0,
            "state": "中性",
            "data_quality": data_quality,
        }
    latest = df.iloc[-1]
    quality_map = {
        "volatility": latest.get("volatility_status", "missing"),
        "volume": latest.get("volume_status", "missing"),
        "futures_basis": latest.get("futures_basis_status", "missing"),
        "price_strength": latest.get("price_strength_status", "missing"),
        "risk_appetite": latest.get("risk_appetite_status", "missing"),
    }
    data_quality = {key: ("ok" if value == "ok" else "fallback_neutral") for key, value in quality_map.items()}
    return {
        "trade_date": str(latest.get("trade_date", pd.Timestamp(trade_date).strftime("%Y%m%d"))),
        **{col: round(float(latest.get(col, 50.0)), 2) for col in SCORE_COLUMNS},
        "afgi": round(float(latest.get("afgi", 50.0)), 2),
        "state": str(latest.get("afgi_state", "中性")),
        "data_quality": data_quality,
    }

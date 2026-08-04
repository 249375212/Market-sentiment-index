from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .cache import load_cache as load_original_afgi_cache, save_cache as save_original_afgi_cache, update_fear_greed_cache
from .data_sources import (
    DATA_DIR,
    INDEX_CODES,
    load_all_stock_daily,
    load_fund_daily,
    load_index_daily,
    load_stock_basic_safe,
    load_trade_dates,
)
from .enhanced_indicators import (
    IndicatorResult,
    calc_industry_up_ratio,
    calc_limit_up_down_ratio,
    calc_ma20_ratio,
    calc_ma60_ratio,
    calc_new_low_60d_ratio,
    calc_open_board_rate,
    calc_small_large_relative_strength,
    calc_stock_industry_up_ratio,
    calc_up_5d_ratio,
    calc_up_ratio,
    calc_yesterday_limit_up_return,
)
from .fear_greed_index import classify_fear_greed_state, score_raw_fear_greed_indicators
from .indicators import (
    calc_market_volume_indicator,
    calc_market_volatility_indicator,
    calc_price_strength_indicator,
    calc_risk_appetite_indicator,
)
from .scoring import rolling_percentile_score, safe_score
from .tushare_client import load_sw_index_history, load_sw_industry_classify


ENHANCED_CACHE_PATH = DATA_DIR / "market_fear_greed_enhanced_cache.csv"
ENHANCED_RAW_DAILY_DIR = DATA_DIR / "enhanced_raw_daily"
ENHANCED_INDUSTRY_DIR = DATA_DIR / "enhanced_industry"
ENHANCED_CACHE_SCHEMA_VERSION = "afgi_enhanced_v3_original_price_strength"
DEFAULT_MAX_UPDATE_DAYS = 80
MIN_PRICE_STRENGTH_HISTORY_DAYS = 252

RAW_INDICATOR_COLUMNS = [
    "up_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "limit_up_down_ratio",
    "open_board_rate",
    "yesterday_limit_up_return",
    "up_5d_ratio",
    "industry_up_ratio",
    "small_large_relative_strength",
    "new_low_60d_ratio",
]

RAW_INDICATOR_META = {
    "up_ratio": ("上涨家数占比", "正向"),
    "ma20_ratio": ("站上20日均线股票占比", "正向"),
    "ma60_ratio": ("站上60日均线股票占比", "正向"),
    "limit_up_down_ratio": ("涨停跌停比", "正向"),
    "open_board_rate": ("炸板率", "反向"),
    "yesterday_limit_up_return": ("昨日涨停今日平均收益", "正向"),
    "up_5d_ratio": ("近5日上涨股票占比", "正向"),
    "industry_up_ratio": ("行业上涨占比", "正向"),
    "small_large_relative_strength": ("小盘/大盘相对强弱", "正向"),
    "new_low_60d_ratio": ("创60日新低股票占比", "反向"),
}

ENHANCED_SCORE_COLUMNS = {
    "up_ratio": "up_ratio_score",
    "ma20_ratio": "ma20_ratio_score",
    "ma60_ratio": "ma60_ratio_score",
    "limit_up_down_ratio": "limit_up_down_ratio_score",
    "open_board_rate": "open_board_rate_score",
    "yesterday_limit_up_return": "yesterday_limit_up_return_score",
    "up_5d_ratio": "up_5d_ratio_score",
    "industry_up_ratio": "industry_up_ratio_score",
    "small_large_relative_strength": "small_large_relative_strength_score",
    "new_low_60d_ratio": "new_low_60d_ratio_score",
}

SENTIMENT_WEIGHTS = {
    "volatility_sentiment": 0.15,
    "volume_sentiment": 0.15,
    "price_strength_sentiment": 0.10,
    "risk_appetite_sentiment": 0.10,
    "breadth_sentiment": 0.15,
    "limit_sentiment": 0.15,
    "profitability_sentiment": 0.10,
    "sector_sentiment": 0.05,
    "style_risk_appetite": 0.05,
}

ENHANCED_SENTIMENT_COLUMNS = list(SENTIMENT_WEIGHTS.keys())


def _normalize_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def _normalize_trade_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    elif "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), errors="coerce")
    else:
        return pd.DataFrame()
    if "trade_date" not in out.columns:
        out["trade_date"] = out["date"].dt.strftime("%Y%m%d")
    out["trade_date"] = out["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)


def _normalize_dated_frame_keep_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    elif "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), errors="coerce")
    else:
        return pd.DataFrame()
    if "trade_date" not in out.columns:
        out["trade_date"] = out["date"].dt.strftime("%Y%m%d")
    out["trade_date"] = out["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    return out.dropna(subset=["date"]).sort_values(["date", "trade_date"]).reset_index(drop=True)


def load_enhanced_cache(cache_path: Path = ENHANCED_CACHE_PATH) -> pd.DataFrame:
    """Read the local enhanced AFGI cache."""
    df = _read_csv_safe(cache_path)
    return _normalize_trade_frame(df)


def get_enhanced_cache_status(
    cache_path: Path = ENHANCED_CACHE_PATH,
    raw_daily_dir: Path = ENHANCED_RAW_DAILY_DIR,
) -> Dict[str, object]:
    """Return the local initialization state used by the Streamlit UI."""
    cache_df = load_enhanced_cache(cache_path)
    raw_dates: List[str] = []
    if raw_daily_dir.exists():
        for path in raw_daily_dir.glob("stock_daily_*.csv"):
            trade_date = path.stem.replace("stock_daily_", "")
            if len(trade_date) == 8 and trade_date.isdigit():
                raw_dates.append(trade_date)
    raw_dates = sorted(set(raw_dates))
    earliest_trade_date = None
    latest_trade_date = None
    if not cache_df.empty and "trade_date" in cache_df.columns:
        earliest_trade_date = str(cache_df["trade_date"].min())
        latest_trade_date = str(cache_df["trade_date"].max())
    history_ready = len(raw_dates) >= MIN_PRICE_STRENGTH_HISTORY_DAYS
    versions = set(
        cache_df.get("enhanced_schema_version", pd.Series(dtype=str)).dropna().astype(str)
    )
    needs_rebuild = not cache_df.empty and versions != {ENHANCED_CACHE_SCHEMA_VERSION}
    return {
        "needs_initialization": cache_df.empty or not history_ready,
        "needs_rebuild": needs_rebuild,
        "history_ready": history_ready,
        "history_trade_days": len(raw_dates),
        "cached_rows": len(cache_df),
        "earliest_trade_date": earliest_trade_date,
        "latest_trade_date": latest_trade_date,
    }


def plan_enhanced_update_ranges(
    cache_status: Dict[str, object],
    requested_start_date: str,
    requested_end_date: str,
    latest_date: Optional[str] = None,
) -> List[tuple[str, str]]:
    """Plan backward and forward cache updates without recalculating the middle."""
    requested_start = pd.Timestamp(requested_start_date)
    requested_end = pd.Timestamp(requested_end_date)
    latest = pd.Timestamp(latest_date) if latest_date else requested_end
    if bool(cache_status.get("needs_initialization", True)) or bool(cache_status.get("needs_rebuild", False)):
        return [(requested_start.strftime("%Y%m%d"), requested_end.strftime("%Y%m%d"))]

    ranges: List[tuple[str, str]] = []
    earliest_cached = pd.to_datetime(cache_status.get("earliest_trade_date"), errors="coerce")
    latest_cached = pd.to_datetime(cache_status.get("latest_trade_date"), errors="coerce")

    if pd.notna(earliest_cached) and requested_start < earliest_cached:
        backward_end = min(requested_end, earliest_cached - pd.Timedelta(days=1))
        if requested_start <= backward_end:
            ranges.append((requested_start.strftime("%Y%m%d"), backward_end.strftime("%Y%m%d")))

    forward_end = max(requested_end, latest)
    if pd.notna(latest_cached) and latest_cached <= forward_end:
        forward_start = max(latest_cached, requested_start)
        ranges.append((forward_start.strftime("%Y%m%d"), forward_end.strftime("%Y%m%d")))

    if not ranges:
        ranges.append((requested_start.strftime("%Y%m%d"), requested_end.strftime("%Y%m%d")))
    return ranges


def save_enhanced_cache(df: pd.DataFrame, cache_path: Path = ENHANCED_CACHE_PATH) -> None:
    """Save enhanced AFGI rows to the local CSV cache."""
    if df is None or df.empty:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = _normalize_trade_frame(df)
    out["enhanced_schema_version"] = ENHANCED_CACHE_SCHEMA_VERSION
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")


def load_stock_daily_cached(token: str, trade_date: str) -> pd.DataFrame:
    """Load one trading day's all-stock daily quote with a persistent CSV cache."""
    trade_date = str(trade_date)
    path = ENHANCED_RAW_DAILY_DIR / f"stock_daily_{trade_date}.csv"
    cached = _read_csv_safe(path)
    if not cached.empty:
        return cached
    if not token:
        return pd.DataFrame()
    try:
        df = load_all_stock_daily(token, trade_date)
    except Exception:
        df = pd.DataFrame()
    if df is not None and not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return df if df is not None else pd.DataFrame()


def _local_stock_daily_dates(start_date: str, end_date: str) -> List[str]:
    if not ENHANCED_RAW_DAILY_DIR.exists():
        return []
    start_date, end_date = str(start_date), str(end_date)
    dates = []
    for path in ENHANCED_RAW_DAILY_DIR.glob("stock_daily_*.csv"):
        date = path.stem.replace("stock_daily_", "")
        if len(date) == 8 and start_date <= date <= end_date:
            dates.append(date)
    return sorted(set(dates))


def load_industry_spot_cached(trade_date: str) -> pd.DataFrame:
    """Load current industry board data with cache; historical dates fallback to empty."""
    trade_date = str(trade_date)
    path = ENHANCED_INDUSTRY_DIR / f"industry_spot_{trade_date}.csv"
    cached = _read_csv_safe(path)
    if not cached.empty:
        return cached
    # AkShare's EM board spot endpoint is a current snapshot. Use it only as a
    # best-effort source; stale/historical dates will naturally fall back to 50.
    today = pd.Timestamp.today().strftime("%Y%m%d")
    if trade_date != today:
        return pd.DataFrame()
    try:
        import akshare as ak

        df = ak.stock_board_industry_name_em()
    except Exception:
        return pd.DataFrame()
    if df is not None and not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return df if df is not None else pd.DataFrame()


def _industry_frame_for_date(industry_history_df: Optional[pd.DataFrame], trade_date: str) -> pd.DataFrame:
    """Prefer historical industry data, then use today's live industry snapshot."""
    industry_df = pd.DataFrame()
    if industry_history_df is not None and not industry_history_df.empty and "trade_date" in industry_history_df.columns:
        industry_df = industry_history_df.loc[
            industry_history_df["trade_date"].astype(str) == str(trade_date)
        ].copy()
    if industry_df.empty:
        industry_df = load_industry_spot_cached(str(trade_date))
    return industry_df


def _load_cached_industry_history_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Load any overlapping SW L1 industry history cache files for a date range."""
    frames = []
    for path in ENHANCED_INDUSTRY_DIR.glob("sw_l1_history_*.csv"):
        parts = path.stem.replace("sw_l1_history_", "").split("_")
        if len(parts) != 2:
            continue
        file_start, file_end = parts
        if file_end < start_date or file_start > end_date:
            continue
        cached = _read_csv_safe(path)
        if cached.empty:
            continue
        cached = _normalize_dated_frame_keep_rows(cached)
        if cached.empty:
            continue
        cached = cached.loc[(cached["trade_date"] >= start_date) & (cached["trade_date"] <= end_date)].copy()
        if not cached.empty:
            frames.append(cached)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = _normalize_dated_frame_keep_rows(out)
    dedupe_cols = [col for col in ["trade_date", "industry_code"] if col in out.columns]
    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="last")
    return out.reset_index(drop=True)


def load_industry_history_cached(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load historical SW L1 industry index returns and cache them for sector breadth."""
    start_date, end_date = _normalize_date(start_date), _normalize_date(end_date)
    path = ENHANCED_INDUSTRY_DIR / f"sw_l1_history_{start_date}_{end_date}.csv"
    cached = _read_csv_safe(path)
    if not cached.empty:
        return _normalize_dated_frame_keep_rows(cached)
    cached_range = _load_cached_industry_history_range(start_date, end_date)
    if not cached_range.empty:
        return cached_range
    if not token:
        return pd.DataFrame()
    try:
        classify = load_sw_industry_classify(token, level="L1", src="SW2021")
    except Exception:
        classify = pd.DataFrame()
    if classify is None or classify.empty or "index_code" not in classify.columns:
        return pd.DataFrame()

    frames = []
    for _, row in classify.drop_duplicates(subset=["index_code"]).iterrows():
        code = str(row.get("index_code", "")).strip()
        name = str(row.get("industry_name", "")).strip()
        if not code:
            continue
        try:
            hist = load_sw_index_history(token, code, start_date, end_date)
        except Exception:
            continue
        if hist is None or hist.empty:
            continue
        hist = hist.copy()
        hist["industry_code"] = code
        hist["industry_name"] = name
        frames.append(hist)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = _normalize_dated_frame_keep_rows(out)
    pct_col = "pct_chg" if "pct_chg" in out.columns else None
    if pct_col is None:
        return pd.DataFrame()
    keep_cols = [col for col in ["date", "trade_date", "industry_code", "industry_name", "pct_chg"] if col in out.columns]
    out = out[keep_cols].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_out = out.copy()
    save_out["date"] = pd.to_datetime(save_out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    save_out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def _status_to_columns(prefix: str, result: IndicatorResult) -> Dict[str, object]:
    return {
        prefix: result.get("value", np.nan),
        f"{prefix}_status": result.get("status", "missing"),
        f"{prefix}_source": result.get("source", ""),
        f"{prefix}_message": result.get("message", ""),
    }


def _concat_daily_frames(daily_map: Dict[str, pd.DataFrame], trade_dates: Iterable[str]) -> pd.DataFrame:
    frames = [daily_map.get(date, pd.DataFrame()) for date in trade_dates]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _index_history_for_date(index_df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if index_df is None or index_df.empty:
        return pd.DataFrame()
    out = _normalize_trade_frame(index_df)
    return out.loc[out["trade_date"] <= trade_date].copy()


def calc_enhanced_raw_indicators_for_date(
    token: str,
    trade_date: str,
    previous_trade_date: Optional[str],
    history_trade_dates: List[str],
    daily_map: Dict[str, pd.DataFrame],
    stock_basic_df: pd.DataFrame,
    index_history_df: pd.DataFrame,
    industry_history_df: Optional[pd.DataFrame] = None,
    history_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Calculate all ten enhanced raw A-share sentiment indicators for one day."""
    current_df = daily_map.get(trade_date, pd.DataFrame())
    previous_df = daily_map.get(previous_trade_date or "", pd.DataFrame())
    if history_df is None:
        history_df = _concat_daily_frames(daily_map, history_trade_dates)
    industry_df = _industry_frame_for_date(industry_history_df, trade_date)
    index_df = _index_history_for_date(index_history_df, trade_date)

    industry_result = calc_industry_up_ratio(industry_df)
    if industry_result.get("status") != "ok":
        industry_result = calc_stock_industry_up_ratio(current_df, stock_basic_df)

    results = {
        "up_ratio": calc_up_ratio(current_df, stock_basic_df),
        "ma20_ratio": calc_ma20_ratio(history_df, current_df, stock_basic_df),
        "ma60_ratio": calc_ma60_ratio(history_df, current_df, stock_basic_df),
        "limit_up_down_ratio": calc_limit_up_down_ratio(current_df, stock_basic_df),
        "open_board_rate": calc_open_board_rate(current_df, stock_basic_df),
        "yesterday_limit_up_return": calc_yesterday_limit_up_return(previous_df, current_df, stock_basic_df),
        "up_5d_ratio": calc_up_5d_ratio(history_df, current_df, stock_basic_df),
        "industry_up_ratio": industry_result,
        "small_large_relative_strength": calc_small_large_relative_strength(index_df, window=5),
        "new_low_60d_ratio": calc_new_low_60d_ratio(history_df, current_df, stock_basic_df),
    }

    row: Dict[str, object] = {"trade_date": trade_date, "date": pd.to_datetime(trade_date)}
    for key, value in results.items():
        row.update(_status_to_columns(key, value))
    return row


def calc_breadth_sentiment(row: pd.Series) -> float:
    """市场广度情绪：上涨占比、MA20、MA60 三项加权。"""
    return safe_score(
        0.40 * row.get("up_ratio_score", 50.0)
        + 0.35 * row.get("ma20_ratio_score", 50.0)
        + 0.25 * row.get("ma60_ratio_score", 50.0)
    )


def calc_limit_sentiment(row: pd.Series) -> float:
    """涨跌停情绪：涨停跌停比、炸板率、昨日涨停收益三项加权。"""
    return safe_score(
        0.40 * row.get("limit_up_down_ratio_score", 50.0)
        + 0.30 * row.get("open_board_rate_score", 50.0)
        + 0.30 * row.get("yesterday_limit_up_return_score", 50.0)
    )


def calc_profitability_sentiment(row: pd.Series) -> float:
    """赚钱效应：近5日上涨股票占比和创60日新低占比加权。"""
    return safe_score(0.60 * row.get("up_5d_ratio_score", 50.0) + 0.40 * row.get("new_low_60d_ratio_score", 50.0))


def calc_sector_sentiment(row: pd.Series) -> float:
    """板块扩散情绪：第一版使用行业上涨占比得分。"""
    return safe_score(row.get("industry_up_ratio_score", 50.0))


def calc_style_risk_appetite(row: pd.Series) -> float:
    """风格风险偏好：第一版使用中证1000相对沪深300强弱得分。"""
    return safe_score(row.get("small_large_relative_strength_score", 50.0))


def calc_afgi_enhanced(row: pd.Series) -> float:
    """Calculate enhanced AFGI using the fixed first-version weights."""
    total = 0.0
    for col, weight in SENTIMENT_WEIGHTS.items():
        total += weight * safe_score(row.get(col, 50.0))
    return safe_score(total)


def _score_raw_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reverse_cols = {"open_board_rate", "new_low_60d_ratio"}
    for raw_col, score_col in ENHANCED_SCORE_COLUMNS.items():
        if raw_col in out.columns:
            out[score_col] = rolling_percentile_score(out[raw_col], reverse=raw_col in reverse_cols)
        else:
            out[score_col] = 50.0
        status_col = f"{raw_col}_status"
        if status_col in out.columns:
            out[score_col] = out[score_col].where(out[status_col].astype(str).eq("ok"), 50.0)
        out[score_col] = out[score_col].map(safe_score)
    return out


def _attach_original_afgi(
    token: str,
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    force_update: bool = False,
    original_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if original_df is not None:
        original = original_df
    else:
        try:
            original = update_fear_greed_cache(token, start_date, end_date, force_update=force_update)
        except Exception:
            original = load_original_afgi_cache()
    original = _normalize_trade_frame(original)
    if original.empty:
        return df
    keep_cols = [
        "date",
        "trade_date",
        "afgi",
        "afgi_state",
        "afgi_change",
        "afgi_ma5",
        "afgi_ma20",
        "volatility_score",
        "volume_score",
        "futures_basis_score",
        "price_strength_score",
        "risk_appetite_score",
        "hs300_close",
        "sh_close",
        "zz1000_close",
        "cyb_close",
    ]
    for prefix in ("volatility", "volume", "price_strength", "risk_appetite"):
        keep_cols.extend(
            [
                f"{prefix}_status",
                f"{prefix}_message",
                f"{prefix}_score_source_date",
            ]
        )
    keep_cols = [col for col in keep_cols if col in original.columns]
    out = _normalize_trade_frame(df).merge(original[keep_cols], on=["date", "trade_date"], how="left")
    out = out.rename(columns={"afgi": "afgi_original"})
    return out


def _finalize_enhanced_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = _score_raw_indicators(_normalize_trade_frame(df))
    core_map = {
        "volatility_sentiment": "volatility_score",
        "volume_sentiment": "volume_score",
        "futures_basis_sentiment": "futures_basis_score",
        "price_strength_sentiment": "price_strength_score",
        "risk_appetite_sentiment": "risk_appetite_score",
    }
    for target, source in core_map.items():
        out[target] = out[source].map(safe_score) if source in out.columns else 50.0

    core_status_map = {
        "volatility_sentiment": "volatility_status",
        "volume_sentiment": "volume_status",
        "price_strength_sentiment": "price_strength_status",
        "risk_appetite_sentiment": "risk_appetite_status",
    }
    for target, source in core_status_map.items():
        out[f"{target}_status"] = (
            out[source].fillna("missing").astype(str).str.lower()
            if source in out.columns
            else "missing"
        )

    out["breadth_sentiment"] = out.apply(calc_breadth_sentiment, axis=1)
    out["limit_sentiment"] = out.apply(calc_limit_sentiment, axis=1)
    out["profitability_sentiment"] = out.apply(calc_profitability_sentiment, axis=1)
    out["sector_sentiment"] = out.apply(calc_sector_sentiment, axis=1)
    out["style_risk_appetite"] = out.apply(calc_style_risk_appetite, axis=1)

    component_sources = {
        "breadth_sentiment": ("up_ratio_status", "ma20_ratio_status", "ma60_ratio_status"),
        "limit_sentiment": (
            "limit_up_down_ratio_status",
            "open_board_rate_status",
            "yesterday_limit_up_return_status",
        ),
        "profitability_sentiment": ("up_5d_ratio_status", "new_low_60d_ratio_status"),
        "sector_sentiment": ("industry_up_ratio_status",),
        "style_risk_appetite": ("small_large_relative_strength_status",),
    }
    available_statuses = {"ok", "stale"}
    for target, source_cols in component_sources.items():
        statuses = pd.DataFrame(
            {
                col: (
                    out[col].fillna("missing").astype(str).str.lower()
                    if col in out.columns
                    else pd.Series("missing", index=out.index)
                )
                for col in source_cols
            },
            index=out.index,
        )
        all_ok = statuses.eq("ok").all(axis=1)
        all_available = statuses.isin(available_statuses).all(axis=1)
        any_available = statuses.isin(available_statuses).any(axis=1)
        component_status = pd.Series("missing", index=out.index, dtype="object")
        component_status.loc[any_available] = "partial"
        component_status.loc[all_available] = "stale"
        component_status.loc[all_ok] = "ok"
        out[f"{target}_status"] = component_status

    out["afgi_enhanced"] = out.apply(calc_afgi_enhanced, axis=1)
    out["afgi_enhanced_state"] = out["afgi_enhanced"].apply(classify_fear_greed_state)
    out["afgi_enhanced_change"] = out["afgi_enhanced"].diff().fillna(0.0)
    out["afgi_enhanced_ma5"] = out["afgi_enhanced"].rolling(5, min_periods=1).mean()
    out["afgi_enhanced_ma20"] = out["afgi_enhanced"].rolling(20, min_periods=1).mean()
    if "afgi_original" in out.columns:
        out["afgi_enhanced_diff"] = out["afgi_enhanced"] - pd.to_numeric(out["afgi_original"], errors="coerce")
    else:
        out["afgi_enhanced_diff"] = np.nan
    return out


def _neutral_enhanced_from_original(original: pd.DataFrame) -> pd.DataFrame:
    """Build a displayable enhanced series from original AFGI cache when raw data is unavailable."""
    base = _normalize_trade_frame(original)
    if base.empty:
        return pd.DataFrame()
    out = base[["date", "trade_date"]].copy()
    out["afgi_original"] = pd.to_numeric(base.get("afgi", 50.0), errors="coerce")
    for raw_col in RAW_INDICATOR_COLUMNS:
        out[raw_col] = np.nan
        out[f"{raw_col}_status"] = "fallback"
        out[f"{raw_col}_source"] = "fallback_neutral"
        out[f"{raw_col}_message"] = "缺少新增指标原始数据，使用中性分"
    for col in ["volatility_score", "volume_score", "futures_basis_score", "price_strength_score", "risk_appetite_score", "hs300_close", "sh_close", "zz1000_close", "cyb_close"]:
        if col in base.columns:
            out[col] = base[col]
    return _finalize_enhanced_scores(out)


def _filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    out = _normalize_trade_frame(df)
    if out.empty:
        return out
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    return out.loc[(out["date"] >= start_ts) & (out["date"] <= end_ts)].reset_index(drop=True)


def _effective_cached_dates(cache_df: pd.DataFrame, token: str) -> set:
    if cache_df is None or cache_df.empty:
        return set()
    out = _normalize_trade_frame(cache_df)
    if out.empty or "trade_date" not in out.columns:
        return set()
    if not token:
        return set(out["trade_date"].astype(str))
    status_cols = [f"{component}_status" for component in SENTIMENT_WEIGHTS]
    if any(col not in out.columns for col in status_cols):
        return set()
    valid_mask = pd.Series(True, index=out.index)
    invalid_status = {"fallback", "fallback_neutral", "missing", "stale", "error", "nan", ""}
    for status_col in status_cols:
        status = out[status_col].astype(str).str.lower()
        valid_mask &= ~status.isin(invalid_status)
    valid = out.loc[valid_mask]
    return set(valid["trade_date"].astype(str))


def _build_index_history(token: str, base_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    base = _normalize_trade_frame(base_df)
    close_cols = [f"{key}_close" for key in INDEX_CODES]
    if not base.empty:
        keep_cols = ["date", "trade_date"] + [col for col in close_cols if col in base.columns]
        out = base[keep_cols].copy()
    else:
        out = pd.DataFrame()

    for key, ts_code in INDEX_CODES.items():
        col = f"{key}_close"
        if not token:
            continue
        try:
            idx = load_index_daily(token, ts_code, start_date, end_date)
        except Exception:
            idx = pd.DataFrame()
        if idx is None or idx.empty:
            continue
        frame = _normalize_trade_frame(idx)[["date", "trade_date", "close"]].rename(columns={"close": col})
        if out.empty:
            out = frame
            continue
        out = out.merge(frame, on=["date", "trade_date"], how="outer", suffixes=("", "_fresh"))
        fresh_col = f"{col}_fresh"
        if fresh_col in out.columns:
            if col in out.columns:
                out[col] = pd.to_numeric(out[fresh_col], errors="coerce").combine_first(pd.to_numeric(out[col], errors="coerce"))
            else:
                out[col] = out[fresh_col]
            out = out.drop(columns=[fresh_col])

    if out.empty:
        return pd.DataFrame()
    return _normalize_trade_frame(out)


def _supplement_original_core_indicators(
    token: str,
    original_df: pd.DataFrame,
    daily_map: Dict[str, pd.DataFrame],
    stock_basic_df: pd.DataFrame,
    index_history_df: pd.DataFrame,
    target_dates: List[str],
) -> pd.DataFrame:
    """Compute the four core factors from lightweight current-day sources."""
    base = _normalize_trade_frame(original_df)
    if not target_dates:
        return base

    missing_rows = pd.DataFrame(
        {
            "trade_date": [date for date in target_dates if date not in set(base.get("trade_date", pd.Series(dtype=str)).astype(str))],
        }
    )
    if not missing_rows.empty:
        missing_rows["date"] = pd.to_datetime(missing_rows["trade_date"], errors="coerce")
        base = pd.concat([base, missing_rows], ignore_index=True, sort=False)
        base = _normalize_trade_frame(base)

    all_daily = _concat_daily_frames(daily_map, sorted(daily_map))
    hs300 = pd.DataFrame()
    if index_history_df is not None and not index_history_df.empty and "hs300_close" in index_history_df.columns:
        hs300 = index_history_df[["date", "trade_date", "hs300_close"]].rename(
            columns={"hs300_close": "close"}
        )
        hs300 = _normalize_trade_frame(hs300)

    history_start = min(str(date) for date in target_dates)
    etf50 = pd.DataFrame()
    if token and not hs300.empty:
        end_ts = pd.to_datetime(max(target_dates))
        try:
            etf50 = load_fund_daily(token, "510050.SH", history_start, end_ts.strftime("%Y%m%d"))
        except Exception:
            etf50 = pd.DataFrame()
    volatility = calc_market_volatility_indicator(etf50, hs300)
    volume = calc_market_volume_indicator(all_daily, {})
    price_strength = calc_price_strength_indicator(all_daily, stock_basic_df)

    bond = pd.DataFrame()
    if token and not hs300.empty:
        end_ts = pd.to_datetime(max(target_dates))
        try:
            bond = load_fund_daily(token, "511010.SH", history_start, end_ts.strftime("%Y%m%d"))
        except Exception:
            bond = pd.DataFrame()
    risk_appetite = calc_risk_appetite_indicator(hs300, bond)

    frame_specs = (
        (volatility, "volatility_status"),
        (volume, "volume_status"),
        (price_strength, "price_strength_status"),
        (risk_appetite, "risk_appetite_status"),
    )
    target_set = set(str(date) for date in target_dates)
    for frame, status_col in frame_specs:
        normalized = _normalize_trade_frame(frame)
        if normalized.empty or status_col not in normalized.columns:
            continue
        normalized = normalized.loc[normalized["trade_date"].astype(str).isin(target_set)].copy()
        for _, row in normalized.iterrows():
            if str(row.get(status_col, "missing")).lower() != "ok":
                continue
            trade_date = str(row["trade_date"])
            mask = base["trade_date"].astype(str).eq(trade_date)
            for col, value in row.items():
                if col in {"date", "trade_date"}:
                    continue
                if col not in base.columns:
                    base[col] = pd.Series(dtype=object)
                elif isinstance(value, str) and not pd.api.types.is_object_dtype(base[col]):
                    base[col] = base[col].astype(object)
                base.loc[mask, col] = value

    if index_history_df is not None and not index_history_df.empty:
        closes = _normalize_trade_frame(index_history_df)
        closes = closes.loc[closes["trade_date"].astype(str).isin(target_set)]
        for _, row in closes.iterrows():
            mask = base["trade_date"].astype(str).eq(str(row["trade_date"]))
            for col in (f"{key}_close" for key in INDEX_CODES):
                value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                if pd.notna(value):
                    if col not in base.columns:
                        base[col] = np.nan
                    base.loc[mask, col] = value

    rescored = score_raw_fear_greed_indicators(base)
    for col in [c for c in base.columns if c.endswith("_close") and c not in rescored.columns]:
        rescored[col] = base[col]
    save_original_afgi_cache(rescored)
    return rescored


def _limit_threshold_fast(ts_code: pd.Series) -> pd.Series:
    code = ts_code.fillna("").astype(str)
    threshold = pd.Series(0.095, index=code.index, dtype="float64")
    threshold.loc[code.str.startswith(("300", "301", "688", "689"))] = 0.195
    threshold.loc[code.str.contains(".BJ", regex=False) | code.str.startswith(("8", "4"))] = 0.295
    return threshold


def _numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _prepare_daily_panel(daily_map: Dict[str, pd.DataFrame], trade_dates: List[str]) -> pd.DataFrame:
    all_daily = _concat_daily_frames(daily_map, trade_dates)
    if all_daily.empty or "ts_code" not in all_daily.columns:
        return pd.DataFrame()
    all_daily = _normalize_dated_frame_keep_rows(all_daily)
    all_daily["ts_code"] = all_daily["ts_code"].astype(str)
    for col in ["close", "pre_close", "pct_chg", "high", "low", "vol", "amount"]:
        if col in all_daily.columns:
            all_daily[col] = _numeric_col(all_daily, col)
    valid = all_daily["close"].notna() & (all_daily["close"] > 0)
    if "vol" in all_daily.columns:
        valid &= all_daily["vol"].fillna(0) > 0
    all_daily = all_daily.loc[valid].copy()
    if "pct_chg" in all_daily.columns:
        all_daily["ret"] = all_daily["pct_chg"] / 100.0
    else:
        all_daily["ret"] = all_daily["close"] / all_daily["pre_close"] - 1.0
    all_daily["limit_threshold"] = _limit_threshold_fast(all_daily["ts_code"])
    all_daily["is_limit_up"] = all_daily["ret"] >= all_daily["limit_threshold"]
    all_daily["is_limit_down"] = all_daily["ret"] <= -all_daily["limit_threshold"]
    all_daily["touched_limit_up"] = (all_daily["high"] / all_daily["pre_close"] - 1.0) >= all_daily["limit_threshold"]
    return all_daily.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def _build_stock_raw_rows_fast(
    daily_map: Dict[str, pd.DataFrame],
    trade_dates: List[str],
    target_dates: List[str],
    stock_basic_df: pd.DataFrame,
    index_history: pd.DataFrame,
    industry_history: pd.DataFrame,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Vectorized local-cache builder for stock-based enhanced indicators."""
    panel = _prepare_daily_panel(daily_map, trade_dates)
    if panel.empty:
        return pd.DataFrame()

    grouped = panel.groupby("trade_date", sort=True)
    count = grouped["ts_code"].count()
    up_ratio = _ratio_series(grouped["ret"].apply(lambda s: (s > 0).sum()), count)
    limit_up_count = grouped["is_limit_up"].sum()
    limit_down_count = grouped["is_limit_down"].sum()
    limit_ratio = (limit_up_count / limit_down_count.clip(lower=1)).replace([np.inf, -np.inf], np.nan)
    touched_count = grouped["touched_limit_up"].sum()
    opened_count = panel.assign(opened=panel["touched_limit_up"] & ~panel["is_limit_up"]).groupby("trade_date")["opened"].sum()
    open_board_rate = _ratio_series(opened_count, touched_count).fillna(0.0)

    close_panel = panel.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last").sort_index()
    ma20 = close_panel.rolling(20, min_periods=20).mean()
    ma60 = close_panel.rolling(60, min_periods=60).mean()
    ret5 = close_panel / close_panel.shift(5) - 1.0
    low60 = close_panel.rolling(60, min_periods=60).min()

    ma20_ratio = _ratio_series((close_panel > ma20).sum(axis=1), ma20.notna().sum(axis=1))
    ma60_ratio = _ratio_series((close_panel > ma60).sum(axis=1), ma60.notna().sum(axis=1))
    up5_ratio = _ratio_series((ret5 > 0).sum(axis=1), ret5.notna().sum(axis=1))
    new_low60_ratio = _ratio_series((close_panel <= low60).sum(axis=1), low60.notna().sum(axis=1))

    trade_order = [date for date in close_panel.index.astype(str).tolist()]
    previous_date = {date: trade_order[idx - 1] if idx > 0 else None for idx, date in enumerate(trade_order)}
    limit_sets = {
        date: set(group.loc[group["is_limit_up"], "ts_code"].astype(str))
        for date, group in panel.groupby("trade_date", sort=True)
    }
    today_returns = {
        date: group.set_index("ts_code")["ret"]
        for date, group in panel.groupby("trade_date", sort=True)
    }

    rows = []
    total = len(target_dates)
    for pos, trade_date in enumerate(target_dates, start=1):
        prev = previous_date.get(trade_date)
        prev_limit_codes = limit_sets.get(prev, set()) if prev else set()
        today_ret = today_returns.get(trade_date, pd.Series(dtype="float64"))
        y_limit_ret = today_ret.loc[today_ret.index.astype(str).isin(prev_limit_codes)].dropna()
        industry_df = _industry_frame_for_date(industry_history, trade_date)
        industry_result = calc_industry_up_ratio(industry_df)
        if industry_result.get("status") != "ok":
            current_stock_df = grouped.get_group(trade_date).copy() if trade_date in grouped.groups else pd.DataFrame()
            industry_result = calc_stock_industry_up_ratio(current_stock_df, stock_basic_df)
        style_result = calc_small_large_relative_strength(_index_history_for_date(index_history, trade_date), window=5)

        row: Dict[str, object] = {"trade_date": trade_date, "date": pd.to_datetime(trade_date)}
        metric_specs = {
            "up_ratio": (up_ratio.get(trade_date, np.nan), "ok" if pd.notna(up_ratio.get(trade_date, np.nan)) else "missing", f"有效股票 {int(count.get(trade_date, 0))} 只", "local stock_daily cache"),
            "ma20_ratio": (ma20_ratio.get(trade_date, np.nan), "ok" if pd.notna(ma20_ratio.get(trade_date, np.nan)) else "missing", "本地日行情滚动 MA20", "local stock_daily cache"),
            "ma60_ratio": (ma60_ratio.get(trade_date, np.nan), "ok" if pd.notna(ma60_ratio.get(trade_date, np.nan)) else "missing", "本地日行情滚动 MA60", "local stock_daily cache"),
            "limit_up_down_ratio": (limit_ratio.get(trade_date, np.nan), "ok" if pd.notna(limit_ratio.get(trade_date, np.nan)) else "missing", "本地日行情近似涨跌停", "local stock_daily cache"),
            "open_board_rate": (open_board_rate.get(trade_date, np.nan), "ok" if pd.notna(open_board_rate.get(trade_date, np.nan)) else "missing", "本地日行情近似炸板率", "local stock_daily cache"),
            "yesterday_limit_up_return": (float(y_limit_ret.mean()) if len(y_limit_ret) else np.nan, "ok" if len(y_limit_ret) else "fallback", f"昨日涨停样本 {len(y_limit_ret)} 只", "local stock_daily cache"),
            "up_5d_ratio": (up5_ratio.get(trade_date, np.nan), "ok" if pd.notna(up5_ratio.get(trade_date, np.nan)) else "missing", "本地日行情 5 日收益", "local stock_daily cache"),
            "new_low_60d_ratio": (new_low60_ratio.get(trade_date, np.nan), "ok" if pd.notna(new_low60_ratio.get(trade_date, np.nan)) else "missing", "本地日行情 60 日新低", "local stock_daily cache"),
        }
        for key, (value, status, message, source) in metric_specs.items():
            row.update({key: value, f"{key}_status": status, f"{key}_source": source, f"{key}_message": message})
        row.update(_status_to_columns("industry_up_ratio", industry_result))
        row.update(_status_to_columns("small_large_relative_strength", style_result))
        rows.append(row)
        if progress_callback is not None:
            progress_callback(pos, total, f"正在快速回填 {trade_date} 新增指标")
    return pd.DataFrame(rows)


def build_afgi_enhanced_timeseries(
    token: str,
    start_date: str,
    end_date: str,
    force_update: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Build enhanced AFGI rows directly from available data sources without using the enhanced cache."""
    start_date = _normalize_date(start_date)
    end_date = _normalize_date(end_date)
    # Every requested date needs enough prior closes for the 252-trading-day
    # price-strength factor, including dates added before the existing cache.
    source_start = (pd.to_datetime(start_date) - pd.Timedelta(days=430)).strftime("%Y%m%d")
    original = load_original_afgi_cache()
    original = _normalize_trade_frame(original)
    if progress_callback is not None:
        progress_callback(0, 1, "正在准备历史交易日")
    try:
        trade_dates = load_trade_dates(token, source_start, end_date) if token else []
    except Exception:
        trade_dates = []
    local_history_start = (pd.to_datetime(start_date) - pd.Timedelta(days=430)).strftime("%Y%m%d")
    local_dates = _local_stock_daily_dates(local_history_start, end_date)
    if local_dates:
        trade_dates = sorted(set(trade_dates).union(local_dates))
    trade_dates = [date for date in trade_dates if date <= end_date]

    stock_basic = load_stock_basic_safe(token)
    daily_map: Dict[str, pd.DataFrame] = {}
    total_trade_dates = max(len(trade_dates), 1)
    for position, trade_date in enumerate(trade_dates, start=1):
        daily_map[trade_date] = load_stock_daily_cached(token, trade_date)
        if progress_callback is not None:
            data_scope = "前置计算" if trade_date < start_date else "所选区间"
            progress_callback(position, total_trade_dates, f"正在获取历史行情（{data_scope}）{trade_date}")
    trade_dates = [date for date in trade_dates if not daily_map[date].empty]
    target_dates = [date for date in trade_dates if start_date <= date <= end_date]
    if not target_dates:
        return pd.DataFrame()

    index_history = _build_index_history(token, original, source_start, end_date)
    original = _supplement_original_core_indicators(
        token,
        original,
        daily_map,
        stock_basic,
        index_history,
        trade_dates,
    )
    industry_history = load_industry_history_cached(token, start_date, end_date)

    fast_raw_df = _build_stock_raw_rows_fast(
        daily_map,
        trade_dates,
        target_dates,
        stock_basic,
        index_history,
        industry_history,
        progress_callback,
    )
    if not fast_raw_df.empty:
        merged = _attach_original_afgi(
            token,
            fast_raw_df,
            source_start,
            end_date,
            force_update=False,
            original_df=original,
        )
        return _filter_date_range(_finalize_enhanced_scores(merged), start_date, end_date)

    all_daily = _concat_daily_frames(daily_map, trade_dates)
    if not all_daily.empty and "trade_date" in all_daily.columns:
        all_daily["trade_date"] = all_daily["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)

    rows = []
    total_targets = len(target_dates)
    completed_targets = 0
    for idx, trade_date in enumerate(trade_dates):
        if trade_date not in target_dates:
            continue
        prior_dates = trade_dates[max(0, idx - 70) : idx + 1]
        previous_date = trade_dates[idx - 1] if idx > 0 else None
        if all_daily.empty:
            history_df = pd.DataFrame()
        else:
            history_df = all_daily.loc[all_daily["trade_date"].isin(prior_dates)].copy()
        rows.append(
            calc_enhanced_raw_indicators_for_date(
                token,
                trade_date,
                previous_date,
                prior_dates,
                daily_map,
                stock_basic,
                index_history,
                industry_history,
                history_df,
            )
        )
        completed_targets += 1
        if progress_callback is not None:
            progress_callback(completed_targets, total_targets, f"正在计算 {trade_date} 新增指标")
    raw_df = pd.DataFrame(rows)
    merged = _attach_original_afgi(
        token,
        raw_df,
        source_start,
        end_date,
        force_update=False,
        original_df=original,
    )
    return _filter_date_range(_finalize_enhanced_scores(merged), start_date, end_date)


def update_enhanced_afgi_cache(
    token: str,
    start_date: str,
    end_date: str,
    force_update: bool = False,
    cache_path: Path = ENHANCED_CACHE_PATH,
    max_update_days: Optional[int] = DEFAULT_MAX_UPDATE_DAYS,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Incrementally update enhanced AFGI cache and return the requested date range."""
    start_date = _normalize_date(start_date)
    end_date = _normalize_date(end_date)
    cache_df = load_enhanced_cache(cache_path)
    if not cache_df.empty:
        versions = set(cache_df.get("enhanced_schema_version", pd.Series(dtype=str)).dropna().astype(str))
        if versions != {ENHANCED_CACHE_SCHEMA_VERSION}:
            force_update = True

    local_target_dates = set(_local_stock_daily_dates(start_date, end_date))
    if not token and not local_target_dates:
        if cache_df.empty or force_update:
            original = load_original_afgi_cache()
            fallback = _neutral_enhanced_from_original(original)
            if not fallback.empty:
                save_enhanced_cache(fallback, cache_path)
                cache_df = fallback
        return _filter_date_range(cache_df, start_date, end_date)

    try:
        target_dates = set(load_trade_dates(token, start_date, end_date)) if token else set()
        target_dates_loaded = bool(token)
    except Exception:
        target_dates = set()
        target_dates_loaded = False
    target_dates = target_dates.union(local_target_dates)
    all_cached_dates = set(cache_df.get("trade_date", pd.Series(dtype=str)).astype(str)) if not cache_df.empty else set()
    cached_dates = _effective_cached_dates(cache_df, token or ("local" if local_target_dates else ""))
    invalid_cached_dates = all_cached_dates - cached_dates
    unavailable_cached_dates = {
        trade_date
        for trade_date in invalid_cached_dates
        if start_date <= trade_date <= end_date and trade_date not in target_dates
    } if target_dates_loaded or local_target_dates else set()
    cache_changed = bool(unavailable_cached_dates)
    if cache_changed:
        cache_df = cache_df.loc[~cache_df["trade_date"].isin(unavailable_cached_dates)].copy()

    missing_dates = sorted(target_dates - cached_dates)
    if force_update:
        missing_dates = sorted(target_dates)
    if max_update_days is not None and max_update_days > 0 and len(missing_dates) > max_update_days:
        missing_dates = missing_dates[-max_update_days:]

    if missing_dates:
        build_start = missing_dates[0]
        if progress_callback is not None:
            progress_callback(0, len(missing_dates), f"准备补算 {len(missing_dates)} 个交易日")
        new_df = build_afgi_enhanced_timeseries(
            token,
            build_start,
            end_date,
            force_update=force_update,
            progress_callback=progress_callback,
        )
        new_dates = set(new_df.get("trade_date", pd.Series(dtype=str)).astype(str)) if not new_df.empty else set()
        unavailable_attempts = (set(missing_dates) - new_dates) & invalid_cached_dates
        if unavailable_attempts and not cache_df.empty:
            cache_df = cache_df.loc[~cache_df["trade_date"].isin(unavailable_attempts)].copy()
        combined = pd.concat([cache_df, new_df], ignore_index=True, sort=False) if not cache_df.empty else new_df
        if not combined.empty and "trade_date" in combined.columns:
            combined["trade_date"] = combined["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
            # Preserve concat order so newly calculated rows reliably replace old
            # fallback rows before the normal date sorting in finalization.
            combined = combined.drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
        combined = _finalize_enhanced_scores(combined)
        if combined.empty:
            cache_path.unlink(missing_ok=True)
        else:
            save_enhanced_cache(combined, cache_path)
        cache_df = combined
        if progress_callback is not None:
            progress_callback(len(missing_dates), len(missing_dates), "市场情绪缓存更新完成")
    else:
        if cache_changed:
            if cache_df.empty:
                cache_path.unlink(missing_ok=True)
            else:
                save_enhanced_cache(cache_df, cache_path)
        if progress_callback is not None:
            progress_callback(1, 1, "当前区间没有需要补算的交易日")

    return _filter_date_range(cache_df, start_date, end_date)


def latest_enhanced_indicator_table(latest: pd.Series) -> pd.DataFrame:
    """Return a display table for the latest ten enhanced raw indicators."""
    rows = []
    for raw_col in RAW_INDICATOR_COLUMNS:
        name, direction = RAW_INDICATOR_META[raw_col]
        rows.append(
            {
                "指标名称": name,
                "原始值": latest.get(raw_col, np.nan),
                "标准化得分": latest.get(ENHANCED_SCORE_COLUMNS[raw_col], 50.0),
                "指标方向": direction,
                "数据源": latest.get(f"{raw_col}_source", "--"),
                "数据状态": latest.get(f"{raw_col}_status", "missing"),
                "说明": latest.get(f"{raw_col}_message", ""),
            }
        )
    return pd.DataFrame(rows)

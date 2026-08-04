from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_sources import DATA_DIR, load_trade_dates
from .fear_greed_index import build_fear_greed_timeseries, score_raw_fear_greed_indicators


AFGI_CACHE_PATH = DATA_DIR / "market_fear_greed_cache.csv"
AFGI_CACHE_SCHEMA_VERSION = "afgi_v16_original_price_strength"


def load_cache(cache_path: Path = AFGI_CACHE_PATH) -> pd.DataFrame:
    """读取本地 AFGI 缓存。"""
    if not cache_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(cache_path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "trade_date" not in df.columns and "date" in df.columns:
        df["trade_date"] = df["date"].dt.strftime("%Y%m%d")
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    return df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)


def save_cache(df: pd.DataFrame, cache_path: Path = AFGI_CACHE_PATH) -> None:
    """保存 AFGI 缓存到 CSV。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["afgi_schema_version"] = AFGI_CACHE_SCHEMA_VERSION
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last")
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")


def _filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.loc[(out["date"] >= start_ts) & (out["date"] <= end_ts)].sort_values("date").reset_index(drop=True)


def _cached_dates_with_market_data(cache_df: pd.DataFrame) -> set[str]:
    """Return cached dates backed by actual all-stock daily turnover."""
    if cache_df is None or cache_df.empty or "trade_date" not in cache_df.columns:
        return set()
    if "total_amount" not in cache_df.columns:
        return set(cache_df["trade_date"].astype(str))
    valid = pd.to_numeric(cache_df["total_amount"], errors="coerce").gt(0)
    if "volume_source" in cache_df.columns:
        valid &= cache_df["volume_source"].fillna("").astype(str).str.contains("全A", regex=False)
    return set(cache_df.loc[valid, "trade_date"].astype(str))


def update_fear_greed_cache(token: str, start_date: str, end_date: str, force_update: bool = False, cache_path: Path = AFGI_CACHE_PATH) -> pd.DataFrame:
    """
    增量更新 AFGI 缓存。

    已有日期直接读取；缺失日期才预取历史并补算，最后去重保存。
    """
    start_date = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_date = pd.Timestamp(end_date).strftime("%Y%m%d")
    cache_df = load_cache(cache_path)
    cache_needs_rescore = False
    if not cache_df.empty:
        versions = set(cache_df.get("afgi_schema_version", pd.Series(dtype=str)).dropna().astype(str))
        cache_needs_rescore = versions != {AFGI_CACHE_SCHEMA_VERSION}
        if cache_needs_rescore:
            force_update = True
            if any(col.startswith("raw_") for col in cache_df.columns):
                rescored = score_raw_fear_greed_indicators(cache_df)
                for col in [c for c in cache_df.columns if c.endswith("_close") and c not in rescored.columns]:
                    rescored[col] = cache_df[col]
                save_cache(rescored, cache_path)
                cache_df = rescored
                force_update = False
    if not token:
        if force_update and not cache_df.empty and any(col.startswith("raw_") for col in cache_df.columns):
            rescored = score_raw_fear_greed_indicators(cache_df)
            for col in [c for c in cache_df.columns if c.endswith("_close") and c not in rescored.columns]:
                rescored[col] = cache_df[col]
            save_cache(rescored, cache_path)
            cache_df = rescored
        return _filter_date_range(cache_df, start_date, end_date)

    target_dates = set(load_trade_dates(token, start_date, end_date))
    all_cached_dates = set(cache_df.get("trade_date", pd.Series(dtype=str)).astype(str)) if not cache_df.empty else set()
    cached_dates = _cached_dates_with_market_data(cache_df)
    invalid_cached_dates = all_cached_dates - cached_dates
    unavailable_cached_dates = {
        trade_date
        for trade_date in invalid_cached_dates
        if start_date <= trade_date <= end_date and trade_date not in target_dates
    }
    cache_changed = bool(unavailable_cached_dates)
    if cache_changed:
        cache_df = cache_df.loc[~cache_df["trade_date"].isin(unavailable_cached_dates)].copy()

    missing_dates = sorted(target_dates - cached_dates)
    if force_update or cache_needs_rescore:
        missing_dates = sorted(target_dates)
    if missing_dates:
        source_start = (pd.to_datetime(missing_dates[0]) - pd.Timedelta(days=430)).strftime("%Y%m%d")
        new_df = build_fear_greed_timeseries(token, source_start, end_date)
        new_dates = set(new_df.get("trade_date", pd.Series(dtype=str)).astype(str)) if not new_df.empty else set()
        unavailable_attempts = (set(missing_dates) - new_dates) & invalid_cached_dates
        if unavailable_attempts and not cache_df.empty:
            cache_df = cache_df.loc[~cache_df["trade_date"].isin(unavailable_attempts)].copy()
        combined = pd.concat([cache_df, new_df], ignore_index=True, sort=False) if not cache_df.empty else new_df
        combined = combined.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date").reset_index(drop=True)
        # 重新评分可以保证历史窗口在合并旧缓存后连续；旧缓存中已有 score 时也不会破坏原始字段。
        rescored = score_raw_fear_greed_indicators(combined) if any(col.startswith("raw_") for col in combined.columns) else combined
        for col in [c for c in combined.columns if c.endswith("_close") and c not in rescored.columns]:
            rescored[col] = combined[col]
        save_cache(rescored, cache_path)
        cache_df = rescored
    elif cache_changed:
        save_cache(cache_df, cache_path)
    return _filter_date_range(cache_df, start_date, end_date)


def get_cached_or_update(token: str, start_date: str, end_date: str, force_update: bool = False) -> pd.DataFrame:
    """页面统一调用入口。"""
    return update_fear_greed_cache(token, start_date, end_date, force_update=force_update)

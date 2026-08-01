from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


IndicatorResult = Dict[str, object]


def _result(value=np.nan, status: str = "missing", message: str = "", source: str = "") -> IndicatorResult:
    """Return the common result object used by enhanced AFGI raw indicators."""
    try:
        numeric = float(value)
    except Exception:
        numeric = np.nan
    if not np.isfinite(numeric):
        numeric = np.nan
    return {"value": numeric, "status": status, "message": message, "source": source}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _normalize_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    elif "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), errors="coerce")
    else:
        out["date"] = pd.NaT
    if "trade_date" not in out.columns:
        out["trade_date"] = out["date"].dt.strftime("%Y%m%d")
    out["trade_date"] = out["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    return out.dropna(subset=["date"]).reset_index(drop=True)


def _valid_stock_daily(df: pd.DataFrame, stock_basic_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Filter tradable A-share daily rows and optionally remove ST stocks."""
    out = _normalize_trade_date(df)
    if out.empty or "ts_code" not in out.columns:
        return pd.DataFrame()
    out["ts_code"] = out["ts_code"].astype(str)
    for col in ["close", "pre_close", "pct_chg", "high", "low", "vol", "amount"]:
        if col in out.columns:
            out[col] = _numeric(out[col])
    if "close" not in out.columns:
        return pd.DataFrame()
    valid = out["close"].notna() & (out["close"] > 0)
    if "pre_close" in out.columns:
        valid &= out["pre_close"].notna() & (out["pre_close"] > 0)
    if "vol" in out.columns:
        valid &= out["vol"].fillna(0) > 0
    out = out.loc[valid].copy()
    if stock_basic_df is not None and not stock_basic_df.empty and {"ts_code", "name"}.issubset(stock_basic_df.columns):
        basic = stock_basic_df[["ts_code", "name"]].copy()
        basic["ts_code"] = basic["ts_code"].astype(str)
        basic["name"] = basic["name"].fillna("").astype(str)
        out = out.merge(basic, on="ts_code", how="left")
        st_mask = out["name"].fillna("").str.upper().str.contains("ST", regex=False)
        out = out.loc[~st_mask].copy()
    return out.reset_index(drop=True)


def _limit_threshold(ts_code: pd.Series) -> pd.Series:
    """Approximate daily limit thresholds by board for A-shares."""
    code = ts_code.fillna("").astype(str)
    threshold = pd.Series(0.095, index=code.index, dtype="float64")
    threshold.loc[code.str.startswith(("300", "301", "688", "689"))] = 0.195
    threshold.loc[code.str.contains(".BJ", regex=False) | code.str.startswith(("8", "4"))] = 0.295
    return threshold


def _daily_return_decimal(df: pd.DataFrame) -> pd.Series:
    if "pct_chg" in df.columns:
        return _numeric(df["pct_chg"]) / 100.0
    if {"close", "pre_close"}.issubset(df.columns):
        return _numeric(df["close"]) / _numeric(df["pre_close"]) - 1.0
    return pd.Series(np.nan, index=df.index)


def _is_limit_up(df: pd.DataFrame) -> pd.Series:
    ret = _daily_return_decimal(df)
    return ret >= _limit_threshold(df.get("ts_code", pd.Series("", index=df.index)))


def _is_limit_down(df: pd.DataFrame) -> pd.Series:
    ret = _daily_return_decimal(df)
    return ret <= -_limit_threshold(df.get("ts_code", pd.Series("", index=df.index)))


def _history_with_current(history_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if history_df is not None and not history_df.empty:
        frames.append(_normalize_trade_date(history_df))
    if current_df is not None and not current_df.empty:
        frames.append(_normalize_trade_date(current_df))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    if "close" in out.columns:
        out["close"] = _numeric(out["close"])
    return out.sort_values(["ts_code", "date"]).reset_index(drop=True)


def calc_up_ratio(current_df: pd.DataFrame, stock_basic_df: Optional[pd.DataFrame] = None) -> IndicatorResult:
    """上涨家数占比：上涨股票数量 / 有效交易股票数量。"""
    valid = _valid_stock_daily(current_df, stock_basic_df)
    if valid.empty:
        return _result(status="missing", message="无有效股票日行情", source="Tushare daily")
    ret = _daily_return_decimal(valid)
    value = float((ret > 0).sum() / max(len(valid), 1))
    return _result(value, "ok", f"有效股票 {len(valid)} 只", "Tushare daily")


def calc_ma20_ratio(history_df: pd.DataFrame, current_df: pd.DataFrame, stock_basic_df: Optional[pd.DataFrame] = None) -> IndicatorResult:
    """站上20日均线股票占比：收盘价高于 MA20 的有效股票占比。"""
    return _calc_ma_ratio(history_df, current_df, 20, stock_basic_df, "站上20日均线股票占比")


def calc_ma60_ratio(history_df: pd.DataFrame, current_df: pd.DataFrame, stock_basic_df: Optional[pd.DataFrame] = None) -> IndicatorResult:
    """站上60日均线股票占比：收盘价高于 MA60 的有效股票占比。"""
    return _calc_ma_ratio(history_df, current_df, 60, stock_basic_df, "站上60日均线股票占比")


def _calc_ma_ratio(
    history_df: pd.DataFrame,
    current_df: pd.DataFrame,
    window: int,
    stock_basic_df: Optional[pd.DataFrame],
    label: str,
) -> IndicatorResult:
    valid = _valid_stock_daily(current_df, stock_basic_df)
    if valid.empty:
        return _result(status="missing", message="无有效股票日行情", source="Tushare daily")
    hist = _history_with_current(history_df, valid)
    if hist.empty:
        return _result(status="missing", message="历史行情不足", source="Tushare daily")
    current_codes = set(valid["ts_code"].astype(str))
    rows = []
    for code, group in hist.loc[hist["ts_code"].astype(str).isin(current_codes)].groupby("ts_code"):
        tail = group.sort_values("date").tail(window)
        if len(tail) < window or tail["close"].isna().any():
            continue
        rows.append((code, float(tail["close"].iloc[-1]), float(tail["close"].mean())))
    if not rows:
        return _result(status="missing", message=f"{label}历史样本不足", source="Tushare daily")
    calc_df = pd.DataFrame(rows, columns=["ts_code", "close", "ma"])
    value = float((calc_df["close"] > calc_df["ma"]).sum() / max(len(calc_df), 1))
    return _result(value, "ok", f"可计算股票 {len(calc_df)} 只", "Tushare daily")


def calc_limit_up_down_ratio(current_df: pd.DataFrame, stock_basic_df: Optional[pd.DataFrame] = None) -> IndicatorResult:
    """涨停跌停比：涨停家数 / max(跌停家数, 1)。"""
    valid = _valid_stock_daily(current_df, stock_basic_df)
    if valid.empty:
        return _result(status="missing", message="无有效股票日行情", source="Tushare daily")
    up_count = int(_is_limit_up(valid).sum())
    down_count = int(_is_limit_down(valid).sum())
    value = float(up_count / max(down_count, 1))
    return _result(value, "ok", f"涨停 {up_count} 只，跌停 {down_count} 只", "Tushare daily approximate limit")


def calc_open_board_rate(current_df: pd.DataFrame, stock_basic_df: Optional[pd.DataFrame] = None) -> IndicatorResult:
    """炸板率：炸板股票数量 / 盘中触及涨停股票数量。"""
    valid = _valid_stock_daily(current_df, stock_basic_df)
    if valid.empty or not {"high", "pre_close", "close"}.issubset(valid.columns):
        return _result(status="missing", message="缺少 high/pre_close/close，无法识别炸板", source="Tushare daily")
    threshold = _limit_threshold(valid["ts_code"])
    touched = (_numeric(valid["high"]) / _numeric(valid["pre_close"]) - 1.0) >= threshold
    final_limit = _is_limit_up(valid)
    opened = touched & ~final_limit
    touched_count = int(touched.sum())
    if touched_count <= 0:
        return _result(0.0, "fallback", "当日无盘中触及涨停股票，炸板率按 0 处理", "Tushare daily approximate limit")
    value = float(opened.sum() / touched_count)
    return _result(value, "ok", f"触及涨停 {touched_count} 只，炸板 {int(opened.sum())} 只", "Tushare daily approximate limit")


def calc_yesterday_limit_up_return(
    yesterday_df: pd.DataFrame,
    current_df: pd.DataFrame,
    stock_basic_df: Optional[pd.DataFrame] = None,
) -> IndicatorResult:
    """昨日涨停今日平均收益：昨日涨停股票在今日的平均涨跌幅。"""
    prev_valid = _valid_stock_daily(yesterday_df, stock_basic_df)
    curr_valid = _valid_stock_daily(current_df, stock_basic_df)
    if prev_valid.empty or curr_valid.empty:
        return _result(status="missing", message="昨日或今日有效行情为空", source="Tushare daily")
    limit_codes = set(prev_valid.loc[_is_limit_up(prev_valid), "ts_code"].astype(str))
    if not limit_codes:
        return _result(np.nan, "fallback", "昨日无涨停股票，分数使用中性值", "Tushare daily approximate limit")
    selected = curr_valid.loc[curr_valid["ts_code"].astype(str).isin(limit_codes)].copy()
    if selected.empty:
        return _result(status="missing", message="今日无法匹配昨日涨停股票", source="Tushare daily")
    ret = _daily_return_decimal(selected).dropna()
    if ret.empty:
        return _result(status="missing", message="昨日涨停股票今日收益缺失", source="Tushare daily")
    return _result(float(ret.mean()), "ok", f"昨日涨停样本 {len(ret)} 只", "Tushare daily approximate limit")


def calc_up_5d_ratio(history_df: pd.DataFrame, current_df: pd.DataFrame, stock_basic_df: Optional[pd.DataFrame] = None) -> IndicatorResult:
    """近5日上涨股票占比：过去5个交易日收益率大于0的有效股票占比。"""
    valid = _valid_stock_daily(current_df, stock_basic_df)
    if valid.empty:
        return _result(status="missing", message="无有效股票日行情", source="Tushare daily")
    hist = _history_with_current(history_df, valid)
    rows = []
    current_codes = set(valid["ts_code"].astype(str))
    for code, group in hist.loc[hist["ts_code"].astype(str).isin(current_codes)].groupby("ts_code"):
        tail = group.sort_values("date").tail(6)
        if len(tail) < 6 or tail["close"].isna().any():
            continue
        rows.append(float(tail["close"].iloc[-1] / tail["close"].iloc[0] - 1.0))
    if not rows:
        return _result(status="missing", message="近5日收益历史样本不足", source="Tushare daily")
    ret = pd.Series(rows)
    return _result(float((ret > 0).sum() / len(ret)), "ok", f"可计算股票 {len(ret)} 只", "Tushare daily")


def calc_industry_up_ratio(industry_df: pd.DataFrame) -> IndicatorResult:
    """行业上涨占比：上涨行业数量 / 全部有效行业数量。"""
    if industry_df is None or industry_df.empty:
        return _result(status="missing", message="行业板块行情为空", source="AkShare/东方财富行业板块")
    out = industry_df.copy()
    pct_col = next((col for col in ["pct_chg", "涨跌幅", "change_pct", "涨跌幅%"] if col in out.columns), None)
    if pct_col is None:
        numeric_cols = [col for col in out.columns if _numeric(out[col]).notna().sum() > 0]
        pct_col = numeric_cols[0] if numeric_cols else None
    if pct_col is None:
        return _result(status="missing", message="行业板块行情缺少涨跌幅字段", source="AkShare/东方财富行业板块")
    pct = _numeric(out[pct_col]).dropna()
    if pct.empty:
        return _result(status="missing", message="行业涨跌幅为空", source="AkShare/东方财富行业板块")
    return _result(float((pct > 0).sum() / len(pct)), "ok", f"有效行业 {len(pct)} 个", "AkShare/东方财富行业板块")


def calc_stock_industry_up_ratio(
    current_df: pd.DataFrame,
    stock_basic_df: Optional[pd.DataFrame],
) -> IndicatorResult:
    """Fallback industry breadth from current stock returns grouped by Tushare industry."""
    if (
        current_df is None
        or current_df.empty
        or stock_basic_df is None
        or stock_basic_df.empty
        or not {"ts_code", "industry"}.issubset(stock_basic_df.columns)
    ):
        return _result(
            status="missing",
            message="缺少股票行业分类",
            source="Tushare stock_basic + daily",
        )
    daily = _valid_stock_daily(current_df, stock_basic_df)
    if daily.empty:
        return _result(
            status="missing",
            message="当日股票行情为空",
            source="Tushare stock_basic + daily",
        )
    basic = stock_basic_df[["ts_code", "industry"]].copy()
    basic["ts_code"] = basic["ts_code"].astype(str)
    basic["industry"] = basic["industry"].fillna("").astype(str).str.strip()
    daily = daily.merge(basic.drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")
    daily["industry"] = daily["industry"].fillna("").astype(str).str.strip()
    daily["stock_return"] = _daily_return_decimal(daily)
    daily = daily.loc[daily["industry"].ne("") & daily["stock_return"].notna()].copy()
    if daily.empty:
        return _result(
            status="missing",
            message="股票行业分类与当日行情无法匹配",
            source="Tushare stock_basic + daily",
        )
    industry_return = daily.groupby("industry")["stock_return"].mean().dropna()
    if industry_return.empty:
        return _result(
            status="missing",
            message="行业收益率为空",
            source="Tushare stock_basic + daily",
        )
    return _result(
        float((industry_return > 0).sum() / len(industry_return)),
        "ok",
        f"按成分股等权计算，有效行业 {len(industry_return)} 个",
        "Tushare stock_basic + daily",
    )


def calc_small_large_relative_strength(index_df: pd.DataFrame, window: int = 5) -> IndicatorResult:
    """小盘/大盘相对强弱：中证1000 N日收益率 - 沪深300 N日收益率。"""
    if index_df is None or index_df.empty:
        return _result(status="missing", message="指数行情为空", source="Tushare index_daily / AFGI cache")
    out = _normalize_trade_date(index_df).sort_values("date")
    required = {"zz1000_close", "hs300_close"}
    if not required.issubset(out.columns):
        return _result(status="missing", message="缺少中证1000或沪深300收盘价", source="Tushare index_daily / AFGI cache")
    zz = _numeric(out["zz1000_close"])
    hs = _numeric(out["hs300_close"])
    valid = pd.DataFrame({"zz": zz, "hs": hs}).dropna()
    if len(valid) <= window:
        return _result(status="missing", message=f"指数历史不足 {window} 日", source="Tushare index_daily / AFGI cache")
    small_ret = valid["zz"].iloc[-1] / valid["zz"].iloc[-window - 1] - 1.0
    large_ret = valid["hs"].iloc[-1] / valid["hs"].iloc[-window - 1] - 1.0
    return _result(float(small_ret - large_ret), "ok", f"中证1000 {window}日收益 - 沪深300 {window}日收益", "Tushare index_daily / AFGI cache")


def calc_new_low_60d_ratio(history_df: pd.DataFrame, current_df: pd.DataFrame, stock_basic_df: Optional[pd.DataFrame] = None) -> IndicatorResult:
    """创60日新低股票占比：今日收盘价小于等于过去60日最低收盘价的股票占比。"""
    valid = _valid_stock_daily(current_df, stock_basic_df)
    if valid.empty:
        return _result(status="missing", message="无有效股票日行情", source="Tushare daily")
    hist = _history_with_current(history_df, valid)
    rows = []
    current_codes = set(valid["ts_code"].astype(str))
    for code, group in hist.loc[hist["ts_code"].astype(str).isin(current_codes)].groupby("ts_code"):
        tail = group.sort_values("date").tail(60)
        if len(tail) < 60 or tail["close"].isna().any():
            continue
        close = float(tail["close"].iloc[-1])
        low = float(tail["close"].min())
        rows.append(close <= low)
    if not rows:
        return _result(status="missing", message="60日历史样本不足", source="Tushare daily")
    flags = pd.Series(rows)
    return _result(float(flags.sum() / len(flags)), "ok", f"可计算股票 {len(flags)} 只", "Tushare daily")

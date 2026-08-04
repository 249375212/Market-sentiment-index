from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from .tushare_client import init_pro

try:
    import akshare as ak
except Exception:
    ak = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "market_fear_greed"

INDEX_CODES = {
    "hs300": "000300.SH",
    "sh": "000001.SH",
    "zz1000": "000852.SH",
    "cyb": "399006.SZ",
}


def _empty(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _normalize_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = out["trade_date"].astype(str)
        out["date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["trade_date"] = out["date"].dt.strftime("%Y%m%d")
    out = out.dropna(subset=["date"])
    return out


def _safe_tushare_call(token: str, api_name: str, columns: List[str], **kwargs) -> pd.DataFrame:
    """安全调用 Tushare；失败时返回空表，由指标层中性化处理。"""
    if not token:
        return _empty(columns)
    try:
        pro = init_pro(token)
        api = getattr(pro, api_name)
        df = api(**kwargs)
    except Exception as exc:
        warnings.warn(f"Tushare {api_name} 获取失败：{exc}", RuntimeWarning)
        return _empty(columns)
    if df is None or df.empty:
        return _empty(columns)
    return df


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_trade_calendar(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    columns = ["cal_date", "is_open"]
    df = _safe_tushare_call(token, "trade_cal", columns, exchange="", start_date=start_date, end_date=end_date)
    if df.empty or "cal_date" not in df.columns:
        dates = pd.bdate_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
        return pd.DataFrame({"trade_date": dates.strftime("%Y%m%d"), "date": dates})
    df = df.loc[pd.to_numeric(df.get("is_open", 1), errors="coerce").fillna(0).astype(int) == 1].copy()
    df["trade_date"] = df["cal_date"].astype(str)
    df["date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")[["trade_date", "date"]].reset_index(drop=True)


def load_trade_dates(token: str, start_date: str, end_date: str) -> List[str]:
    return load_trade_calendar(token, start_date, end_date)["trade_date"].dropna().astype(str).tolist()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_stock_basic_safe(token: str) -> pd.DataFrame:
    columns = ["ts_code", "symbol", "name", "market", "industry", "list_date"]
    df = _safe_tushare_call(
        token,
        "stock_basic",
        columns,
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,market,industry,list_date",
    )
    if df.empty:
        return _empty(columns)
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_all_stock_daily(token: str, trade_date: str) -> pd.DataFrame:
    columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]
    df = _safe_tushare_call(token, "daily", columns, trade_date=trade_date)
    if df.empty:
        return _empty(columns + ["date"])
    df = _normalize_trade_date(df)
    for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_index_daily(token: str, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]
    df = _safe_tushare_call(token, "index_daily", columns, ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty:
        return _empty(columns + ["date"])
    df = _normalize_trade_date(df)
    for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_fund_daily(token: str, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]
    df = _safe_tushare_call(token, "fund_daily", columns, ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty:
        return _empty(columns + ["date"])
    df = _normalize_trade_date(df)
    for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df.sort_values("date").reset_index(drop=True)


def _normalize_ivix_values(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize an iVIX-like series to decimal volatility, e.g. 18.5 -> 0.185."""
    if df is None or df.empty:
        return _empty(["trade_date", "date", "raw_market_volatility", "volatility_source"])
    out = df.copy()
    date_col = next(
        (
            col
            for col in out.columns
            if str(col).lower() in {"date", "trade_date", "cal_date"}
            or any(key in str(col) for key in ["日期", "时间"])
        ),
        None,
    )
    if date_col is None:
        return _empty(["trade_date", "date", "raw_market_volatility", "volatility_source"])
    value_candidates = [
        "ivix",
        "iVIX",
        "IVIX",
        "china_vix",
        "vix",
        "VIX",
        "close",
        "收盘",
        "当前值",
        "最新价",
        "指数",
        "波指",
    ]
    value_col = next((col for col in value_candidates if col in out.columns), None)
    if value_col is None:
        numeric_cols = []
        for col in out.columns:
            if col == date_col:
                continue
            numeric = pd.to_numeric(out[col], errors="coerce")
            if numeric.notna().sum() > 0:
                numeric_cols.append(col)
        value_col = numeric_cols[0] if numeric_cols else None
    if value_col is None:
        return _empty(["trade_date", "date", "raw_market_volatility", "volatility_source"])
    out["date"] = pd.to_datetime(out[date_col], errors="coerce")
    out["trade_date"] = out["date"].dt.strftime("%Y%m%d")
    value = pd.to_numeric(out[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    median_value = value.dropna().abs().median()
    # Official iVIX/China VIX is commonly quoted as a percent-like index.
    if pd.notna(median_value) and median_value > 2:
        value = value / 100.0
    out["raw_market_volatility"] = value
    out["volatility_source"] = source
    out = out.dropna(subset=["date", "raw_market_volatility"])
    return out[["trade_date", "date", "raw_market_volatility", "volatility_source"]].sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_local_ivix_series(start_date: str, end_date: str) -> pd.DataFrame:
    """Read a local official/history iVIX CSV when the user provides one.

    Supported file names under data/market_fear_greed:
    ivix.csv, china_ivix.csv, ivix_history.csv.
    """
    candidates = [DATA_DIR / "ivix.csv", DATA_DIR / "china_ivix.csv", DATA_DIR / "ivix_history.csv"]
    frames = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                warnings.warn(f"Local iVIX CSV read failed: {path} {exc}", RuntimeWarning)
                continue
        normalized = _normalize_ivix_values(df, "本地原中国波指 iVIX")
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return _empty(["trade_date", "date", "raw_market_volatility", "volatility_source"])
    out = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(subset=["trade_date"], keep="last")
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    out = out.loc[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
    return out.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_ivix_akshare(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch an iVIX/VIX-like series from AkShare when available.

    Note: the official SSE China iVIX has been discontinued publicly. This source
    is treated as an external compatible volatility index and can be replaced by
    a local official iVIX CSV without changing calculation code.
    """
    if ak is None:
        return _empty(["trade_date", "date", "raw_market_volatility", "volatility_source"])
    try:
        df = ak.index_vix(start_date=start_date, end_date=end_date)
    except Exception as exc:
        warnings.warn(f"AkShare index_vix 获取失败：{exc}", RuntimeWarning)
        return _empty(["trade_date", "date", "raw_market_volatility", "volatility_source"])
    normalized = _normalize_ivix_values(df, "AkShare iVIX/VIX兼容波动率指数")
    if normalized.empty:
        return normalized
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    return normalized.loc[(normalized["date"] >= start_ts) & (normalized["date"] <= end_ts)].reset_index(drop=True)


def load_ivix_series(start_date: str, end_date: str) -> pd.DataFrame:
    """Load the preferred market volatility series: official/local iVIX first."""
    local_df = load_local_ivix_series(start_date, end_date)
    if not local_df.empty:
        return local_df
    return load_ivix_akshare(start_date, end_date)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_50etf_option_basic(token: str) -> pd.DataFrame:
    """读取上交所 50ETF 期权基础信息，用于反推隐含波动率。"""
    columns = ["ts_code", "name", "call_put", "exercise_price", "maturity_date", "s_date", "delist_date"]
    df = _safe_tushare_call(
        token,
        "opt_basic",
        columns,
        exchange="SSE",
        fields="ts_code,name,call_put,exercise_price,maturity_date,s_date,delist_date",
    )
    if df.empty:
        return _empty(columns)
    for col in ["ts_code", "name", "call_put", "maturity_date", "s_date", "delist_date"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df["exercise_price"] = pd.to_numeric(df.get("exercise_price", np.nan), errors="coerce")
    if "name" in df.columns:
        name = df["name"].fillna("").astype(str).str.upper()
        filtered = df.loc[name.str.contains("50ETF", regex=False)].copy()
        if not filtered.empty:
            df = filtered
    return df.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_50etf_option_daily(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """读取上交所期权日行情；失败时返回空表并降级到历史波动率。"""
    columns = ["ts_code", "trade_date", "close", "settle", "vol", "amount", "oi"]
    frames = []
    df = _safe_tushare_call(
        token,
        "opt_daily",
        columns,
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,close,settle,vol,amount,oi",
    )
    if not df.empty:
        frames.append(df)
    if not frames:
        # 部分账户/版本只支持按交易日查期权行情。
        for trade_date in load_trade_dates(token, start_date, end_date):
            df = _safe_tushare_call(
                token,
                "opt_daily",
                columns,
                exchange="SSE",
                trade_date=trade_date,
                fields="ts_code,trade_date,close,settle,vol,amount,oi",
            )
            if not df.empty:
                frames.append(df)
    if not frames:
        return _empty(columns + ["date"])
    out = _normalize_trade_date(pd.concat(frames, ignore_index=True, sort=False))
    for col in ["close", "settle", "vol", "amount", "oi"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["date", "ts_code"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_northbound_tushare(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    columns = ["trade_date", "north_money"]
    df = _safe_tushare_call(token, "moneyflow_hsgt", columns, start_date=start_date, end_date=end_date)
    if df.empty:
        return _empty(columns + ["date"])
    df = _normalize_trade_date(df)
    candidate_cols = ["north_money", "north_net_amount", "ggt_ss", "ggt_sz"]
    if "north_money" not in df.columns:
        usable = [col for col in candidate_cols if col in df.columns]
        df["north_money"] = df[usable].sum(axis=1, min_count=1) if usable else np.nan
    df["north_money"] = pd.to_numeric(df["north_money"], errors="coerce")
    return df[["trade_date", "date", "north_money"]].sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_northbound_akshare(start_date: str, end_date: str) -> pd.DataFrame:
    """AkShare 北向资金兜底；接口字段会随版本变动，因此做宽松字段识别。"""
    if ak is None:
        return _empty(["trade_date", "date", "north_money"])
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
    except Exception as exc:
        warnings.warn(f"AkShare 北向资金获取失败：{exc}", RuntimeWarning)
        return _empty(["trade_date", "date", "north_money"])
    if df is None or df.empty:
        return _empty(["trade_date", "date", "north_money"])
    out = df.copy()
    date_col = next((col for col in out.columns if "日期" in str(col) or str(col).lower() in {"date", "trade_date"}), None)
    money_col = next((col for col in out.columns if "净买入" in str(col) or "资金" in str(col) or "north" in str(col).lower()), None)
    if date_col is None or money_col is None:
        return _empty(["trade_date", "date", "north_money"])
    out["date"] = pd.to_datetime(out[date_col], errors="coerce")
    out["trade_date"] = out["date"].dt.strftime("%Y%m%d")
    out["north_money"] = pd.to_numeric(out[money_col], errors="coerce")
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    out = out.loc[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
    return out[["trade_date", "date", "north_money"]].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_northbound_series(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = load_northbound_tushare(token, start_date, end_date)
    if df.empty:
        df = load_northbound_akshare(start_date, end_date)
    return df


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_margin_tushare(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """读取融资融券市场汇总数据，优先使用 Tushare margin 接口。"""
    columns = ["trade_date", "exchange_id", "rzmre"]
    frames = []
    all_market = _safe_tushare_call(token, "margin", columns, start_date=start_date, end_date=end_date)
    if not all_market.empty:
        frames = [all_market]
    else:
        for exchange_id in ["SSE", "SZSE"]:
            df = _safe_tushare_call(token, "margin", columns, exchange_id=exchange_id, start_date=start_date, end_date=end_date)
            if not df.empty:
                frames.append(df)
    if not frames:
        # 某些 Tushare 环境只支持按交易日查询，作为兜底逐日请求。
        daily_frames = []
        for trade_date in load_trade_dates(token, start_date, end_date):
            df = _safe_tushare_call(token, "margin", columns, trade_date=trade_date)
            if not df.empty:
                daily_frames.append(df)
        frames = daily_frames
    if not frames:
        return _empty(columns + ["date", "margin_buy_amount"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = _normalize_trade_date(out)
    if "rzmre" not in out.columns:
        return _empty(columns + ["date", "margin_buy_amount"])
    out["margin_buy_amount"] = pd.to_numeric(out["rzmre"], errors="coerce")
    if "exchange_id" not in out.columns:
        out["exchange_id"] = ""
    out = (
        out.groupby(["trade_date", "date"], as_index=False)
        .agg(
            margin_buy_amount=("margin_buy_amount", "sum"),
            margin_exchange_count=("exchange_id", lambda s: int(s.fillna("").astype(str).replace("", np.nan).nunique())),
            margin_record_count=("margin_buy_amount", "count"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    out["margin_source"] = "Tushare融资买入额"
    return out


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_margin_akshare(start_date: str, end_date: str) -> pd.DataFrame:
    """AkShare 融资余额/融资买入额兜底，字段随版本变化，采用宽松识别。"""
    if ak is None:
        return _empty(["trade_date", "date", "margin_buy_amount", "margin_source"])
    candidates = [
        ("stock_margin_sse", {}),
        ("stock_margin_szse", {}),
        ("stock_margin_detail_sse", {}),
        ("stock_margin_detail_szse", {}),
    ]
    frames = []
    for func_name, kwargs in candidates:
        func = getattr(ak, func_name, None)
        if func is None:
            continue
        try:
            df = func(**kwargs)
        except Exception as exc:
            warnings.warn(f"AkShare {func_name} 获取失败：{exc}", RuntimeWarning)
            continue
        if df is None or df.empty:
            continue
        out = df.copy()
        date_col = next((col for col in out.columns if "日期" in str(col) or str(col).lower() in {"date", "trade_date"}), None)
        amount_col = next((col for col in out.columns if "融资买入" in str(col) or "rzmre" in str(col).lower()), None)
        if date_col is None or amount_col is None:
            continue
        out["date"] = pd.to_datetime(out[date_col], errors="coerce")
        out["trade_date"] = out["date"].dt.strftime("%Y%m%d")
        out["margin_buy_amount"] = pd.to_numeric(out[amount_col], errors="coerce")
        out["margin_source"] = f"AkShare {func_name}"
        frames.append(out[["trade_date", "date", "margin_buy_amount", "margin_source"]])
    if not frames:
        return _empty(["trade_date", "date", "margin_buy_amount", "margin_source"])
    result = pd.concat(frames, ignore_index=True, sort=False)
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    result = result.loc[(result["date"] >= start_ts) & (result["date"] <= end_ts)]
    return (
        result.groupby(["trade_date", "date"], as_index=False)
        .agg(
            margin_buy_amount=("margin_buy_amount", "sum"),
            margin_source=("margin_source", lambda s: " + ".join(sorted(set(s.dropna().astype(str))))),
            margin_record_count=("margin_buy_amount", "count"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_margin_series(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = load_margin_tushare(token, start_date, end_date)
    if df.empty:
        df = load_margin_akshare(start_date, end_date)
    return df


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_futures_if_akshare(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch IF main continuous daily prices from AkShare/Sina as a broad fallback."""
    columns = ["ts_code", "trade_date", "date", "close", "vol", "oi"]
    if ak is None:
        return _empty(columns)
    try:
        df = ak.futures_main_sina(symbol="IF0", start_date=start_date, end_date=end_date)
    except Exception as exc:
        warnings.warn(f"AkShare IF0 主力连续获取失败：{exc}", RuntimeWarning)
        return _empty(columns)
    if df is None or df.empty:
        return _empty(columns)
    out = df.copy()
    date_col = next((col for col in out.columns if str(col).lower() in {"date", "trade_date"} or "日期" in str(col)), None)
    close_col = next((col for col in out.columns if str(col).lower() in {"close", "收盘价"} or "收盘" in str(col)), None)
    if date_col is None or close_col is None:
        return _empty(columns)
    out["date"] = pd.to_datetime(out[date_col], errors="coerce")
    out["trade_date"] = out["date"].dt.strftime("%Y%m%d")
    out["close"] = pd.to_numeric(out[close_col], errors="coerce")
    out["ts_code"] = "IF0.AK"
    out["vol"] = pd.to_numeric(out.get("volume", out.get("成交量", np.nan)), errors="coerce")
    out["oi"] = pd.to_numeric(out.get("hold", out.get("持仓量", np.nan)), errors="coerce")
    return out[columns].dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_futures_if_contracts_akshare(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch monthly IF contracts from AkShare/Sina for the second-month basis metric."""
    columns = ["ts_code", "trade_date", "date", "close", "vol", "oi"]
    if ak is None:
        return _empty(columns)
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    # 预取结束日后两个月，确保每个交易日都有次月合约候选。
    months = pd.period_range(start_ts.to_period("M"), (end_ts + pd.DateOffset(months=2)).to_period("M"), freq="M")
    frames = []
    for month in months:
        symbol = f"IF{str(month.year)[-2:]}{month.month:02d}"
        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
        except Exception as exc:
            warnings.warn(f"AkShare {symbol} 日线获取失败：{exc}", RuntimeWarning)
            continue
        if df is None or df.empty:
            continue
        out = df.copy()
        date_col = next((col for col in out.columns if str(col).lower() in {"date", "trade_date"} or "日期" in str(col)), None)
        close_col = next((col for col in out.columns if str(col).lower() in {"close", "收盘价"} or "收盘" in str(col)), None)
        if date_col is None or close_col is None:
            continue
        out["date"] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.loc[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
        if out.empty:
            continue
        out["trade_date"] = out["date"].dt.strftime("%Y%m%d")
        out["close"] = pd.to_numeric(out[close_col], errors="coerce")
        out["ts_code"] = f"{symbol}.AK"
        out["vol"] = pd.to_numeric(out.get("volume", out.get("成交量", np.nan)), errors="coerce")
        out["oi"] = pd.to_numeric(out.get("hold", out.get("持仓量", np.nan)), errors="coerce")
        frames.append(out[columns])
    if not frames:
        return _empty(columns)
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .dropna(subset=["date", "close"])
        .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        .sort_values(["date", "ts_code"])
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_futures_basis_proxy(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取沪深300股指期货 IF 实际合约日行情。

    优先取中金所 IF 实际合约，指标层会选择近月合约并年化升贴水；若不可得再降级到主力连续近似。
    """
    columns = ["ts_code", "trade_date", "close", "vol", "oi"]
    frames = []
    all_contracts = _safe_tushare_call(
        token,
        "fut_daily",
        columns,
        exchange="CFFEX",
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,close,vol,oi",
    )
    if not all_contracts.empty:
        frames.append(all_contracts)
    # 连续合约无论实际合约是否可用都尝试补取，用于填补历史次月合约缺口。
    for ts_code in ["IF.CFX", "IF9999.CFE", "IF9999.CFX"]:
        df = _safe_tushare_call(token, "fut_daily", columns, ts_code=ts_code, start_date=start_date, end_date=end_date)
        if not df.empty:
            frames.append(df)
            break
    ak_contracts = load_futures_if_contracts_akshare(start_date, end_date)
    if not ak_contracts.empty:
        frames.append(ak_contracts)
    ak_if = load_futures_if_akshare(start_date, end_date)
    if not ak_if.empty:
        frames.append(ak_if)
    if not frames:
        return _empty(columns + ["date"])
    out = _normalize_trade_date(pd.concat(frames, ignore_index=True, sort=False))
    for col in ["close", "vol", "oi"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["ts_code"] = out["ts_code"].fillna("").astype(str).str.upper()
    if out["ts_code"].str.contains("IF", regex=False).any():
        out = out.loc[out["ts_code"].str.contains("IF", regex=False)].copy()
    return out[["ts_code", "trade_date", "date", "close", "vol", "oi"]].sort_values(["date", "ts_code"]).reset_index(drop=True)


def load_market_fear_greed_source_data(token: str, start_date: str, end_date: str) -> Dict[str, object]:
    """读取 AFGI 所需的原始数据，所有外部源失败都返回空表。"""
    calendar_dates = load_trade_dates(token, start_date, end_date)
    available_trade_dates = []
    daily_frames = []
    for trade_date in calendar_dates:
        daily = load_all_stock_daily(token, trade_date)
        if not daily.empty:
            available_trade_dates.append(trade_date)
            daily_frames.append(daily)
    index_frames = {key: load_index_daily(token, code, start_date, end_date) for key, code in INDEX_CODES.items()}
    return {
        "trade_dates": available_trade_dates,
        "stock_basic": load_stock_basic_safe(token),
        "daily": pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
        "index": index_frames,
        "ivix": load_ivix_series(start_date, end_date),
        "etf50": load_fund_daily(token, "510050.SH", start_date, end_date),
        "option_basic": load_50etf_option_basic(token),
        "option_daily": load_50etf_option_daily(token, start_date, end_date),
        "bond_etf": load_fund_daily(token, "511010.SH", start_date, end_date),
        "margin": load_margin_series(token, start_date, end_date),
        "futures_if": load_futures_basis_proxy(token, start_date, end_date),
    }

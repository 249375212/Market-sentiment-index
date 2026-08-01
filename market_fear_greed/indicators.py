from __future__ import annotations

import math
import re
from typing import Dict

import numpy as np
import pandas as pd


IF_CONTINUOUS_CODES = {"IF.CFX", "IF9999.CFE", "IF9999.CFX", "IF0.AK"}


def _quality(status: str, source: str, message: str = "") -> Dict[str, str]:
    return {"status": status, "source": source, "message": message}


def _clean_price_frame(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date", close_col])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[close_col] = pd.to_numeric(out[close_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(spot: float, strike: float, ttm: float, rate: float, sigma: float, option_type: str) -> float:
    if spot <= 0 or strike <= 0 or ttm <= 0 or sigma <= 0:
        return np.nan
    sqrt_t = math.sqrt(ttm)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * ttm) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if option_type == "C":
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * ttm) * _norm_cdf(d2)
    return strike * math.exp(-rate * ttm) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _implied_vol(price: float, spot: float, strike: float, ttm: float, option_type: str, rate: float = 0.02) -> float:
    """用二分法反推 Black-Scholes 隐含波动率。"""
    try:
        price, spot, strike, ttm = float(price), float(spot), float(strike), float(ttm)
    except Exception:
        return np.nan
    if not all(np.isfinite([price, spot, strike, ttm])) or price <= 0 or spot <= 0 or strike <= 0 or ttm <= 0:
        return np.nan
    intrinsic = max(spot - strike, 0.0) if option_type == "C" else max(strike - spot, 0.0)
    upper_bound = spot if option_type == "C" else strike * math.exp(-rate * ttm)
    if price < intrinsic * 0.98 or price > upper_bound * 1.2:
        return np.nan
    low, high = 1e-4, 5.0
    for _ in range(80):
        mid = (low + high) / 2.0
        model_price = _bs_price(spot, strike, ttm, rate, mid, option_type)
        if not np.isfinite(model_price):
            return np.nan
        if model_price > price:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def _normalize_option_type(value: str) -> str:
    text = str(value or "").strip().upper()
    if text in {"C", "CALL"} or "购" in text:
        return "C"
    if text in {"P", "PUT"} or "沽" in text:
        return "P"
    return ""


def _calc_50etf_option_iv_indicator(option_daily_df: pd.DataFrame, option_basic_df: pd.DataFrame, etf50_df: pd.DataFrame) -> pd.DataFrame:
    """使用 50ETF 期权链反推近 30 日 ATM 隐含波动率。"""
    if option_daily_df is None or option_daily_df.empty or option_basic_df is None or option_basic_df.empty or etf50_df is None or etf50_df.empty:
        return pd.DataFrame()
    daily = option_daily_df.copy()
    basic = option_basic_df.copy()
    spot = _clean_price_frame(etf50_df).rename(columns={"close": "spot_close"})
    if spot.empty:
        return pd.DataFrame()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["option_price"] = pd.to_numeric(daily.get("settle", np.nan), errors="coerce").where(
        pd.to_numeric(daily.get("settle", np.nan), errors="coerce").gt(0),
        pd.to_numeric(daily.get("close", np.nan), errors="coerce"),
    )
    basic["exercise_price"] = pd.to_numeric(basic.get("exercise_price", np.nan), errors="coerce")
    basic["maturity_date"] = pd.to_datetime(basic.get("maturity_date", ""), errors="coerce")
    call_put = basic["call_put"] if "call_put" in basic.columns else pd.Series("", index=basic.index)
    basic["option_type"] = call_put.map(_normalize_option_type)
    merged = daily.merge(
        basic[["ts_code", "exercise_price", "maturity_date", "option_type"]],
        on="ts_code",
        how="left",
    ).merge(spot[["date", "spot_close"]], on="date", how="left")
    merged["days_to_maturity"] = (merged["maturity_date"] - merged["date"]).dt.days
    merged = merged.loc[
        merged["option_price"].gt(0)
        & merged["spot_close"].gt(0)
        & merged["exercise_price"].gt(0)
        & merged["days_to_maturity"].between(7, 90)
        & merged["option_type"].isin(["C", "P"])
    ].copy()
    if merged.empty:
        return pd.DataFrame()
    merged["maturity_distance"] = (merged["days_to_maturity"] - 30).abs()
    merged["moneyness_distance"] = np.log(merged["exercise_price"] / merged["spot_close"]).abs()
    merged = merged.sort_values(["date", "maturity_distance", "moneyness_distance"])
    selected = merged.groupby("date", group_keys=False).head(12).copy()
    selected["iv"] = [
        _implied_vol(row.option_price, row.spot_close, row.exercise_price, row.days_to_maturity / 365.0, row.option_type)
        for row in selected.itertuples(index=False)
    ]
    selected = selected.loc[selected["iv"].between(0.03, 1.0)].copy()
    if selected.empty:
        return pd.DataFrame()
    out = selected.groupby("date", as_index=False).agg(
        raw_market_volatility=("iv", "median"),
        option_iv_sample_count=("iv", "count"),
        option_iv_days_to_maturity=("days_to_maturity", "median"),
    )
    out["volatility_status"] = np.where(out["option_iv_sample_count"] >= 2, "ok", "missing")
    out["volatility_source"] = "50ETF期权链反推近30日ATM隐含波动率"
    out["volatility_message"] = np.where(out["volatility_status"].eq("ok"), "", "50ETF期权可用样本不足，使用中性分")
    return out[["date", "raw_market_volatility", "option_iv_sample_count", "option_iv_days_to_maturity", "volatility_status", "volatility_source", "volatility_message"]]


def _calc_realized_volatility_fallback(etf50_df: pd.DataFrame, hs300_df: pd.DataFrame) -> pd.DataFrame:
    """构造连续的历史波动率兜底序列：优先 50ETF，其次沪深300。"""
    source = "50ETF 20日年化历史波动率"
    price = _clean_price_frame(etf50_df)
    if price.empty or price["close"].notna().sum() < 21:
        source = "沪深300 20日年化历史波动率"
        price = _clean_price_frame(hs300_df)
    if price.empty:
        return pd.DataFrame(columns=["date", "raw_market_volatility", "volatility_status", "volatility_source", "volatility_message"])
    price["raw_market_volatility"] = price["close"].pct_change().rolling(20, min_periods=20).std() * math.sqrt(252)
    price["volatility_status"] = np.where(price["raw_market_volatility"].notna(), "ok", "missing")
    price["volatility_source"] = source
    price["volatility_message"] = np.where(price["raw_market_volatility"].notna(), "", "波动率样本不足，使用中性分")
    return price[["date", "raw_market_volatility", "volatility_status", "volatility_source", "volatility_message"]]


def _normalize_ivix_indicator(ivix_df: pd.DataFrame | None) -> pd.DataFrame:
    """清洗 iVIX/中国波指序列，统一为小数形式的年化波动率。"""
    if ivix_df is None or ivix_df.empty or "date" not in ivix_df.columns:
        return pd.DataFrame()
    out = ivix_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["raw_market_volatility"] = pd.to_numeric(out.get("raw_market_volatility", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "raw_market_volatility"])
    out = out.loc[out["raw_market_volatility"].between(0.03, 1.0)].copy()
    if out.empty:
        return pd.DataFrame()
    if "volatility_source" not in out.columns:
        out["volatility_source"] = "原中国波指 iVIX"
    out["volatility_status"] = "ok"
    out["volatility_message"] = ""
    return out[["date", "raw_market_volatility", "volatility_status", "volatility_source", "volatility_message"]].sort_values("date").drop_duplicates(subset=["date"], keep="last")


def _prefer_primary_volatility(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """有 iVIX/期权 IV 时优先使用，缺口日期使用历史波动率补齐。"""
    if primary is None or primary.empty:
        return fallback
    if fallback is None or fallback.empty:
        return primary
    merged = fallback.merge(primary, on="date", how="outer", suffixes=("_fallback", "_primary")).sort_values("date")
    out = pd.DataFrame({"date": merged["date"]})
    for col in ["raw_market_volatility", "volatility_status", "volatility_source", "volatility_message"]:
        out[col] = merged[f"{col}_primary"].where(merged[f"{col}_primary"].notna(), merged[f"{col}_fallback"])
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _has_enough_coverage(candidate: pd.DataFrame, base_dates: pd.Series, min_ratio: float = 0.80, min_count: int = 60) -> bool:
    """判断候选数据源是否足够完整，避免不同口径在同一条曲线中硬拼接。"""
    if candidate is None or candidate.empty or "date" not in candidate.columns:
        return False
    candidate_dates = set(pd.to_datetime(candidate["date"], errors="coerce").dropna())
    base = pd.to_datetime(base_dates, errors="coerce").dropna()
    if base.empty:
        return len(candidate_dates) >= min_count
    covered = len(candidate_dates.intersection(set(base)))
    return covered >= min_count and covered / max(len(base), 1) >= min_ratio


def calc_market_volatility_indicator(
    etf50_df: pd.DataFrame,
    hs300_df: pd.DataFrame,
    option_daily_df: pd.DataFrame | None = None,
    option_basic_df: pd.DataFrame | None = None,
    ivix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    市场波动率原始指标。

    优先用原中国波指 iVIX；失败时使用 50ETF 期权链反推 IV；
    仍不可用时使用 50ETF/沪深300 历史波动率。
    """
    fallback = _calc_realized_volatility_fallback(etf50_df, hs300_df)
    base_dates = fallback["date"] if fallback is not None and not fallback.empty else pd.Series(dtype="datetime64[ns]")
    ivix = _normalize_ivix_indicator(ivix_df)
    if _has_enough_coverage(ivix, base_dates):
        return ivix
    option_iv = _calc_50etf_option_iv_indicator(option_daily_df, option_basic_df, etf50_df)
    if _has_enough_coverage(option_iv, base_dates):
        return option_iv
    if (fallback is None or fallback.empty) and option_iv is not None and not option_iv.empty:
        return option_iv
    return fallback


def calc_leverage_indicator(margin_df: pd.DataFrame) -> pd.DataFrame:
    """杠杆率原始数据：A股融资买入金额，稍后与全A成交额合成比例。"""
    if margin_df is None or margin_df.empty:
        return pd.DataFrame(columns=["date", "raw_leverage_ratio", "margin_buy_amount_yuan", "margin_record_count", "margin_exchange_count", "leverage_status", "leverage_source", "leverage_message"])
    df = margin_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["margin_buy_amount"] = pd.to_numeric(df["margin_buy_amount"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    median_abs = df["margin_buy_amount"].abs().median()
    # 融资买入额不同来源可能是元、万元或亿元，按典型量级做宽松换算。
    if pd.isna(median_abs):
        unit_multiplier = np.nan
    elif median_abs < 10000:
        unit_multiplier = 100000000.0
    elif median_abs < 1000000000:
        unit_multiplier = 10000.0
    else:
        unit_multiplier = 1.0
    df["margin_buy_amount_yuan"] = df["margin_buy_amount"] * unit_multiplier
    if "margin_record_count" not in df.columns:
        df["margin_record_count"] = np.nan
    if "margin_exchange_count" not in df.columns:
        df["margin_exchange_count"] = np.nan
    df["raw_leverage_ratio"] = np.nan
    df["leverage_status"] = np.where(df["margin_buy_amount_yuan"].notna(), "missing", "missing")
    source = df.get("margin_source", pd.Series("融资买入额", index=df.index)).fillna("融资买入额").astype(str)
    df["leverage_source"] = source
    df["leverage_message"] = "等待合并全A成交额后计算融资买入占比"
    return df[
        [
            "date",
            "raw_leverage_ratio",
            "margin_buy_amount",
            "margin_buy_amount_yuan",
            "margin_record_count",
            "margin_exchange_count",
            "leverage_status",
            "leverage_source",
            "leverage_message",
        ]
    ]


def calc_market_volume_indicator(daily_df: pd.DataFrame, index_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """两市成交活跃度：全 A 股成交额 / 20 日均值，缺失时用主要指数成交额代理。"""
    if daily_df is not None and not daily_df.empty and "amount" in daily_df.columns:
        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        amount = df.groupby("date", as_index=False).agg(total_amount=("amount", "sum"))
        source = "Tushare全A日线成交额"
    else:
        frames = []
        for key in ["hs300", "zz1000", "cyb", "sh"]:
            idx = index_frames.get(key, pd.DataFrame())
            if idx is not None and not idx.empty and "amount" in idx.columns:
                work = idx[["date", "amount"]].copy()
                work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
                frames.append(work)
        if not frames:
            return pd.DataFrame(columns=["date", "raw_amount_ratio", "total_amount", "volume_status", "volume_source", "volume_message"])
        amount = pd.concat(frames, ignore_index=True).groupby("date", as_index=False).agg(total_amount=("amount", "sum"))
        source = "主要指数成交额代理"
    amount = amount.sort_values("date")
    amount["total_amount_yuan"] = amount["total_amount"] * 1000.0
    amount["raw_amount_ratio"] = amount["total_amount"] / amount["total_amount"].rolling(20, min_periods=5).mean()
    amount["volume_status"] = np.where(amount["raw_amount_ratio"].notna(), "ok", "missing")
    amount["volume_source"] = source
    amount["volume_message"] = np.where(amount["raw_amount_ratio"].notna(), "", "成交额样本不足，使用中性分")
    return amount[["date", "raw_amount_ratio", "total_amount", "total_amount_yuan", "volume_status", "volume_source", "volume_message"]]


def calc_futures_basis_indicator(futures_df: pd.DataFrame, hs300_df: pd.DataFrame) -> pd.DataFrame:
    """沪深300股指期货近似升贴水率。

    使用 IF 代理/连续合约相对沪深300现货指数的简单贴水率：
    future_close / spot_close - 1，不按到期天数年化。
    """
    if futures_df is None or futures_df.empty or hs300_df is None or hs300_df.empty:
        return pd.DataFrame(columns=["date", "raw_futures_basis", "future_close", "spot_close", "futures_basis_status", "futures_basis_source", "futures_basis_message"])
    fut = futures_df.copy()
    fut["date"] = pd.to_datetime(fut["date"], errors="coerce")
    fut["future_close"] = pd.to_numeric(fut.get("close", np.nan), errors="coerce")
    fut["ts_code"] = fut.get("ts_code", "").fillna("").astype(str).str.upper()
    spot = _clean_price_frame(hs300_df).rename(columns={"close": "spot_close"})
    # 只允许真正的 IF 主力/连续代码参与近似升贴水率。
    # IFL1/IFL2/IFL3 是中金所序列/阶梯合约代码，不能当作主力连续价格和沪深300直接比较。
    proxy = fut.loc[fut["ts_code"].isin(IF_CONTINUOUS_CODES) & fut["future_close"].gt(0)].copy()
    proxy_selected = pd.DataFrame()
    if not proxy.empty:
        priority = {"IF.CFX": 0, "IF9999.CFE": 1, "IF9999.CFX": 2, "IF0.AK": 3}
        proxy = proxy.copy()
        proxy["proxy_priority"] = proxy["ts_code"].map(priority).fillna(99)
        proxy_selected = proxy.sort_values(["date", "proxy_priority", "ts_code"]).drop_duplicates(subset=["date"], keep="first").copy()
        proxy_selected["days_to_maturity"] = np.nan
        proxy_selected["futures_basis_source"] = "沪深300股指期货近似升贴水率"
    if proxy_selected.empty:
        return pd.DataFrame(columns=["date", "raw_futures_basis", "future_close", "spot_close", "futures_basis_status", "futures_basis_source", "futures_basis_message"])
    fut_selected = proxy_selected
    df = fut_selected[["date", "ts_code", "future_close", "days_to_maturity"]].merge(spot[["date", "spot_close"]], on="date", how="inner")
    if df.empty:
        return pd.DataFrame(columns=["date", "raw_futures_basis", "future_close", "spot_close", "futures_basis_status", "futures_basis_source", "futures_basis_message"])
    df = df.merge(fut_selected[["date", "futures_basis_source"]].drop_duplicates(subset=["date"], keep="last"), on="date", how="left")
    simple_basis = df["future_close"] / df["spot_close"] - 1
    df["raw_futures_basis_simple"] = simple_basis
    df["raw_futures_basis_daily"] = simple_basis
    df = df.sort_values("date")
    df["raw_futures_basis"] = pd.to_numeric(df["raw_futures_basis_simple"], errors="coerce")
    df["futures_basis_status"] = np.where(df["raw_futures_basis"].notna(), "ok", "missing")
    df["futures_basis_source"] = df["futures_basis_source"].fillna("沪深300股指期货近似升贴水率")
    df["futures_basis_message"] = np.where(df["raw_futures_basis"].notna(), "", "股指期货升贴水样本不足，使用中性分")
    return df[["date", "raw_futures_basis", "raw_futures_basis_daily", "raw_futures_basis_simple", "future_close", "spot_close", "days_to_maturity", "ts_code", "futures_basis_status", "futures_basis_source", "futures_basis_message"]]


def calc_price_strength_indicator(daily_df: pd.DataFrame, stock_basic_df: pd.DataFrame) -> pd.DataFrame:
    """股价强度：创 252 日收盘新高股票占当日有效股票比例。"""
    if daily_df is None or daily_df.empty or "close" not in daily_df.columns:
        return pd.DataFrame(columns=["date", "raw_new_high_ratio", "new_high_count", "effective_stock_count", "price_strength_status", "price_strength_source", "price_strength_message"])
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "ts_code", "close"]).sort_values(["ts_code", "date"])
    if stock_basic_df is not None and not stock_basic_df.empty and {"ts_code", "name"}.issubset(stock_basic_df.columns):
        basic = stock_basic_df[["ts_code", "name", "list_date"] if "list_date" in stock_basic_df.columns else ["ts_code", "name"]].copy()
        df = df.merge(basic, on="ts_code", how="left")
        name = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
        df = df.loc[~name.str.contains("ST", regex=False)].copy()
        if "list_date" in df.columns:
            listed = pd.to_datetime(df["list_date"], errors="coerce")
            df = df.loc[listed.isna() | ((df["date"] - listed).dt.days >= 250)].copy()
    rolling_high = df.groupby("ts_code")["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    df["is_252d_high"] = rolling_high.notna() & (df["close"] >= rolling_high)
    df["has_strength_sample"] = rolling_high.notna()
    out = (
        df.groupby("date", as_index=False)
        .agg(
            raw_new_high_ratio=("is_252d_high", lambda s: float(s.sum()) / max(len(s), 1)),
            new_high_count=("is_252d_high", "sum"),
            effective_stock_count=("has_strength_sample", "sum"),
        )
        .sort_values("date")
    )
    out["raw_new_high_ratio"] = out["new_high_count"] / out["effective_stock_count"].replace(0, np.nan)
    out["price_strength_status"] = np.where(out["raw_new_high_ratio"].notna(), "ok", "missing")
    out["price_strength_source"] = "全A 252日新高占比，剔除ST和样本不足股票"
    out["price_strength_message"] = np.where(out["raw_new_high_ratio"].notna(), "", "新高样本不足，使用中性分")
    return out[["date", "raw_new_high_ratio", "new_high_count", "effective_stock_count", "price_strength_status", "price_strength_source", "price_strength_message"]]


def calc_risk_appetite_indicator(hs300_df: pd.DataFrame, bond_df: pd.DataFrame) -> pd.DataFrame:
    """风险偏好：沪深300 20日收益率 - 债券ETF 20日收益率。"""
    stock = _clean_price_frame(hs300_df).rename(columns={"close": "hs300_close"})
    bond = _clean_price_frame(bond_df).rename(columns={"close": "bond_close"})
    if stock.empty or bond.empty:
        return pd.DataFrame(columns=["date", "raw_risk_appetite", "raw_hs300_ret_20d", "raw_bond_ret_20d", "risk_appetite_status", "risk_appetite_source", "risk_appetite_message"])
    df = stock[["date", "hs300_close"]].merge(bond[["date", "bond_close"]], on="date", how="left").sort_values("date")
    df["bond_close"] = df["bond_close"].ffill()
    df["raw_hs300_ret_20d"] = df["hs300_close"].pct_change(20)
    df["raw_bond_ret_20d"] = df["bond_close"].pct_change(20)
    df["raw_risk_appetite"] = df["raw_hs300_ret_20d"] - df["raw_bond_ret_20d"]
    df["risk_appetite_status"] = np.where(df["raw_risk_appetite"].notna(), "ok", "missing")
    df["risk_appetite_source"] = "沪深300收益率 - 国债ETF收益率"
    df["risk_appetite_message"] = np.where(df["raw_risk_appetite"].notna(), "", "债券代理样本不足，使用中性分")
    return df[["date", "raw_risk_appetite", "raw_hs300_ret_20d", "raw_bond_ret_20d", "risk_appetite_status", "risk_appetite_source", "risk_appetite_message"]]


def build_raw_indicator_timeseries(source_data: Dict[str, object]) -> pd.DataFrame:
    """把 5 个入选原始指标和杠杆率观察项拼成同一张按日期排序的表。"""
    index_frames = source_data.get("index", {})
    hs300 = index_frames.get("hs300", pd.DataFrame()) if isinstance(index_frames, dict) else pd.DataFrame()
    frames = [
        calc_market_volatility_indicator(
            source_data.get("etf50", pd.DataFrame()),
            hs300,
            source_data.get("option_daily", pd.DataFrame()),
            source_data.get("option_basic", pd.DataFrame()),
            source_data.get("ivix", pd.DataFrame()),
        ),
        calc_leverage_indicator(source_data.get("margin", pd.DataFrame())),
        calc_market_volume_indicator(source_data.get("daily", pd.DataFrame()), index_frames if isinstance(index_frames, dict) else {}),
        calc_futures_basis_indicator(source_data.get("futures_if", pd.DataFrame()), hs300),
        calc_price_strength_indicator(source_data.get("daily", pd.DataFrame()), source_data.get("stock_basic", pd.DataFrame())),
        calc_risk_appetite_indicator(hs300, source_data.get("bond_etf", pd.DataFrame())),
    ]
    base_dates = source_data.get("trade_dates", [])
    if base_dates:
        result = pd.DataFrame({"trade_date": base_dates})
        result["date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    else:
        dates = sorted(set().union(*[set(pd.to_datetime(frame["date"], errors="coerce").dropna()) for frame in frames if not frame.empty and "date" in frame.columns]))
        result = pd.DataFrame({"date": dates})
    for frame in frames:
        if frame is not None and not frame.empty and "date" in frame.columns:
            result = result.merge(frame, on="date", how="left")
    if {"margin_buy_amount_yuan", "total_amount_yuan"}.issubset(result.columns):
        leverage_ratio = pd.to_numeric(result["margin_buy_amount_yuan"], errors="coerce") / pd.to_numeric(result["total_amount_yuan"], errors="coerce").replace(0, np.nan)
        usable = leverage_ratio.notna()
        result.loc[usable, "raw_leverage_ratio"] = leverage_ratio.loc[usable]
        result.loc[usable, "leverage_source"] = result.loc[usable, "leverage_source"].fillna("融资买入额").astype(str) + " / 全A成交额"
        result.loc[usable, "leverage_message"] = ""
        result.loc[usable, "leverage_status"] = "ok"
        margin_yuan = pd.to_numeric(result["margin_buy_amount_yuan"], errors="coerce")
        recent_median = margin_yuan.shift(1).rolling(20, min_periods=5).median()
        partial_amount = margin_yuan.notna() & recent_median.notna() & (margin_yuan < recent_median * 0.65)
        if "margin_exchange_count" in result.columns:
            exchange_count = pd.to_numeric(result["margin_exchange_count"], errors="coerce")
            recent_exchange_max = exchange_count.shift(1).rolling(20, min_periods=5).max()
            partial_exchange = exchange_count.notna() & recent_exchange_max.notna() & (recent_exchange_max >= 2) & (exchange_count < 2)
        else:
            partial_exchange = pd.Series(False, index=result.index)
        partial_mask = usable & (partial_amount | partial_exchange)
        result.loc[partial_mask, "raw_leverage_ratio"] = np.nan
        result.loc[partial_mask, "leverage_status"] = "missing"
        result.loc[partial_mask, "leverage_message"] = "融资买入额疑似只返回部分市场或最新日尚未更新完整，当前杠杆率仅作观察项展示"
    if "trade_date" not in result.columns:
        result["trade_date"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y%m%d")
    return result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

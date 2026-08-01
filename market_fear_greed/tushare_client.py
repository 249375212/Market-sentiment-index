from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import tushare as ts
except ImportError:
    ts = None


PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _normalize_sw_code(code: str) -> str:
    raw = str(code or "").strip().upper()
    if not raw:
        return raw
    return raw if raw.endswith(".SI") else f"{raw}.SI"


def _disable_broken_local_proxy() -> None:
    bad_targets = ("127.0.0.1:9", "localhost:9")
    for key in PROXY_ENV_KEYS:
        value = str(os.environ.get(key, "")).strip().lower()
        if any(target in value for target in bad_targets):
            os.environ.pop(key, None)


def get_default_token() -> str:
    environment_token = os.getenv("TUSHARE_TOKEN", "").strip()
    if environment_token:
        return environment_token
    project_secrets = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    user_secrets = Path.home() / ".streamlit" / "secrets.toml"
    if not project_secrets.exists() and not user_secrets.exists():
        return ""
    try:
        return str(st.secrets.get("TUSHARE_TOKEN", "")).strip()
    except Exception:
        return ""


def ensure_tushare() -> None:
    if ts is None:
        raise ImportError("未检测到 tushare，请先运行：pip install -r requirements.txt")


@st.cache_resource(show_spinner=False)
def init_pro(token: str):
    ensure_tushare()
    if not token:
        raise ValueError("Tushare Token 不能为空。")
    _disable_broken_local_proxy()
    return ts.pro_api(token)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_sw_industry_classify(token: str, level: str = "L1", src: str = "SW2021") -> pd.DataFrame:
    pro = init_pro(token)
    df = pro.index_classify(level=level, src=src)
    if df is None or df.empty:
        return pd.DataFrame(columns=["index_code", "industry_name"])
    for col in ("index_code", "industry_name", "parent_code", "level", "industry_code", "is_pub", "src"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df["index_code"] = df["index_code"].map(_normalize_sw_code)
    return df.sort_values(["industry_name", "index_code"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_sw_index_history(token: str, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    pro = init_pro(token)
    df = pro.sw_daily(ts_code=_normalize_sw_code(ts_code), start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"trade_date": "date", "vol": "volume", "pct_change": "pct_chg"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "change", "pct_chg", "volume", "amount", "pe", "pb", "float_mv"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

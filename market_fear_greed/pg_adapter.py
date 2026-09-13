"""PostgreSQL 数据源适配层。

用本地 ``reasonix_db`` 数据库完全替换 Tushare / AkShare 数据源，使
Market-sentiment-index 项目可以在不依赖任何外部行情接口的情况下运行。

启用方式
--------
设置环境变量 ``AFGI_DATA_SOURCE=postgres``，然后在应用入口（app.py）启动时
调用 :func:`install_pg_adapter` 即可。本模块会把项目各处引用的数据源函数
（``load_all_stock_daily`` / ``load_index_daily`` / ``load_fund_daily`` /
``load_stock_basic_safe`` / ``load_trade_dates`` / ``load_sw_*`` /
``load_market_fear_greed_source_data``）替换为从 PostgreSQL 取数的实现。

数据库连接
----------
默认读取环境变量 ``DATABASE_URL``（例如
``postgresql://user:password@localhost:5432/reasonix_db``），也支持
``PG_DSN``。需已安装 ``psycopg2``。

数据映射概要
------------
- 个股日线: ``stock_daily`` (+ ``security_master`` 构造 ts_code)
- ETF 日线: ``etf_daily``
- 指数日线: ``index_daily``；中证1000/500 缺失时用
  ``index_constituent_weight`` + ``stock_daily`` 加权聚合
- 交易日历: ``trading_calendar`` (market='stock_cn', is_trading_day=true)
- 股票基本信息: ``security_master`` + ``industry_info`` (申万一级)
- 申万行业指数: 本库无，返回空触发降级到 ``calc_stock_industry_up_ratio``
  （用 stock_basic.industry 个股聚合，板块扩散情绪仍可计算）
- ivix / option / margin / futures_if: 本库无对应业务表，返回空，
  由指标层降级到中性值 50 或历史波动率
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import psycopg2
except ImportError as exc:  # pragma: no cover
    psycopg2 = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# --------------------------------------------------------------------------- #
# 连接管理
# --------------------------------------------------------------------------- #
_DSN = os.getenv("DATABASE_URL") or os.getenv("PG_DSN", "")
_conn = None


def _get_conn():
    if psycopg2 is None:
        raise ImportError(
            "未安装 psycopg2，请先运行: pip install psycopg2-binary"
        ) from _IMPORT_ERROR
    global _conn
    if _conn is None or _conn.closed:
        if not _DSN:
            raise RuntimeError(
                "未配置数据库连接，请设置环境变量 DATABASE_URL 或 PG_DSN"
            )
        _conn = psycopg2.connect(_DSN)
        _conn.autocommit = True
    return _conn


def _query_df(sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """执行 SQL 并返回 DataFrame；失败时回滚并抛出。

    使用 psycopg2 原生 cursor 取数后构造 DataFrame，避免 pandas 3.x 对
    DBAPI2 连接的 SQLAlchemy 警告，同时保持参数化查询安全。
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


# --------------------------------------------------------------------------- #
# 轻量缓存（按参数键缓存，避免重复查库）
# --------------------------------------------------------------------------- #
_CACHE: Dict[tuple, object] = {}


def _cached(key, factory):
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def clear_cache() -> None:
    """清空本适配层的内存缓存。"""
    _CACHE.clear()


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _to_date_str(value) -> str:
    """任意日期 -> 'YYYYMMDD' 字符串。"""
    return pd.Timestamp(value).strftime("%Y%m%d")


def _exchange_suffix(security_id: str) -> str:
    """从 security_id 解析交易所后缀: XSHG->SH, XSHE->SZ, 其他->SH。

    security_id 形如 ``XSHG:600000`` / ``XSHE:000001`` / ``INDEX:XSHG:000300``。
    """
    parts = str(security_id or "").split(":")
    prefix = "INDEX" if parts[0] == "INDEX" and len(parts) >= 3 else parts[0]
    if prefix == "INDEX" and len(parts) >= 3:
        prefix = parts[1]
    return {"XSHG": "SH", "XSHE": "SZ", "BJSE": "BJ", "XBJ": "BJ", "BSE": "BJ"}.get(
        prefix, "SH"
    )


def _make_ts_code(symbol: str, security_id: str) -> str:
    return f"{symbol}.{_exchange_suffix(security_id)}"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """把 *_price / volume / pct_change 列重命名为项目期望的 Tushare 风格列名。"""
    if df is None or df.empty:
        return df
    rename = {
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
        "pct_change": "pct_chg",
        "volume": "vol",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    return df


_DAILY_COLUMNS = [
    "ts_code", "trade_date", "open", "high", "low", "close",
    "pre_close", "pct_chg", "vol", "amount", "date",
]


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame(columns=_DAILY_COLUMNS)


# --------------------------------------------------------------------------- #
# 交易日历
# --------------------------------------------------------------------------- #
def pg_load_trade_calendar(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """交易日历，返回 [trade_date(YYYYMMDD), date(datetime)]。"""
    start, end = _to_date_str(start_date), _to_date_str(end_date)

    def factory():
        sql = (
            "SELECT to_char(tc.trade_date, 'YYYYMMDD') AS trade_date, tc.trade_date AS date "
            "FROM trading_calendar tc "
            "WHERE tc.market = 'stock_cn' AND tc.is_trading_day = true "
            "  AND tc.trade_date BETWEEN to_date(%s,'YYYYMMDD') AND to_date(%s,'YYYYMMDD') "
            "  AND EXISTS (SELECT 1 FROM stock_daily sd WHERE sd.trade_date = tc.trade_date) "
            "ORDER BY tc.trade_date"
        )
        df = _query_df(sql, (start, end))
        df["date"] = pd.to_datetime(df["date"])
        df["trade_date"] = df["trade_date"].astype(str)
        return df.reset_index(drop=True)

    return _cached(("trade_cal", start, end), factory)


def pg_load_trade_dates(token: str, start_date: str, end_date: str) -> List[str]:
    return (
        pg_load_trade_calendar(token, start_date, end_date)["trade_date"]
        .dropna()
        .astype(str)
        .tolist()
    )


# --------------------------------------------------------------------------- #
# 股票基本信息
# --------------------------------------------------------------------------- #
def pg_load_stock_basic_safe(token: str) -> pd.DataFrame:
    """返回 [ts_code, symbol, name, market, industry, list_date]。

    industry 取申万一级（``industry_info.level = 'L1'``）最新有效分类。
    """
    def factory():
        sql = """
            WITH first_dates AS (
                SELECT security_id, MIN(trade_date) AS first_date
                FROM stock_daily GROUP BY security_id
            )
            SELECT sm.symbol, sm.security_id, sm.name, sm.exchange,
                   COALESCE(sm.list_date, fd.first_date) AS list_date,
                   ii.industry_name AS industry
            FROM security_master sm
            LEFT JOIN first_dates fd ON fd.security_id = sm.security_id
            LEFT JOIN LATERAL (
                SELECT industry_name
                FROM industry_info
                WHERE industry_info.security_id = sm.security_id
                  AND industry_info.level = 'L1'
                  AND (industry_info.expire_date IS NULL
                       OR industry_info.expire_date > CURRENT_DATE)
                ORDER BY industry_info.effective_date DESC
                LIMIT 1
            ) ii ON true
            WHERE sm.asset_type = 'stock_cn'
              AND sm.list_status = 'active'
            ORDER BY sm.symbol
        """
        df = _query_df(sql)
        if df.empty:
            return pd.DataFrame(
                columns=["ts_code", "symbol", "name", "market", "industry", "list_date"]
            )
        df["ts_code"] = df.apply(
            lambda r: _make_ts_code(r["symbol"], r["security_id"]), axis=1
        )
        df["market"] = df["exchange"].map(
            {"XSHG": "SSE", "XSHE": "SZSE", "BJSE": "BSE"}
        )
        for col in ["ts_code", "symbol", "name", "market", "industry"]:
            df[col] = df[col].fillna("").astype(str)
        # list_date: date -> 'YYYYMMDD' 字符串（与 Tushare 一致），NaT -> ''
        ld = pd.to_datetime(df["list_date"], errors="coerce")
        df["list_date"] = ld.dt.strftime("%Y%m%d").fillna("")
        return df[
            ["ts_code", "symbol", "name", "market", "industry", "list_date"]
        ].reset_index(drop=True)

    return _cached("stock_basic", factory)


# --------------------------------------------------------------------------- #
# 个股日线
# --------------------------------------------------------------------------- #
def pg_load_all_stock_daily(token: str, trade_date: str) -> pd.DataFrame:
    """单日全市场个股日线，返回 Tushare daily 风格 DataFrame。"""
    td = _to_date_str(trade_date)

    def factory():
        sql = """
            SELECT symbol, security_id,
                   open_price, high_price, low_price, close_price,
                   pre_close, pct_change, volume, amount
            FROM stock_daily
            WHERE trade_date = to_date(%s, 'YYYYMMDD')
        """
        df = _query_df(sql, (td,))
        if df.empty:
            return _empty_daily()
        df["ts_code"] = df.apply(
            lambda r: _make_ts_code(r["symbol"], r["security_id"]), axis=1
        )
        df["trade_date"] = td
        df["date"] = pd.to_datetime(td)
        df = _normalize_ohlcv(df)
        return df[_DAILY_COLUMNS].reset_index(drop=True)

    return _cached(("stock_daily", td), factory)


def _load_all_stock_range(start: str, end: str) -> pd.DataFrame:
    """日期范围内的全市场个股日线（用于原始 5 指标的 volume / price_strength）。

    只取指标实际需要的列以减少传输：close、amount、pre_close、pct_chg、high。
    """
    sql = """
        SELECT symbol, security_id,
               to_char(trade_date, 'YYYYMMDD') AS trade_date,
               trade_date AS date,
               close_price, amount
        FROM stock_daily
        WHERE trade_date BETWEEN to_date(%s, 'YYYYMMDD') AND to_date(%s, 'YYYYMMDD')
    """
    df = _query_df(sql, (start, end))
    if df.empty:
        return _empty_daily()
    df["ts_code"] = df.apply(
        lambda r: _make_ts_code(r["symbol"], r["security_id"]), axis=1
    )
    df["trade_date"] = df["trade_date"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = _normalize_ohlcv(df)
    for col in _DAILY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[_DAILY_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 指数日线
# --------------------------------------------------------------------------- #
# 项目 INDEX_CODES: hs300=000300.SH, sh=000001.SH, zz1000=000852.SH, cyb=399006.SZ
_INDEX_SYMBOL_BY_TSCODE = {
    "000300.SH": "000300",
    "000852.SH": "000852",
    "000905.SH": "000905",
    "000001.SH": "000001",
    "399006.SZ": "399006",
    "399001.SZ": "399001",
    "000016.SH": "000016",
}
# index_daily 中缺失、需用成分股加权聚合的指数
_AGGREGATE_INDEX_SYMBOLS = {"000852", "000905"}
# 交易所指数偶尔比 stock_daily 少最后一个交易日，用本地股票市值加权收益连续延伸。
_INDEX_TAIL_PROXY_FILTERS = {
    "000001": "sm.exchange = 'XSHG'",
    "399006": "sm.exchange = 'XSHE' AND (sm.symbol LIKE '300%%' OR sm.symbol LIKE '301%%')",
}
_INDEX_TAIL_PROXY_SHARES = {
    "000001": "v.total_shares",
    "399006": "v.float_shares",
}


def pg_load_index_daily(
    token: str, ts_code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """指数日线。

    中证1000/500 整段缺失时用成分股加权聚合；上证指数/创业板指若仅尾部
    少于本地股票交易日，则以最后一个真实指数收盘价为锚，用本地股票市值
    加权收益连续延伸，避免把股票均价误当成指数点位。
    """
    start, end = _to_date_str(start_date), _to_date_str(end_date)
    sym = _INDEX_SYMBOL_BY_TSCODE.get(ts_code, ts_code.split(".")[0])

    def factory():
        sql = """
            SELECT to_char(trade_date, 'YYYYMMDD') AS trade_date, trade_date AS date,
                   open_price, high_price, low_price, close_price,
                   pre_close, pct_change, volume, amount
            FROM index_daily
            WHERE symbol = %s
              AND trade_date BETWEEN to_date(%s, 'YYYYMMDD') AND to_date(%s, 'YYYYMMDD')
            ORDER BY trade_date
        """
        df = _query_df(sql, (sym, start, end))
        if sym in _INDEX_TAIL_PROXY_FILTERS:
            df = _extend_index_tail_from_stock_returns(df, sym, start, end)
        if df.empty and sym in _AGGREGATE_INDEX_SYMBOLS:
            df = _aggregate_index_by_constituents(sym, start, end)
        if df.empty and sym == "399006":
            df = _aggregate_cyb_index(start, end)
        if df.empty:
            return _empty_daily()
        df["ts_code"] = ts_code
        df["trade_date"] = df["trade_date"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        df = _normalize_ohlcv(df)
        return df[_DAILY_COLUMNS].sort_values("date").reset_index(drop=True)

    return _cached(("index_daily", ts_code, start, end), factory)


def _load_index_tail_proxy_returns(
    index_symbol: str, start_exclusive: str, end: str
) -> pd.DataFrame:
    """读取指数真实数据尾部之后的本地股票市值加权收益。"""
    stock_filter = _INDEX_TAIL_PROXY_FILTERS.get(index_symbol)
    shares = _INDEX_TAIL_PROXY_SHARES.get(index_symbol)
    if not stock_filter or not shares:
        return pd.DataFrame(columns=["trade_date", "date", "return_ratio"])

    sql = f"""
        WITH stock_returns AS (
            SELECT d.trade_date,
                   d.close_price / NULLIF(d.pre_close, 0) - 1.0 AS return_ratio,
                   d.pre_close * {shares} AS previous_market_cap
            FROM stock_daily d
            JOIN security_master sm ON d.security_id = sm.security_id
            JOIN stock_valuation_daily v
              ON v.security_id = d.security_id AND v.trade_date = d.trade_date
            WHERE {stock_filter}
              AND sm.asset_type = 'stock_cn'
              AND d.trade_date > to_date(%s, 'YYYYMMDD')
              AND d.trade_date <= to_date(%s, 'YYYYMMDD')
              AND d.close_price > 0
              AND d.pre_close > 0
              AND {shares} > 0
        )
        SELECT to_char(trade_date, 'YYYYMMDD') AS trade_date,
               trade_date AS date,
               SUM(return_ratio * previous_market_cap)
                   / NULLIF(SUM(previous_market_cap), 0) AS return_ratio
        FROM stock_returns
        GROUP BY trade_date
        ORDER BY trade_date
    """
    return _query_df(sql, (start_exclusive, end))


def _extend_index_tail_from_stock_returns(
    index_df: pd.DataFrame, index_symbol: str, start: str, end: str
) -> pd.DataFrame:
    """以最后一个真实指数点位为锚，仅合成其后的缺失交易日。"""
    frame = index_df.copy() if index_df is not None else pd.DataFrame()
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close_price"] = pd.to_numeric(frame["close_price"], errors="coerce")
        anchors = frame.dropna(subset=["date", "close_price"]).sort_values("date")
    else:
        anchors = pd.DataFrame()

    if anchors.empty:
        anchor_sql = """
            SELECT to_char(trade_date, 'YYYYMMDD') AS trade_date, trade_date AS date,
                   open_price, high_price, low_price, close_price,
                   pre_close, pct_change, volume, amount
            FROM index_daily
            WHERE symbol = %s
              AND trade_date < to_date(%s, 'YYYYMMDD')
            ORDER BY trade_date DESC
            LIMIT 1
        """
        anchors = _query_df(anchor_sql, (index_symbol, start))
        if anchors.empty:
            return frame
        anchors["date"] = pd.to_datetime(anchors["date"], errors="coerce")
        anchors["close_price"] = pd.to_numeric(
            anchors["close_price"], errors="coerce"
        )
        anchors = anchors.dropna(subset=["date", "close_price"]).sort_values("date")
        if anchors.empty:
            return frame

    anchor = anchors.iloc[-1]
    anchor_date = pd.Timestamp(anchor["date"])
    if anchor_date >= pd.to_datetime(end):
        return frame

    proxy = _load_index_tail_proxy_returns(
        index_symbol, anchor_date.strftime("%Y%m%d"), end
    )
    if proxy is None or proxy.empty:
        return frame

    proxy = proxy.copy()
    proxy["date"] = pd.to_datetime(proxy["date"], errors="coerce")
    proxy["return_ratio"] = pd.to_numeric(proxy["return_ratio"], errors="coerce")
    proxy = proxy[
        (proxy["date"] > anchor_date)
        & proxy["date"].notna()
        & proxy["return_ratio"].map(np.isfinite)
    ].sort_values("date")
    if proxy.empty:
        return frame

    previous_close = float(anchor["close_price"])
    requested_start = pd.to_datetime(start)
    synthetic_rows = []
    for row in proxy.itertuples(index=False):
        row_date = pd.Timestamp(row.date)
        return_ratio = float(row.return_ratio)
        close_price = previous_close * (1.0 + return_ratio)
        if row_date < requested_start:
            previous_close = close_price
            continue
        synthetic_rows.append(
            {
                "trade_date": row_date.strftime("%Y%m%d"),
                "date": row_date,
                "open_price": np.nan,
                "high_price": np.nan,
                "low_price": np.nan,
                "close_price": close_price,
                "pre_close": previous_close,
                "pct_change": return_ratio * 100.0,
                "volume": np.nan,
                "amount": np.nan,
            }
        )
        previous_close = close_price

    if not synthetic_rows:
        return frame
    synthetic = pd.DataFrame(synthetic_rows)
    if frame.empty:
        return synthetic
    return pd.concat([frame, synthetic], ignore_index=True)


def _aggregate_index_by_constituents(
    index_symbol: str, start: str, end: str
) -> pd.DataFrame:
    """用 index_constituent_weight 最新一期成分股 + stock_daily 加权聚合指数 close。

    用于中证1000/500 指数日线缺失场景。聚合得到的是加权均价序列（未归一化），
    但相对强弱（5 日收益）与真实指数接近，足以支撑风格风险偏好分项（权重仅 5%）。
    """
    sql = """
        SELECT to_char(d.trade_date, 'YYYYMMDD') AS trade_date,
               d.trade_date AS date,
               SUM(d.close_price * w.weight) / NULLIF(SUM(w.weight), 0) AS close_price
        FROM stock_daily d
        JOIN index_constituent_weight w ON d.security_id = w.security_id
        WHERE w.index_symbol = %s
          AND w.trade_date = (
              SELECT MAX(trade_date) FROM index_constituent_weight
              WHERE index_symbol = %s
          )
          AND d.trade_date BETWEEN to_date(%s, 'YYYYMMDD') AND to_date(%s, 'YYYYMMDD')
        GROUP BY d.trade_date
        ORDER BY d.trade_date
    """
    df = _query_df(sql, (index_symbol, index_symbol, start, end))
    if df.empty:
        return df
    close = pd.to_numeric(df["close_price"], errors="coerce")
    df["pre_close"] = close.shift(1)
    df["pct_change"] = (close / df["pre_close"] - 1.0) * 100.0
    # 聚合序列仅 close 有意义，其余 OHLC / volume / amount 置空
    for col in ["open_price", "high_price", "low_price", "volume", "amount"]:
        df[col] = np.nan
    return df


def _aggregate_cyb_index(start: str, end: str) -> pd.DataFrame:
    """创业板指 (399006) 近似聚合：用 300/301 开头活跃股等权平均 close。

    真实创业板指有固定成分股与权重，本聚合为等权近似，趋势高度相关，
    数值水平有偏差，仅用于叠加图参考。
    """
    sql = """
        SELECT to_char(d.trade_date, 'YYYYMMDD') AS trade_date,
               d.trade_date AS date,
               AVG(d.close_price) AS close_price
        FROM stock_daily d
        JOIN security_master sm ON d.security_id = sm.security_id
        WHERE (sm.symbol LIKE '300%%' OR sm.symbol LIKE '301%%')
          AND sm.asset_type = 'stock_cn'
          AND sm.list_status = 'active'
          AND d.trade_date BETWEEN to_date(%s,'YYYYMMDD') AND to_date(%s,'YYYYMMDD')
        GROUP BY d.trade_date
        ORDER BY d.trade_date
    """
    df = _query_df(sql, (start, end))
    if df.empty:
        return df
    close = pd.to_numeric(df["close_price"], errors="coerce")
    df["pre_close"] = close.shift(1)
    df["pct_change"] = (close / df["pre_close"] - 1.0) * 100.0
    for col in ["open_price", "high_price", "low_price", "volume", "amount"]:
        df[col] = np.nan
    return df


# --------------------------------------------------------------------------- #
# ETF / 基金日线
# --------------------------------------------------------------------------- #
def pg_load_fund_daily(
    token: str, ts_code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """ETF 日线（510050 / 511010 等）。"""
    start, end = _to_date_str(start_date), _to_date_str(end_date)
    sym = ts_code.split(".")[0]

    def factory():
        sql = """
            SELECT to_char(trade_date, 'YYYYMMDD') AS trade_date, trade_date AS date,
                   open_price, high_price, low_price, close_price,
                   pre_close, pct_change, volume, amount
            FROM etf_daily
            WHERE symbol = %s
              AND trade_date BETWEEN to_date(%s, 'YYYYMMDD') AND to_date(%s, 'YYYYMMDD')
            ORDER BY trade_date
        """
        df = _query_df(sql, (sym, start, end))
        if df.empty:
            return _empty_daily()
        df["ts_code"] = ts_code
        df["trade_date"] = df["trade_date"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        df = _normalize_ohlcv(df)
        return df[_DAILY_COLUMNS].sort_values("date").reset_index(drop=True)

    return _cached(("fund_daily", ts_code, start, end), factory)


# --------------------------------------------------------------------------- #
# 申万行业（本库无行业指数日线，返回空触发降级到个股聚合）
# --------------------------------------------------------------------------- #
def pg_load_sw_industry_classify(
    token: str, level: str = "L1", src: str = "SW2021"
) -> pd.DataFrame:
    return pd.DataFrame(columns=["index_code", "industry_name"])


def pg_load_sw_index_history(
    token: str, ts_code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    return pd.DataFrame()


# --------------------------------------------------------------------------- #
# 原始 5 指标数据汇总（fear_greed_index.build_fear_greed_timeseries 入口）
# --------------------------------------------------------------------------- #
_INDEX_CODES = {
    "hs300": "000300.SH",
    "sh": "000001.SH",
    "zz1000": "000852.SH",
    "cyb": "399006.SZ",
}


def pg_load_market_fear_greed_source_data(
    token: str, start_date: str, end_date: str
) -> Dict[str, object]:
    """与 data_sources.load_market_fear_greed_source_data 同结构的字典。

    ivix / option / margin / futures_if 本库无对应业务表，返回空，
    由 indicators 层降级（波动率降级到 50ETF 历史波动率；其余降级到中性 50）。
    """
    start, end = _to_date_str(start_date), _to_date_str(end_date)
    trade_dates = pg_load_trade_dates(token, start_date, end_date)
    index_frames = {
        key: pg_load_index_daily(token, code, start_date, end_date)
        for key, code in _INDEX_CODES.items()
    }
    return {
        "trade_dates": trade_dates,
        "stock_basic": pg_load_stock_basic_safe(token),
        "daily": _load_all_stock_range(start, end),
        "index": index_frames,
        "ivix": pd.DataFrame(
            columns=["trade_date", "date", "raw_market_volatility", "volatility_source"]
        ),
        "etf50": pg_load_fund_daily(token, "510050.SH", start_date, end_date),
        "option_basic": pd.DataFrame(),
        "option_daily": pd.DataFrame(),
        "bond_etf": pg_load_fund_daily(token, "511010.SH", start_date, end_date),
        "margin": pd.DataFrame(columns=["trade_date", "date", "margin_buy_amount"]),
        "futures_if": pd.DataFrame(),
    }


# --------------------------------------------------------------------------- #
# 安装：把上述函数 patch 到项目各模块的命名空间
# --------------------------------------------------------------------------- #
# 模块名 -> {该模块内的属性名: 本模块对应的函数名}
_PATCH_PLAN = {
    "market_fear_greed.data_sources": {
        "load_trade_calendar": "pg_load_trade_calendar",
        "load_trade_dates": "pg_load_trade_dates",
        "load_stock_basic_safe": "pg_load_stock_basic_safe",
        "load_all_stock_daily": "pg_load_all_stock_daily",
        "load_index_daily": "pg_load_index_daily",
        "load_fund_daily": "pg_load_fund_daily",
        "load_market_fear_greed_source_data": "pg_load_market_fear_greed_source_data",
    },
    "market_fear_greed.fear_greed_index": {
        "load_index_daily": "pg_load_index_daily",
        "load_market_fear_greed_source_data": "pg_load_market_fear_greed_source_data",
    },
    "market_fear_greed.cache": {
        "load_trade_dates": "pg_load_trade_dates",
    },
    "market_fear_greed.enhanced_index": {
        "load_all_stock_daily": "pg_load_all_stock_daily",
        "load_fund_daily": "pg_load_fund_daily",
        "load_index_daily": "pg_load_index_daily",
        "load_stock_basic_safe": "pg_load_stock_basic_safe",
        "load_trade_dates": "pg_load_trade_dates",
        "load_sw_index_history": "pg_load_sw_index_history",
        "load_sw_industry_classify": "pg_load_sw_industry_classify",
    },
    "market_fear_greed.tushare_client": {
        "load_sw_industry_classify": "pg_load_sw_industry_classify",
        "load_sw_index_history": "pg_load_sw_index_history",
    },
}


def install_pg_adapter() -> int:
    """把 PostgreSQL 数据源函数注入到项目各模块，返回替换的函数数量。"""
    import importlib
    import sys

    this = sys.modules[__name__]
    patched = 0
    for mod_name, mapping in _PATCH_PLAN.items():
        mod = importlib.import_module(mod_name)
        for attr, pg_attr in mapping.items():
            func = getattr(this, pg_attr, None)
            if func is None:
                continue
            setattr(mod, attr, func)
            patched += 1
    warnings.warn(
        f"[pg_adapter] 已将 {patched} 个数据源函数替换为 PostgreSQL 本地库 "
        f"({_DSN.split('@')[-1] if _DSN else 'reasonix_db'})，"
        "Tushare/AkShare 调用将被完全绕过。",
        RuntimeWarning,
    )
    return patched


def is_enabled() -> bool:
    """是否启用 PostgreSQL 数据源。"""
    return os.getenv("AFGI_DATA_SOURCE", "").strip().lower() == "postgres"

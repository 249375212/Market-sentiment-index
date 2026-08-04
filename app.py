from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from market_fear_greed import (
    SENTIMENT_WEIGHTS,
    generate_enhanced_fear_greed_explanation,
    get_enhanced_cache_status,
    latest_enhanced_indicator_table,
    load_enhanced_cache,
    plan_enhanced_update_ranges,
    update_enhanced_afgi_cache,
)
from market_fear_greed.charts import (
    SENTIMENT_COMPONENTS,
    make_component_chart,
    make_gauge,
    make_history_chart,
)
from market_fear_greed.pg_adapter import (
    install_pg_adapter,
    is_enabled as _pg_data_source_enabled,
)
from market_fear_greed.tushare_client import get_default_token

if _pg_data_source_enabled():
    # 启用本地 PostgreSQL 数据源，完全绕过 Tushare/AkShare
    install_pg_adapter()


st.set_page_config(page_title="A股市场情绪指数", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp { background: #f6f7f9; color: #202938; }
    [data-testid="stHeader"] { background: rgba(246,247,249,.92); }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
    .block-container { max-width: 1380px; padding-top: 2rem; padding-bottom: 3rem; }
    .app-title { font-size: 2rem; font-weight: 760; color: #172033; margin-bottom: .25rem; }
    .app-subtitle { color: #667085; font-size: .96rem; margin-bottom: 1.4rem; }
    .analysis-note {
        background: #ffffff;
        border-left: 4px solid #d94a4a;
        padding: .9rem 1rem;
        color: #344054;
        margin: .4rem 0 1.15rem;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: .85rem 1rem;
        min-height: 116px;
    }
    [data-testid="stMetricLabel"] { color: #667085; }
    [data-testid="stMetricValue"] { color: #172033; }
    .stDownloadButton button, .stButton button { border-radius: 6px; }
    .stRadio [role="radiogroup"] { gap: .45rem; }
    .stRadio label[data-baseweb="radio"] {
        border: 1px solid #d0d5dd;
        background: #ffffff;
        border-radius: 6px;
        padding: .38rem .8rem;
        min-height: 36px;
    }
    .stRadio label[data-baseweb="radio"] > div:first-child { display: none; }
    .stRadio label[data-baseweb="radio"]:has(input:checked) {
        background: #172033;
        border-color: #172033;
        color: #ffffff;
    }
    .stRadio label[data-baseweb="radio"]:has(input:checked) > div:last-child {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


COMPONENT_METHOD = {
    "volatility_sentiment": ("波动率情绪", "50ETF 20日年化历史波动率，缺失时使用沪深300"),
    "volume_sentiment": ("成交情绪", "全市场成交活跃度"),
    "price_strength_sentiment": ("股价强度情绪", "全A 252日收盘新高股票占比"),
    "risk_appetite_sentiment": ("风险偏好情绪", "沪深300与国债ETF的20日相对收益"),
    "breadth_sentiment": ("市场广度情绪", "上涨占比、MA20、MA60"),
    "limit_sentiment": ("涨跌停情绪", "涨跌停比、炸板率、昨日涨停收益"),
    "profitability_sentiment": ("赚钱效应", "近5日上涨占比、60日新低占比"),
    "sector_sentiment": ("板块扩散情绪", "行业上涨占比"),
    "style_risk_appetite": ("风格风险偏好", "中证1000相对沪深300强弱"),
}

COMPONENT_STATUS_TEXT = {
    "ok": "实时",
    "stale": "延用",
    "partial": "部分",
    "missing": "待更新",
}

def _fmt(value, digits: int = 1, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.{digits}f}{suffix}"


def _filter_dates(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return out.loc[(out["date"] >= start_ts) & (out["date"] <= end_ts)].sort_values("date").reset_index(drop=True)


def _format_raw_table(latest: pd.Series) -> pd.DataFrame:
    table = latest_enhanced_indicator_table(latest)
    rows = []
    for _, row in table.iterrows():
        name = str(row.get("指标名称", ""))
        value = pd.to_numeric(pd.Series([row.get("原始值")]), errors="coerce").iloc[0]
        if any(key in name for key in ("率", "收益", "占比", "强弱")) and pd.notna(value):
            value_text = f"{value * 100:.2f}%"
        else:
            value_text = _fmt(value, digits=3)
        rows.append(
            {
                "原始指标": name,
                "当前值": value_text,
                "得分": _fmt(row.get("标准化得分")),
                "方向": str(row.get("指标方向", "")),
                "状态": str(row.get("数据状态", "")),
            }
        )
    return pd.DataFrame(rows)


def _method_table() -> pd.DataFrame:
    rows = []
    for key, _label in SENTIMENT_COMPONENTS:
        name, input_text = COMPONENT_METHOD[key]
        rows.append(
            {
                "分项": name,
                "权重": f"{SENTIMENT_WEIGHTS[key] * 100:.0f}%",
                "主要输入": input_text,
            }
        )
    return pd.DataFrame(rows)


def _component_data_summary(latest: pd.Series) -> tuple[int, list[str], list[str], list[str]]:
    usable = 0
    stale = []
    partial = []
    missing = []
    for key, label in SENTIMENT_COMPONENTS:
        status = str(latest.get(f"{key}_status", "missing")).lower()
        if status in {"ok", "stale", "partial"}:
            usable += 1
        if status == "stale":
            stale.append(label)
        elif status == "partial":
            partial.append(label)
        elif status != "ok":
            missing.append(label)
    return usable, stale, partial, missing


def _sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.subheader("数据设置")
        token = st.text_input(
            "Tushare Token",
            value=get_default_token(),
            type="password",
            help="建议通过环境变量 TUSHARE_TOKEN 或 .streamlit/secrets.toml 配置。",
        ).strip()
        overlay = st.selectbox("叠加指数", ("沪深300", "上证指数", "中证1000", "创业板指"))
        if _pg_data_source_enabled():
            st.success("已启用本地 PostgreSQL 数据源（reasonix_db），无需 Tushare Token。")
            st.caption("数据全部来自本地数据库，点击主界面的更新按钮即可计算。")
        else:
            st.info("请使用至少5000积分的 Tushare Token。")
            st.caption("页面启动只读取本地缓存，只有点击主界面的更新按钮后才会联网。")
    return token, overlay


def _main_date_range(today: date) -> tuple[str, date, date]:
    selected_range = st.radio(
        "数据范围",
        ("近1年", "近3年", "近5年", "自定义"),
        horizontal=True,
        label_visibility="collapsed",
        key="data_range",
    )
    if selected_range == "自定义":
        start_col, end_col = st.columns(2)
        with start_col:
            start = st.date_input("开始日期", value=today - timedelta(days=365), key="custom_start_date")
        with end_col:
            end = st.date_input("结束日期", value=today, key="custom_end_date")
    else:
        years = {"近1年": 1, "近3年": 3, "近5年": 5}[selected_range]
        start = (pd.Timestamp(today) - pd.DateOffset(years=years)).date()
        end = today
    return selected_range, start, end


def main() -> None:
    token, overlay = _sidebar()
    if _pg_data_source_enabled() and not token:
        # PostgreSQL 本地数据库模式不需要 Tushare Token，用占位符绕过空值校验
        token = "PG_LOCAL"
    st.markdown('<div class="app-title">A股市场情绪指数</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">九大分项刻画波动、成交、价格、风险偏好与 A 股赚钱效应</div>',
        unsafe_allow_html=True,
    )

    today = date.today()
    cache_status = get_enhanced_cache_status()
    first_run = bool(cache_status["needs_initialization"])
    if first_run:
        st.info(
            "第一次获取历史数据比较慢，请耐心等待。计算时会读取前置历史，"
            "图表仍只显示所选日期。"
        )
    else:
        latest_date = pd.Timestamp(cache_status["latest_trade_date"]).strftime("%Y-%m-%d")
        st.success(
            f"本地数据已保存，当前截至 {latest_date}。更新时会自动补齐所选日期和最近交易日。"
        )

    range_col, action_button = st.columns((3.2, 1), gap="large")
    with range_col:
        _selected_range, start, end = _main_date_range(today)
    with action_button:
        button_label = "获取历史数据并开始计算" if first_run else "补齐并更新数据"
        update_requested = st.button(button_label, type="primary", use_container_width=True)

    if start > end:
        st.error("开始日期不能晚于结束日期。")
        return

    if update_requested and not token:
        st.warning("请先在左侧填写 Tushare Token。该 Token 至少需要 5000 积分。")

    try:
        if update_requested and token:
            progress_text = st.empty()
            progress_bar = st.progress(0.0)
            update_ranges = plan_enhanced_update_ranges(
                cache_status,
                pd.Timestamp(start).strftime("%Y%m%d"),
                pd.Timestamp(end).strftime("%Y%m%d"),
                pd.Timestamp(today).strftime("%Y%m%d"),
            )
            phase_state = {"index": 0, "count": max(len(update_ranges), 1)}

            def show_progress(done: int, total: int, message: str) -> None:
                total = max(int(total or 1), 1)
                done = min(max(int(done or 0), 0), total)
                ratio = done / total
                if "获取历史行情" in message:
                    visual_ratio = 0.05 + ratio * 0.65
                elif "回填" in message or "计算" in message:
                    visual_ratio = 0.70 + ratio * 0.29
                elif "完成" in message or "没有需要补算" in message:
                    visual_ratio = 1.0
                else:
                    visual_ratio = 0.03
                overall_ratio = (phase_state["index"] + visual_ratio) / phase_state["count"]
                progress_bar.progress(min(overall_ratio, 1.0))
                phase_text = f"阶段 {phase_state['index'] + 1}/{phase_state['count']} · " if phase_state["count"] > 1 else ""
                progress_text.caption(f"{phase_text}{message}（{done}/{total}）")

            if first_run:
                st.warning("第一次获取历史数据比较慢，请耐心等待。请勿关闭当前页面。")
            spinner_text = "正在建立本地历史数据并计算九大分项..." if first_run else "正在补齐所选日期和最近交易日数据..."
            with st.spinner(spinner_text):
                for phase_index, (update_start, update_end) in enumerate(update_ranges):
                    phase_state["index"] = phase_index
                    update_enhanced_afgi_cache(
                        token,
                        update_start,
                        update_end,
                        force_update=False,
                        max_update_days=None,
                        progress_callback=show_progress,
                    )
            progress_bar.progress(1.0)
            progress_text.caption("本地数据已更新完成（100%）")
            st.success("所选日期和最近交易日数据已补齐并保存到本地。")
            frame = _filter_dates(load_enhanced_cache(), start, end)
        else:
            with st.spinner("正在读取本地九分项缓存..."):
                frame = _filter_dates(load_enhanced_cache(), start, end)
    except Exception as exc:
        st.error(f"数据更新失败：{exc}")
        return

    if frame.empty:
        st.warning("本地暂无九分项数据。填写 Token 后点击“获取历史数据并开始计算”。")
        return

    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else latest
    delta = float(latest.get("afgi_enhanced", 50.0)) - float(previous.get("afgi_enhanced", 50.0))
    usable_count, stale_components, partial_components, missing_components = _component_data_summary(latest)

    metric_rows = (st.columns(3), st.columns(3))
    metrics = (*metric_rows[0], *metric_rows[1])
    metrics[0].metric("市场情绪指数", _fmt(latest.get("afgi_enhanced")), _fmt(delta))
    metrics[1].metric("当前状态", str(latest.get("afgi_enhanced_state", "--")))
    metrics[2].metric("5日均值", _fmt(latest.get("afgi_enhanced_ma5")))
    metrics[3].metric("20日均值", _fmt(latest.get("afgi_enhanced_ma20")))
    metrics[4].metric("有效分项", f"{usable_count} / 9")
    metrics[5].metric("最新交易日", pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"))

    data_notes = []
    if stale_components:
        data_notes.append("沿用最近有效值：" + "、".join(stale_components))
    if partial_components:
        data_notes.append("部分底层数据可用：" + "、".join(partial_components))
    if missing_components:
        data_notes.append("暂用 50 分中性值：" + "、".join(missing_components))
    if data_notes:
        st.warning("数据状态：" + "；".join(data_notes) + "。")

    st.markdown(
        f'<div class="analysis-note">{generate_enhanced_fear_greed_explanation(latest)}</div>',
        unsafe_allow_html=True,
    )

    gauge_col, component_col = st.columns((0.9, 1.7), gap="large")
    with gauge_col:
        st.plotly_chart(make_gauge(latest), use_container_width=True, config={"displaylogo": False})
    with component_col:
        st.plotly_chart(make_component_chart(latest), use_container_width=True, config={"displaylogo": False})

    st.plotly_chart(
        make_history_chart(frame, overlay),
        use_container_width=True,
        config={"displaylogo": False},
    )

    st.subheader("十项底层原始指标")
    st.dataframe(_format_raw_table(latest), use_container_width=True, hide_index=True)

    export_cols = (
        "date",
        "trade_date",
        *SENTIMENT_WEIGHTS.keys(),
        "afgi_enhanced",
        "afgi_enhanced_state",
        "afgi_enhanced_change",
        "afgi_enhanced_ma5",
        "afgi_enhanced_ma20",
    )
    detail = frame[[col for col in export_cols if col in frame.columns]].copy()
    with st.expander("查看指数明细"):
        st.dataframe(detail.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
        st.download_button(
            "下载 CSV",
            detail.to_csv(index=False).encode("utf-8-sig"),
            file_name="a_share_market_sentiment_9_factor.csv",
            mime="text/csv",
        )

    with st.expander("九分项计算方法"):
        st.dataframe(_method_table(), use_container_width=True, hide_index=True)
        st.caption("各分项使用 252 个交易日滚动历史分位数标准化；数据缺失或样本不足时使用 50 分中性值。")

    st.caption("仅用于量化研究与市场观察，不构成投资建议。")


if __name__ == "__main__":
    main()

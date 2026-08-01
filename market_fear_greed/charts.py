from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SENTIMENT_RED = "#d94a4a"
INDEX_BLUE = "#2563eb"
GRID_COLOR = "rgba(100, 116, 139, 0.18)"

BANDS = (
    (0, 20, "极度恐惧", "rgba(15, 118, 110, 0.09)"),
    (20, 40, "恐惧", "rgba(14, 116, 144, 0.08)"),
    (40, 60, "中性", "rgba(217, 119, 6, 0.08)"),
    (60, 80, "贪婪", "rgba(234, 88, 12, 0.09)"),
    (80, 100, "极度贪婪", "rgba(220, 38, 38, 0.10)"),
)

SENTIMENT_COMPONENTS = (
    ("volatility_sentiment", "波动率情绪"),
    ("volume_sentiment", "成交情绪"),
    ("price_strength_sentiment", "股价强度情绪"),
    ("risk_appetite_sentiment", "风险偏好情绪"),
    ("breadth_sentiment", "市场广度情绪"),
    ("limit_sentiment", "涨跌停情绪"),
    ("profitability_sentiment", "赚钱效应"),
    ("sector_sentiment", "板块扩散情绪"),
    ("style_risk_appetite", "风格风险偏好"),
)

INDEX_DISPLAY = {
    "沪深300": "hs300_close",
    "上证指数": "sh_close",
    "中证1000": "zz1000_close",
    "创业板指": "cyb_close",
}


def _base_layout(fig: go.Figure, height: int = 410) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=54, b=34),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Microsoft YaHei, Arial, sans-serif", color="#253046"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def make_history_chart(df: pd.DataFrame, index_label: str = "沪深300") -> go.Figure:
    chart_df = _frame(df)
    index_col = INDEX_DISPLAY.get(index_label, "hs300_close")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for low, high, _label, color in BANDS:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, line_width=0, layer="below")
    fig.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["afgi_enhanced"],
            mode="lines",
            name="市场情绪指数",
            line=dict(color=SENTIMENT_RED, width=2.7),
            hovertemplate="%{x|%Y-%m-%d}<br>指数：%{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )
    if "afgi_enhanced_ma20" in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"],
                y=chart_df["afgi_enhanced_ma20"],
                mode="lines",
                name="20日均值",
                line=dict(color="#64748b", width=1.6, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>20日均值：%{y:.1f}<extra></extra>",
            ),
            secondary_y=False,
        )
    if index_col in chart_df.columns and chart_df[index_col].notna().any():
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"],
                y=pd.to_numeric(chart_df[index_col], errors="coerce"),
                mode="lines",
                name=index_label,
                line=dict(color=INDEX_BLUE, width=2.1),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{index_label}：%{{y:.2f}}<extra></extra>",
            ),
            secondary_y=True,
        )
    for level in (20, 40, 60, 80):
        fig.add_hline(y=level, line_dash="dot", line_color=GRID_COLOR)
    fig.update_layout(title=f"市场情绪指数与{index_label}")
    fig.update_yaxes(title_text="情绪指数", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text=index_label, secondary_y=True, showgrid=False)
    return _base_layout(fig, height=470)


def make_overlay_chart(df: pd.DataFrame, index_label: str) -> go.Figure:
    """Backward-compatible alias for the merged history chart."""
    return make_history_chart(df, index_label)


def make_component_chart(latest: pd.Series) -> go.Figure:
    status_labels = {
        "ok": "",
        "stale": " · 延用",
        "partial": " · 部分",
        "missing": " · 待更新",
    }
    statuses = [
        str(latest.get(f"{key}_status", "missing")).lower()
        for key, _ in SENTIMENT_COMPONENTS
    ]
    labels = [
        f"{label}{status_labels.get(status, ' · 待更新')}"
        for (_, label), status in zip(SENTIMENT_COMPONENTS, statuses)
    ]
    values = [
        float(latest.get(key, np.nan)) if pd.notna(latest.get(key, np.nan)) else np.nan
        for key, _ in SENTIMENT_COMPONENTS
    ]
    colors = (
        "#0f766e",
        "#0891b2",
        INDEX_BLUE,
        "#7c3aed",
        "#16a34a",
        SENTIMENT_RED,
        "#d97706",
        "#0284c7",
        "#9333ea",
    )
    display_colors = [
        (
            "#64748b"
            if status == "stale"
            else "#d97706"
            if status == "partial"
            else "#cbd5e1"
            if status != "ok"
            else color
        )
        for color, status in zip(colors, statuses)
    ]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=display_colors,
            text=[f"{value:.1f}" if pd.notna(value) else "--" for value in values],
            textposition="outside",
            customdata=statuses,
            hovertemplate="%{y}<br>得分：%{x:.1f}<br>状态：%{customdata}<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.add_vline(x=50, line_dash="dot", line_color=GRID_COLOR)
    fig.update_layout(title="九大情绪分项")
    fig.update_xaxes(title_text="得分", range=[0, 108])
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12))
    fig = _base_layout(fig, height=410)
    fig.update_layout(margin=dict(l=142, r=44, t=54, b=42), hovermode="closest")
    return fig


def make_gauge(latest: pd.Series) -> go.Figure:
    score = pd.to_numeric(pd.Series([latest.get("afgi_enhanced")]), errors="coerce").iloc[0]
    score = float(score) if pd.notna(score) else 50.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0.08, 0.92], "y": [0.04, 0.96]},
            number={"suffix": " / 100", "font": {"size": 22}},
            title={"text": str(latest.get("afgi_enhanced_state", "--"))},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 20, 40, 60, 80, 100],
                    "ticktext": ["0", "20", "40", "60", "80", "100"],
                    "tickfont": {"size": 11},
                },
                "bar": {"color": "#172033", "thickness": 0.2},
                "steps": [
                    {"range": [low, high], "color": color}
                    for low, high, _label, color in BANDS
                ],
                "threshold": {
                    "line": {"color": SENTIMENT_RED, "width": 4},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(title="情绪仪表盘")
    fig = _base_layout(fig, height=410)
    fig.update_layout(margin=dict(l=26, r=26, t=54, b=32))
    return fig

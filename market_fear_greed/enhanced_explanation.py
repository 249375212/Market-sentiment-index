from __future__ import annotations

import numpy as np
import pandas as pd


def _num(row: pd.Series, key: str, default: float = 50.0) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if np.isfinite(value) else default


def generate_enhanced_fear_greed_explanation(latest: pd.Series) -> str:
    """Generate a concise explanation for enhanced AFGI and its A-share sentiment components."""
    afgi = _num(latest, "afgi_enhanced")
    state = str(latest.get("afgi_enhanced_state", "中性"))
    base = f"当前市场情绪指数为 {afgi:.1f}，处于{state}区间。"

    notes = []
    breadth = _num(latest, "breadth_sentiment")
    limit = _num(latest, "limit_sentiment")
    profitability = _num(latest, "profitability_sentiment")
    sector = _num(latest, "sector_sentiment")
    style = _num(latest, "style_risk_appetite")
    open_board_rate = _num(latest, "open_board_rate", default=np.nan)
    new_low_ratio = _num(latest, "new_low_60d_ratio", default=np.nan)

    if breadth >= 65:
        notes.append("市场广度较好，赚钱效应正在扩散。")
    elif breadth <= 35:
        notes.append("市场广度偏弱，上涨股票覆盖面不足。")

    if limit >= 65:
        notes.append("涨跌停情绪较强，短线投机资金较活跃。")
    elif limit <= 35:
        notes.append("涨跌停情绪偏弱，短线接力意愿不足。")

    if np.isfinite(open_board_rate) and open_board_rate >= 0.35:
        notes.append("炸板率偏高，短线分歧开始加大。")

    if profitability >= 65:
        notes.append("近5日赚钱效应较好。")
    elif profitability <= 35:
        notes.append("近5日赚钱效应偏弱。")

    if np.isfinite(new_low_ratio) and new_low_ratio >= 0.12:
        notes.append("创60日新低股票增多，市场恐慌情绪上升。")

    if sector >= 65:
        notes.append("行业上涨范围较广，行情扩散较健康。")
    elif sector <= 35:
        notes.append("行业上涨范围较窄，行情仍偏局部。")

    if style >= 65:
        notes.append("小盘股强于大盘股，风险偏好较强。")
    elif style <= 35:
        notes.append("大盘股相对占优，市场可能偏防御或权重行情。")

    if not notes:
        notes.append("各分项整体接近中性，建议继续观察赚钱效应和市场广度变化。")
    return base + "".join(notes)

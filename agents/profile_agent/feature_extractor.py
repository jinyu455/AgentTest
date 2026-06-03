from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

# 活跃时段分桶标签
BUCKET_LABELS = ["凌晨", "上午", "下午", "晚上"]


def _hour_to_bucket(hour: int) -> str:
    """将 24 小时制的小时数映射到四个时段桶。"""
    if hour < 6:
        return "凌晨"    # 00:00 - 05:59
    if hour < 12:
        return "上午"    # 06:00 - 11:59
    if hour < 18:
        return "下午"    # 12:00 - 17:59
    return "晚上"        # 18:00 - 23:59


def _compute_intensity_trend(intensities: list[int]) -> str:
    """通过简单线性回归判断情绪强度的趋势方向。

    将记录按时间排序后，对强度序列拟合一条直线。
    斜率 > 0.5 视为上升趋势，< -0.5 视为下降，否则平稳。

    参数:
        intensities: 按时间排序后的强度值列表

    返回:
        "上升" | "下降" | "平稳"
    """
    n = len(intensities)
    # 数据太少时无法判断趋势
    if n < 3:
        return "平稳"

    # 计算 x（时间索引）和 y（强度值）的均值
    mean_x = (n - 1) / 2.0
    mean_y = sum(intensities) / n

    # 最小二乘法求斜率：slope = Σ(xi - mean_x)(yi - mean_y) / Σ(xi - mean_x)²
    numerator = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(intensities))
    denominator = sum((i - mean_x) ** 2 for i in range(n))

    if denominator == 0:
        return "平稳"

    slope = numerator / denominator

    # 阈值 0.5 的含义：强度范围 0-100，记录数作为 x 轴，
    # 斜率 0.5 表示每条记录平均变化 0.5 个强度单位
    if slope > 0.5:
        return "上升"
    if slope < -0.5:
        return "下降"
    return "平稳"


def extract_features(records: list[dict[str, Any]]) -> dict[str, Any]:
    """从情绪记录列表中提取统计特征。

    纯 Python 计算，不依赖任何 LLM 调用。
    每条 record 需要包含: final_emotion, final_intensity, final_confidence,
    is_sarcasm, is_mixed, created_at。

    返回包含以下字段的字典:
        total_records      — 总记录数
        emotion_distribution — 各情绪占比 {"焦虑": 0.4, ...}
        avg_intensity      — 平均情绪强度
        avg_confidence     — 平均置信度
        sarcasm_rate       — 反讽比例
        mixed_rate         — 混合情绪比例
        dominant_emotion   — 最高频情绪
        intensity_trend    — 强度趋势方向
        activity_pattern   — 各时段活跃次数
    """
    # 空记录时返回默认值
    if not records:
        return {
            "total_records": 0,
            "emotion_distribution": {},
            "avg_intensity": 0.0,
            "avg_confidence": 0.0,
            "sarcasm_rate": 0.0,
            "mixed_rate": 0.0,
            "dominant_emotion": "中性",
            "intensity_trend": "平稳",
            "activity_pattern": {b: 0 for b in BUCKET_LABELS},
        }

    total = len(records)

    # --- 1. 情绪分布：统计每种情绪出现的次数和占比 ---
    counts: Counter[str] = Counter()
    for r in records:
        label = str(r.get("final_emotion", "")).strip()
        if label:
            counts[label] += 1
    distribution = {label: round(c / total, 4) for label, c in counts.items()}

    # --- 2. 聚合统计：强度、置信度的均值，反讽和混合情绪的比例 ---
    intensities = [r["final_intensity"] for r in records if r.get("final_intensity") is not None]
    confidences = [r["final_confidence"] for r in records if r.get("final_confidence") is not None]

    avg_intensity = round(sum(intensities) / len(intensities), 2) if intensities else 0.0
    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    sarcasm_rate = round(sum(1 for r in records if r.get("is_sarcasm")) / total, 4)
    mixed_rate = round(sum(1 for r in records if r.get("is_mixed")) / total, 4)

    # 取出现次数最多的情绪作为主导情绪
    dominant_emotion = counts.most_common(1)[0][0] if counts else "中性"

    # --- 3. 强度趋势：按时间排序后做线性回归 ---
    sorted_records = sorted(records, key=lambda r: str(r.get("created_at", "")))
    sorted_intensities = [
        r["final_intensity"] for r in sorted_records if r.get("final_intensity") is not None
    ]
    intensity_trend = _compute_intensity_trend(sorted_intensities)

    # --- 4. 活跃时段：解析 created_at 的小时，分桶统计 ---
    activity: dict[str, int] = {b: 0 for b in BUCKET_LABELS}
    for r in records:
        created = str(r.get("created_at", ""))
        try:
            # 支持 "Z" 后缀和标准 ISO 格式
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            bucket = _hour_to_bucket(dt.hour)
            activity[bucket] += 1
        except (ValueError, TypeError):
            # 无法解析的时间戳跳过
            continue

    return {
        "total_records": total,
        "emotion_distribution": distribution,
        "avg_intensity": avg_intensity,
        "avg_confidence": avg_confidence,
        "sarcasm_rate": sarcasm_rate,
        "mixed_rate": mixed_rate,
        "dominant_emotion": dominant_emotion,
        "intensity_trend": intensity_trend,
        "activity_pattern": activity,
    }

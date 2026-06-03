from __future__ import annotations

from typing import Any

from base.schemas import EMOTION_LABELS


def build_visualization_data(
    features: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """生成三种图表的结构化数据，供前端 ECharts / Chart.js 渲染。

    参数:
        features: extract_features() 返回的统计特征字典
        records:  原始情绪记录列表（时间线和直方图需要逐条数据）

    返回:
        {
            "radar_chart": ...               # 情绪分布雷达图
            "timeline": ...                  # 情绪时间线
            "intensity_distribution": ...    # 强度直方图
        }
    """
    return {
        "radar_chart": _build_radar(features),
        "timeline": _build_timeline(records),
        "intensity_distribution": _build_histogram(records),
    }


def _build_radar(features: dict[str, Any]) -> dict[str, Any]:
    """构建情绪分布雷达图数据。

    以 9 种基本情绪为轴，每轴的值为该情绪的出现占比（0-1）。
    前端可用 ECharts radar 或 Chart.js radar chart 直接渲染。
    """
    distribution = features.get("emotion_distribution", {})
    # 排序确保每次返回的 labels 顺序一致
    sorted_labels = sorted(EMOTION_LABELS)
    return {
        "type": "radar",
        "labels": sorted_labels,
        "values": [round(distribution.get(label, 0.0), 4) for label in sorted_labels],
        "title": "情绪分布雷达图",
    }


def _build_timeline(records: list[dict[str, Any]]) -> dict[str, Any]:
    """构建情绪时间线数据。

    按 created_at 升序排列，每条数据点包含时间、情绪标签和强度。
    前端可用折线图或散点图展示情绪随时间的变化。
    """
    sorted_records = sorted(records, key=lambda r: str(r.get("created_at", "")))
    data_points = [
        {
            "created_at": r.get("created_at", ""),
            "emotion": r.get("final_emotion", ""),
            "intensity": r.get("final_intensity", 0),
        }
        for r in sorted_records
    ]
    return {
        "type": "timeline",
        "data_points": data_points,
        "title": "情绪时间线",
    }


def _build_histogram(records: list[dict[str, Any]]) -> dict[str, Any]:
    """构建情绪强度分布直方图数据。

    将 0-100 的强度值分为 5 个桶：0-20, 21-40, 41-60, 61-80, 81-100。
    前端可用柱状图展示用户情绪强度的整体分布。
    """
    bucket_labels = ["0-20", "21-40", "41-60", "61-80", "81-100"]
    buckets = [0] * 5

    for r in records:
        intensity = r.get("final_intensity")
        if intensity is None:
            continue
        val = int(intensity)
        # 按区间分桶
        if val <= 20:
            buckets[0] += 1
        elif val <= 40:
            buckets[1] += 1
        elif val <= 60:
            buckets[2] += 1
        elif val <= 80:
            buckets[3] += 1
        else:
            buckets[4] += 1

    return {
        "type": "histogram",
        "labels": bucket_labels,
        "values": buckets,
        "title": "情绪强度分布",
    }

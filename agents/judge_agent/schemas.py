"""Judge Agent 的数据模型定义。

包含最终裁决所需的输入数据结构 JudgeInput 和输出结果数据结构 JudgeResult。
Judge Agent 是情绪分析流水线的最后一环，负责整合上游各 Agent 的结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class JudgeInput:
    """Judge Agent 的输入数据。

    汇聚了上游各 Agent（Router、Emotion、Sarcasm、Mix）的分析结果，
    供 Judge Agent 进行最终裁决。
    """

    router_result: dict[str, Any]                # 路由 Agent 的结果（包含 sample_type 等）
    emotion_result: dict[str, Any]               # 情感 Agent 的结果（包含 emotion、intensity、confidence 等）
    sarcasm_result: dict[str, Any] | None = None  # 反讽 Agent 的结果（仅当路由判断需要反讽检测时存在）
    mix_result: dict[str, Any] | None = None      # 混合情绪 Agent 的结果（仅当路由判断需要混合检测时存在）
    text: str | None = None                       # 原始文本（可选，用于辅助裁决）


@dataclass(slots=True)
class JudgeResult:
    """Judge Agent 的最终裁决结果。

    包含经过整合后的最终情绪判定，是整个情绪分析流水线的最终输出。
    """

    final_emotion: str                # 最终判定的主情绪标签
    secondary_emotion: str | None     # 次情绪标签，无次情绪时为 None
    final_intensity: int              # 最终情绪强度，范围 0~100
    final_confidence: float           # 最终置信度，范围 0~1
    is_sarcasm: bool                  # 是否包含反讽成分
    is_mixed: bool                    # 是否包含混合情绪
    reason: str                       # 裁决理由的简短说明

    def to_dict(self) -> dict[str, Any]:
        """将裁决结果转换为字典格式，便于序列化传输。"""
        return {
            "final_emotion": self.final_emotion,
            "secondary_emotion": self.secondary_emotion,
            "final_intensity": self.final_intensity,
            "final_confidence": self.final_confidence,
            "is_sarcasm": self.is_sarcasm,
            "is_mixed": self.is_mixed,
            "reason": self.reason,
        }

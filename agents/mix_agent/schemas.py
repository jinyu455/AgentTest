"""Mix Agent 的数据模型定义。

包含混合情绪分析的输入数据结构 MixInput 和输出结果数据结构 MixResult。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from base.schemas import BaseTextInput

# MixInput 与 BaseTextInput 字段完全一致，直接复用
MixInput = BaseTextInput


@dataclass(slots=True)
class MixResult:
    """混合情绪分析的输出结果。

    当文本被判定为包含混合情绪时，返回主情绪、次情绪及其比例等信息。
    """

    is_mixed: bool                                # 是否为混合情绪
    primary_emotion: str                          # 主情绪标签（如 "疲惫"、"开心"）
    secondary_emotion: str                        # 次情绪标签
    mix_ratio: dict[str, float] = field(default_factory=dict)  # 各情绪的比例，键为情绪标签，值为 0~1 的浮点数
    adjusted_intensity: int = 0                    # 混合情绪场景下的主情绪强度，范围 0~100
    confidence: float = 0.0                        # 模型置信度，范围 0~1
    reason: str = ""                               # 判定理由的简短说明

    def to_dict(self) -> dict[str, Any]:
        """将结果转换为字典格式，便于序列化传输。"""
        return {
            "is_mixed": self.is_mixed,
            "primary_emotion": self.primary_emotion,
            "secondary_emotion": self.secondary_emotion,
            "mix_ratio": self.mix_ratio,
            "adjusted_intensity": self.adjusted_intensity,
            "confidence": self.confidence,
            "reason": self.reason,
        }

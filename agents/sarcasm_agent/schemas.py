"""反讽代理的输入输出数据模型。

定义了 SarcasmAgent 接收的输入结构（SarcasmInput）和
输出的反讽检测结果结构（SarcasmResult），用于判断文本是否
包含反讽表达并给出修正后的真实情绪。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from base.schemas import BaseTextInput

# SarcasmInput 与 BaseTextInput 字段完全一致，直接复用
SarcasmInput = BaseTextInput


@dataclass(slots=True)
class SarcasmResult:
    """反讽代理的输出数据结构。

    包含大模型对输入文本进行反讽检测后的完整结果，
    涵盖反讽判断、表层情绪、真实情绪、修正强度和置信度。

    Attributes:
        is_sarcasm: 是否检测到反讽表达。
        surface_emotion: 句面情绪，按文本表面词义判断的情绪标签。
        true_emotion: 真实情绪，结合语境修正后的情绪标签。
        revised_intensity: 修正后的情绪强度，0-100 的整数。
        confidence: 模型对判断结果的置信度，0-1 的浮点数。
        reason: 简短的判断理由说明。
    """
    is_sarcasm: bool
    surface_emotion: str
    true_emotion: str
    revised_intensity: int
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """将结果转换为普通字典，便于序列化或日志记录。"""
        return {
            "is_sarcasm": self.is_sarcasm,
            "surface_emotion": self.surface_emotion,
            "true_emotion": self.true_emotion,
            "revised_intensity": self.revised_intensity,
            "confidence": self.confidence,
            "reason": self.reason,
        }

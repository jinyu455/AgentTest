"""情绪代理的输入输出数据模型。

定义了 EmotionAgent 接收的输入结构（EmotionInput）和
输出的情绪识别结果结构（EmotionResult），包含表层情绪判断
所需的各类语言特征字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from base.schemas import BaseTextInput

# EmotionInput 与 BaseTextInput 字段完全一致，直接复用
EmotionInput = BaseTextInput


@dataclass(slots=True)
class EmotionResult:
    """情绪代理的输出数据结构。

    包含大模型对输入文本进行表层情绪分析后的完整结果，
    涵盖分词、情绪词提取、情绪分类、强度和置信度等信息。

    Attributes:
        tokens: 分词或关键短语切分结果，保留语义片段。
        emotion_words: 直接表达情绪或具有明显情绪方向的词语。
        degree_words: 程度修饰词（如"很"、"特别"、"稍微"等）。
        negation_words: 否定词（如"不"、"没有"、"别"等）。
        contrast_words: 转折词（如"但是"、"不过"、"然而"等）。
        emotion: 主情绪标签，取 9 类预定义标签之一。
        intensity: 情绪强度，0-100 的整数。
        confidence: 模型对判断结果的置信度，0-1 的浮点数。
        reason: 简短的判断理由说明。
    """
    tokens: list[str] = field(default_factory=list)
    emotion_words: list[str] = field(default_factory=list)
    degree_words: list[str] = field(default_factory=list)
    negation_words: list[str] = field(default_factory=list)
    contrast_words: list[str] = field(default_factory=list)
    emotion: str = "中性"
    intensity: int = 0
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """将结果转换为普通字典，便于序列化或日志记录。"""
        return {
            "tokens": self.tokens,
            "emotion_words": self.emotion_words,
            "degree_words": self.degree_words,
            "negation_words": self.negation_words,
            "contrast_words": self.contrast_words,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "reason": self.reason,
        }

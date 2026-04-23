from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EmotionInput:
    id: str
    user_id: str
    text: str
    source: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmotionResult:
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

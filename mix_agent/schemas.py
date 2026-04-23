from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MixInput:
    id: str
    user_id: str
    text: str
    source: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MixResult:
    is_mixed: bool
    primary_emotion: str
    secondary_emotion: str
    mix_ratio: dict[str, float] = field(default_factory=dict)
    revised_intensity: int = 0
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_mixed": self.is_mixed,
            "primary_emotion": self.primary_emotion,
            "secondary_emotion": self.secondary_emotion,
            "mix_ratio": self.mix_ratio,
            "revised_intensity": self.revised_intensity,
            "confidence": self.confidence,
            "reason": self.reason,
        }

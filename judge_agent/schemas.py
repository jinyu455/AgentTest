from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class JudgeInput:
    router_result: dict[str, Any]
    emotion_result: dict[str, Any]
    sarcasm_result: dict[str, Any] | None = None
    mix_result: dict[str, Any] | None = None


@dataclass(slots=True)
class JudgeResult:
    final_emotion: str
    secondary_emotion: str | None
    final_intensity: int
    final_confidence: float
    is_sarcasm: bool
    is_mixed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_emotion": self.final_emotion,
            "secondary_emotion": self.secondary_emotion,
            "final_intensity": self.final_intensity,
            "final_confidence": self.final_confidence,
            "is_sarcasm": self.is_sarcasm,
            "is_mixed": self.is_mixed,
            "reason": self.reason,
        }

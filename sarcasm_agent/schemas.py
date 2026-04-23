from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SarcasmInput:
    id: str
    user_id: str
    text: str
    source: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SarcasmResult:
    is_sarcasm: bool
    surface_emotion: str
    true_emotion: str
    revised_intensity: int
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_sarcasm": self.is_sarcasm,
            "surface_emotion": self.surface_emotion,
            "true_emotion": self.true_emotion,
            "revised_intensity": self.revised_intensity,
            "confidence": self.confidence,
            "reason": self.reason,
        }

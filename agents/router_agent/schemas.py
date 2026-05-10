from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RouterInput:
    id: str
    user_id: str
    text: str
    source: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouterResult:
    sample_type: str
    need_sarcasm_check: bool
    need_mix_check: bool
    routing_reason: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_type": self.sample_type,
            "need_sarcasm_check": self.need_sarcasm_check,
            "need_mix_check": self.need_mix_check,
            "routing_reason": self.routing_reason,
            "evidence": self.evidence,
        }

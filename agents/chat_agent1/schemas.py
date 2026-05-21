# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import Any


# @dataclass(slots=True)
# class ChatMessage:
#     role: str
#     content: str


# @dataclass(slots=True)
# class ChatInput:
#     text: str
#     user_id: str | None = None
#     conversation_id: str | None = None
#     judge_result: dict[str, Any] | None = None
#     history: list[dict[str, Any]] = field(default_factory=list)
#     metadata: dict[str, Any] = field(default_factory=dict)


# @dataclass(slots=True)
# class ChatResult:
#     reply: str
#     tone: str
#     risk_hint: str
#     suggested_actions: list[str] = field(default_factory=list)
#     reason: str = ""

#     def to_dict(self) -> dict[str, Any]:
#         return {
#             "reply": self.reply,
#             "tone": self.tone,
#             "risk_hint": self.risk_hint,
#             "suggested_actions": self.suggested_actions,
#             "reason": self.reason,
#         }

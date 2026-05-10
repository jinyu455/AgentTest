from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

from .schemas import ChatInput, ChatResult


TONE_LABELS = {"supportive", "calm", "encouraging", "reflective", "crisis_support"}
RISK_HINTS = {"none", "possible_crisis"}

SYSTEM_PROMPT = """你是 EmoAgent 中的 Chat Agent，负责生成情绪聊天助手的回复。
你的目标不是给用户贴标签，而是基于情绪分析结果，用温和、尊重、具体的方式回应用户。

回复原则：
- 先回应用户真实感受，再给出轻量建议。
- 不要说教，不要夸大判断，不要假装自己能替代专业帮助。
- 如果 judge_result 中存在较高 safety_score，或文本出现自伤、自杀、伤害他人、极端危机倾向，risk_hint 必须为 possible_crisis，tone 使用 crisis_support。
- 危机场景下，不提供危险方法，不鼓励危险行为，应建议联系可信任的人、当地紧急服务或专业支持。
- 普通场景下，回复控制在 1 到 3 句话，像一个稳定的聊天助手，而不是分析报告。

只返回 JSON，不要输出 markdown，不要增加额外字段。

输出格式：
{
  "reply": "给用户的中文回复",
  "tone": "supportive | calm | encouraging | reflective | crisis_support",
  "risk_hint": "none | possible_crisis",
  "suggested_actions": ["可选的简短行动建议"],
  "reason": "简短说明生成依据"
}
"""


class ChatLLMClient(Protocol):
    def generate(self, payload: ChatInput) -> dict[str, Any]:
        """Send payload to an LLM and return the parsed JSON result."""


class ChatAgent:
    """LLM-based emotional chat response agent."""

    def __init__(self, client: ChatLLMClient) -> None:
        self.client = client

    def chat(self, payload: ChatInput | dict[str, Any]) -> ChatResult:
        item = payload if isinstance(payload, ChatInput) else ChatInput(**payload)
        raw_result = self.client.generate(item)
        return self._build_result(raw_result)

    def chat_dict(self, payload: ChatInput | dict[str, Any]) -> dict[str, Any]:
        return self.chat(payload).to_dict()

    def _build_result(self, raw_result: dict[str, Any]) -> ChatResult:
        reply = str(raw_result.get("reply", "")).strip()
        if not reply:
            raise ValueError("Invalid reply from LLM: empty")

        tone = str(raw_result.get("tone", "")).strip()
        if tone not in TONE_LABELS:
            raise ValueError(f"Invalid tone from LLM: {tone!r}")

        risk_hint = str(raw_result.get("risk_hint", "")).strip()
        if risk_hint not in RISK_HINTS:
            raise ValueError(f"Invalid risk_hint from LLM: {risk_hint!r}")

        return ChatResult(
            reply=reply,
            tone=tone,
            risk_hint=risk_hint,
            suggested_actions=self._coerce_str_list(raw_result.get("suggested_actions", []), "suggested_actions"),
            reason=str(raw_result.get("reason", "")).strip(),
        )

    def _coerce_str_list(self, value: Any, field_name: str) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"Invalid {field_name} from LLM: expected list")
        return [str(item).strip() for item in value if str(item).strip()]
    
    def build_messages(self, payload: ChatInput | dict[str, Any]) -> list[dict[str, str]]:
        item = payload if isinstance(payload, ChatInput) else ChatInput(**payload)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(item)},
        ]

    def _build_user_prompt(self, payload: ChatInput) -> str:
        return (
            "请基于下面的用户文本、情绪分析结果和对话历史，生成情绪聊天助手回复。\n\n"
            f"{json.dumps(asdict(payload), ensure_ascii=False, indent=2)}"
        )


from __future__ import annotations

import json
from typing import Any, Protocol

from .schemas import ChatInput, ChatResult


TONE_LABELS = {"supportive", "calm", "encouraging", "reflective", "crisis_support"}
RISK_HINTS = {"none", "possible_crisis"}

SYSTEM_PROMPT = """你是 EmoAgent 中的 Chat Agent，负责生成情绪聊天助手的回复。
你的目标不是给用户贴标签，也不是做机械的心理咨询式追问，而是基于情绪分析结果和最近对话历史，用温和、尊重、具体、有用的方式回应用户。

回复原则：
- 优先理解最近对话历史，尤其是最近 3 轮。如果当前用户说“能给我一些建议吗”“那怎么办”“这个怎么处理”“我该咋办”等，必须结合上文直接回答。
- 当用户请求建议、办法、下一步、安慰或陪伴时，先给出可执行帮助，不要先追问。信息不足时，也要先给通用可行建议，再在结尾只问 1 个必要问题。
- 回复要抓住用户说过的具体事实，不要只说“听起来你很难受”“能具体说说吗”这类空泛模板。
- 可以先简短回应感受，但不要停在共情上；普通场景下至少给出 2 个具体做法或下一步。
- 建议要轻量、具体、可执行，例如“先列清单”“先确认对方要求”“把任务拆成 3 块”“休息 10 分钟再处理最小一项”等。
- 不要说教，不要夸大判断，不要假装自己能替代专业帮助。
- 如果 judge_result 中存在较高 safety_score，或文本出现自伤、自杀、伤害他人、极端危机倾向，risk_hint 必须为 possible_crisis，tone 使用 crisis_support。
- 危机场景中，不提供危险方法，不鼓励危险行为，应建议联系可信任的人、当地紧急服务或专业支持。
- 普通场景下，回复控制在 2 到 5 句话，像一个可靠、有行动感的聊天助手，而不是分析报告。
- 不要在 reply 里展示情绪标签、置信度、强度分数或 JSON 字段名。

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
        return build_chat_user_prompt(payload)


def build_chat_user_prompt(payload: ChatInput) -> str:
    history = format_chat_history(payload.history)
    judge_result = json.dumps(payload.judge_result or {}, ensure_ascii=False, indent=2)
    metadata = json.dumps(payload.metadata or {}, ensure_ascii=False, indent=2)
    return (
        "请根据最近对话历史、当前用户消息和情绪分析结果生成回复。\n"
        "如果当前用户消息里出现“这/这个/刚才/它/建议/怎么办”等依赖上下文的表达，"
        "必须优先结合最近对话历史理解指代，不要把它当成全新的泛泛问题。\n\n"
        f"conversation_id: {payload.conversation_id or ''}\n"
        f"user_id: {payload.user_id or ''}\n\n"
        "最近对话历史（按时间从旧到新）：\n"
        f"{history}\n\n"
        "当前用户消息：\n"
        f"{payload.text}\n\n"
        "当前消息的情绪分析 judge_result：\n"
        f"{judge_result}\n\n"
        "metadata：\n"
        f"{metadata}"
    )


def format_chat_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "（无历史）"

    lines: list[str] = []
    for item in history[-20:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}: {content}")

    return "\n".join(lines) if lines else "（无历史）"

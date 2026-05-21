import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from agents import config
from agents.llm_server import get_llm_server, LLMServer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_server import get_llm_server, LLMServer

TONE_LABELS = config.TONE_LABELS
RISK_HINTS = config.RISK_HINTS

@dataclass
class ChatInput:
    id: str
    text: str
    sample_type: str
    emotion: str
    secondary_emotion: Optional[str]
    intensity: int
    final_confidence: float
    is_sarcasm: bool
    is_mixed: bool
    reason: str
    tokens: List[str] = field(default_factory=list)
    emotion_words: List[str] = field(default_factory=list)
    source: str = "chat"
    created_at: str = ""


class ChatAgent:
    CRISIS_KEYWORDS = {
        "自杀", "轻生", "不想活", "活不下去", "结束自己", "伤害自己",
        "割腕", "跳楼", "想死", "去死", "杀了我", "杀了他", "同归于尽",
    }

    ACTION_MAP = {
        "焦虑": ["先只列出最急的一件事", "把注意力放回一次缓慢呼吸"],
        "悲伤": ["先允许自己难受一会儿", "找一个可信任的人说一句近况"],
        "愤怒": ["先离开让你更上头的情境一会儿", "等情绪降一点再决定下一步"],
        "厌烦": ["先把最烦的一件事拆小一点", "给自己留一个短暂缓冲"],
        "开心": ["把这份状态记下来", "顺手推进一件你想完成的小事"],
        "中性": ["先确认你现在最在意的点", "如果愿意，可以继续多说一点"],
    }

    SYSTEM_PROMPT = """你是 EmoAgent 的聊天助手。你的输入不是原始分析任务，而是已经完成情绪分析后的结果。
        你的任务：
        1. 根据用户原话和分析结果，生成自然、温和、简洁的中文回复。
        2. 先回应用户感受，再给出支持，不要说教，不要输出分析报告口吻。
        3. 如果结果显示 is_sarcasm=true，要识别到用户可能在用反话或带刺的表达方式释放情绪。
        4. 如果结果显示 is_mixed=true，要承认用户情绪里可能有两股力量同时存在。
        5. 如果文本里出现明显自伤、自杀、伤害他人或极端危机倾向，tone 必须为 crisis_support，risk_hint 必须为 possible_crisis，并建议联系可信任的人或专业支持。
        6. 普通场景回复控制在 1 到 3 句话。

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

    def __init__(self, llm: Optional[LLMServer] = None):
        self.llm = llm or get_llm_server()

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._coerce_input(input_data)
        rule_result = self._rule_result(payload)
        llm_result = self._call_llm(payload, rule_result)
        return self._merge_result(rule_result, llm_result)

    def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(payload)

    def chat_dict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(payload)

    def build_messages(self, payload: Dict[str, Any] | ChatInput) -> List[Dict[str, str]]:
        item = self._coerce_input(payload)
        rule_result = self._rule_result(item)
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(item, rule_result)},
        ]

    def _coerce_input(self, payload: Dict[str, Any] | ChatInput) -> ChatInput:
        if isinstance(payload, ChatInput):
            item = payload
        else:
            item = ChatInput(
                id=str(payload.get("id", "")).strip(),
                text=str(payload.get("text", "")).strip(),
                sample_type=str(payload.get("sample_type", "direct")).strip() or "direct",
                emotion=str(payload.get("emotion", "中性")).strip() or "中性",
                secondary_emotion=self._optional_str(payload.get("secondary_emotion")),
                intensity=self._int_value(payload.get("intensity"), 50),
                final_confidence=self._float_value(payload.get("final_confidence"), 0.5),
                is_sarcasm=bool(payload.get("is_sarcasm", False)),
                is_mixed=bool(payload.get("is_mixed", False)),
                reason=str(payload.get("reason", "")).strip(),
                tokens=self._string_list(payload.get("tokens", [])),
                emotion_words=self._string_list(payload.get("emotion_words", [])),
                source=str(payload.get("source", "chat")).strip() or "chat",
                created_at=str(payload.get("created_at", "")).strip(),
            )

        if not item.text:
            raise ValueError("text 字段为必填")
        return item

    def _rule_result(self, payload: ChatInput) -> Dict[str, Any]:
        risk_hint = self._risk_hint(payload)
        tone = self._tone(payload, risk_hint)
        return {
            "reply": self._fallback_reply(payload, risk_hint),
            "tone": tone,
            "risk_hint": risk_hint,
            "suggested_actions": self._suggested_actions(payload, risk_hint),
            "reason": self._rule_reason(payload, risk_hint),
        }

    def _risk_hint(self, payload: ChatInput) -> str:
        text = payload.text.replace(" ", "")
        return "possible_crisis" if any(keyword in text for keyword in self.CRISIS_KEYWORDS) else "none"

    def _tone(self, payload: ChatInput, risk_hint: str) -> str:
        if risk_hint == "possible_crisis":
            return "crisis_support"
        if payload.is_sarcasm or payload.is_mixed:
            return "reflective"
        if payload.emotion == "焦虑":
            return "calm"
        if payload.emotion == "开心":
            return "encouraging"
        return "supportive"

    def _fallback_reply(self, payload: ChatInput, risk_hint: str) -> str:
        if risk_hint == "possible_crisis":
            return (
                "我很在意你现在提到的这些内容。"
                "如果你已经有伤害自己或他人的打算，请立刻联系身边可信任的人、当地紧急服务或专业心理援助，先确保你不是一个人扛着。"
            )

        if payload.is_sarcasm:
            prefix = f"我能感觉到你这句话表面像在轻描淡写，实际是在表达{payload.emotion}。"
        elif payload.is_mixed and payload.secondary_emotion:
            prefix = f"听起来你现在不只是{payload.emotion}，里面还夹着一些{payload.secondary_emotion}。"
        elif payload.is_mixed:
            prefix = f"听起来你现在的感受有点复杂，核心还是{payload.emotion}。"
        else:
            prefix = f"我能感觉到你现在有些{payload.emotion}。"

        intensity_line = ""
        if payload.intensity >= 80:
            intensity_line = "这股感觉已经很强了，先别急着把自己逼得更紧。"
        elif payload.intensity >= 60:
            intensity_line = "这份感受已经挺明显了，先照顾一下当下的自己也很重要。"
        else:
            intensity_line = "如果你愿意，也可以继续多说一点，我陪你一起理一理。"

        return prefix + intensity_line

    def _suggested_actions(self, payload: ChatInput, risk_hint: str) -> List[str]:
        if risk_hint == "possible_crisis":
            return [
                "立刻联系一个你信任的人陪着你",
                "尽快联系当地紧急服务或专业心理援助",
            ]

        actions = list(self.ACTION_MAP.get(payload.emotion, self.ACTION_MAP["中性"]))
        if payload.is_mixed and payload.secondary_emotion:
            actions = actions[:1] + [f"分别说一句你对“{payload.emotion}”和“{payload.secondary_emotion}”最想表达的话"]
        return actions[:2]

    def _rule_reason(self, payload: ChatInput, risk_hint: str) -> str:
        if risk_hint == "possible_crisis":
            return "文本包含明显危机表达，优先提供安全支持。"
        if payload.is_sarcasm:
            return f"分析结果显示存在反讽，最终情绪为{payload.emotion}。"
        if payload.is_mixed and payload.secondary_emotion:
            return f"分析结果显示混合情绪，主情绪为{payload.emotion}，次情绪为{payload.secondary_emotion}。"
        return f"分析结果显示当前主情绪为{payload.emotion}，强度为{payload.intensity}。"

    def _call_llm(self, payload: ChatInput, rule_result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(payload, rule_result)},
            ]
            content = self.llm.chat(messages, temperature=0.4)
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            return {}
        return {}

    def _build_user_prompt(self, payload: ChatInput, rule_result: Dict[str, Any]) -> str:
        return (
            "请基于下面已经完成的情绪分析结果，为用户生成聊天回复。\n\n"
            f"分析输入：{json.dumps(asdict(payload), ensure_ascii=False, indent=2)}\n\n"
            f"规则建议：{json.dumps(rule_result, ensure_ascii=False, indent=2)}"
        )

    def _merge_result(self, rule_result: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
        reply = str(llm_result.get("reply", "")).strip() if llm_result else ""
        tone = str(llm_result.get("tone", "")).strip() if llm_result else ""
        risk_hint = str(llm_result.get("risk_hint", "")).strip() if llm_result else ""
        reason = str(llm_result.get("reason", "")).strip() if llm_result else ""

        if tone not in TONE_LABELS:
            tone = rule_result["tone"]
        if risk_hint not in RISK_HINTS:
            risk_hint = rule_result["risk_hint"]

        actions = self._string_list(llm_result.get("suggested_actions", [])) if llm_result else []
        if not actions:
            actions = rule_result["suggested_actions"]

        return {
            "reply": reply or rule_result["reply"],
            "tone": tone,
            "risk_hint": risk_hint,
            "suggested_actions": actions,
            "reason": reason or rule_result["reason"],
        }

    def _optional_str(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _int_value(self, value: Any, default: int) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    def _float_value(self, value: Any, default: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

from .schemas import EmotionInput, EmotionResult


EMOTION_LABELS = {"开心", "悲伤", "愤怒", "焦虑", "厌烦", "中性"}

SYSTEM_PROMPT = """你是情绪识别系统中的 Emotion Agent。

你的任务是做“表层情绪判断”，不要负责反讽修正，也不要负责复杂混合情绪融合。
即使句子可能存在反讽，你也只需要按文本表面表达输出结果，后续会交给 Sarcasm Agent 或 Mix Agent 修正。

你需要一次性完成：
1. 分词或切分关键短语 tokens
2. 提取情绪词 emotion_words
3. 提取程度词 degree_words
4. 提取否定词 negation_words
5. 提取转折词 contrast_words
6. 判断主情绪 emotion
7. 给出情绪强度 intensity
8. 给出置信度 confidence
9. 给出简短初步解释 reason

第一版主情绪标签只能从以下 6 类中选择：
- 开心
- 悲伤
- 愤怒
- 焦虑
- 厌烦
- 中性

判断规则：
- emotion 只能输出上述标签之一。
- intensity 是 0 到 100 的整数。中性通常为 0 到 30；明显但不强烈的情绪通常为 40 到 65；强烈情绪通常为 66 到 100。
- confidence 是 0 到 1 的小数。
- tokens 应尽量保留中文词语、短语、标点外的语义片段。
- emotion_words 只放直接表达情绪或明显情绪方向的词语。
- degree_words 放“很、特别、太、稍微、有点、极其、非常”等程度修饰。
- negation_words 放“不、没、没有、别、无、并非”等否定表达。
- contrast_words 放“但、但是、不过、然而、却、只是、可、虽然”等转折表达。
- reason 用一句中文解释表层判断依据，不要超过 80 字。

输出要求：
- 只返回 JSON
- 不要输出 markdown
- 字段必须完整
- 不要增加额外字段

输出格式：
{
  "tokens": ["太好了", "周末", "又", "能", "继续", "改", "需求"],
  "emotion_words": ["太好了"],
  "degree_words": [],
  "negation_words": [],
  "contrast_words": [],
  "emotion": "开心",
  "intensity": 62,
  "confidence": 0.61,
  "reason": "文本表面存在明显正向表达“太好了”，情绪方向初步判为正向"
}
"""


class EmotionLLMClient(Protocol):
    def analyze(self, payload: EmotionInput) -> dict[str, Any]:
        """Send payload to an LLM and return the parsed JSON result."""


class EmotionAgent:
    """LLM-based surface emotion agent."""

    def __init__(self, client: EmotionLLMClient) -> None:
        self.client = client

    def emotionRe(self, payload: EmotionInput | dict[str, Any]) -> EmotionResult:
        item = payload if isinstance(payload, EmotionInput) else EmotionInput(**payload)
        raw_result = self.client.analyze(item)
        return self._build_result(raw_result)

    def emotionRe_dict(self, payload: EmotionInput | dict[str, Any]) -> dict[str, Any]:
        return self.emotionRe(payload).to_dict()

    def _build_result(self, raw_result: dict[str, Any]) -> EmotionResult:
        emotion = str(raw_result.get("emotion", "")).strip()
        if emotion not in EMOTION_LABELS:
            raise ValueError(f"Invalid emotion from LLM: {emotion!r}")

        intensity = self._coerce_int(raw_result.get("intensity"), "intensity")
        if not 0 <= intensity <= 100:
            raise ValueError(f"Invalid intensity from LLM: {intensity!r}")

        confidence = self._coerce_float(raw_result.get("confidence"), "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence from LLM: {confidence!r}")

        return EmotionResult(
            tokens=self._coerce_str_list(raw_result.get("tokens", []), "tokens"),
            emotion_words=self._coerce_str_list(raw_result.get("emotion_words", []), "emotion_words"),
            degree_words=self._coerce_str_list(raw_result.get("degree_words", []), "degree_words"),
            negation_words=self._coerce_str_list(raw_result.get("negation_words", []), "negation_words"),
            contrast_words=self._coerce_str_list(raw_result.get("contrast_words", []), "contrast_words"),
            emotion=emotion,
            intensity=intensity,
            confidence=confidence,
            reason=str(raw_result.get("reason", "")).strip(),
        )
    
    def _coerce_str_list(self, value: Any, field_name: str) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"Invalid {field_name} from LLM: expected list")
        return [str(item).strip() for item in value if str(item).strip()]

    def _coerce_int(self, value: Any, field_name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}") from exc

    def _coerce_float(self, value: Any, field_name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}") from exc


    def build_messages(self, payload: EmotionInput | dict[str, Any]) -> list[dict[str, str]]:
        item = payload if isinstance(payload, EmotionInput) else EmotionInput(**payload)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(item)},
        ]

    def _build_user_prompt(self, payload: EmotionInput) -> str:
        return (
            "请对下面这条消息做表层情绪识别，并严格返回 JSON 结果。\n\n"
            f"{json.dumps(asdict(payload), ensure_ascii=False, indent=2)}"
        )

   

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any
from typing import Protocol

from .schemas import JudgeInput, JudgeResult


SYSTEM_PROMPT = """你是情绪分析流水线中的最终 Judge Agent。
你的任务是在上游各个 agent 的结果之间进行裁决，并输出一个最终的 JSON 结果。

输入中可能包含：
- 原始文本（如果有）
- router_result
- emotion_result
- sarcasm_result
- mix_result
- rule_result（基于规则的确定性兜底结果）

裁决原则：
- 优先采用与原始文本证据更一致、置信度更高的上游结果。
- 如果上游结果彼此一致，优先保留 rule_result。
- 如果反讽证据充分，采用 sarcasm_result 中识别出的真实情绪。
- 如果混合情绪证据充分，保留主情绪和次情绪。
- 除非原始文本有非常明确的依据，否则不要凭空创造与上游输出无关的情绪标签。
- final_confidence 必须反映证据质量，而不只是多个 agent 是否一致。

只返回 JSON，并且字段必须严格为：
{
  "final_emotion": "string",
  "secondary_emotion": "string or null",
  "final_intensity": 0,
  "final_confidence": 0.0,
  "is_sarcasm": false,
  "is_mixed": false,
  "reason": "简短中文说明"
}
"""


class JudgeLLMClient(Protocol):
    def arbitrate(self, payload: JudgeInput, rule_result: JudgeResult) -> dict[str, Any]:
        """Send payload and rule result to an LLM and return the parsed JSON result."""


class JudgeAgent:
    """Hybrid final judge that uses rules first and calls an LLM for ambiguous cases."""

    def __init__(
        self,
        client: JudgeLLMClient | None = None,
        sarcasm_confidence_threshold: float = 0.65,
        mix_confidence_threshold: float = 0.65,
        emotion_confidence_threshold: float = 0.65,
        review_confidence_margin: float = 0.15,
    ) -> None:
        self.client = client
        self.sarcasm_confidence_threshold = sarcasm_confidence_threshold
        self.mix_confidence_threshold = mix_confidence_threshold
        self.emotion_confidence_threshold = emotion_confidence_threshold
        self.review_confidence_margin = review_confidence_margin

    def judge(self, payload: JudgeInput | dict[str, Any]) -> JudgeResult:
        item = payload if isinstance(payload, JudgeInput) else JudgeInput(**payload)
        rule_result = self._judge_by_rules(item)

        if self.client is None or not self._should_call_llm(item):
            return rule_result

        raw_result = self.client.arbitrate(item, rule_result)
        return self._build_result(raw_result)

    def _judge_by_rules(self, item: JudgeInput) -> JudgeResult:

        router = item.router_result
        emotion = item.emotion_result
        sarcasm = item.sarcasm_result or {}
        mix = item.mix_result or {}

        sample_type = str(router.get("sample_type", "")).strip()
        if sample_type not in {"direct", "sarcasm_suspected", "mix"}:
            raise ValueError(f"Invalid sample_type from router_result: {sample_type!r}")

        emotion_label = str(emotion.get("emotion", "")).strip()
        emotion_intensity = self._coerce_int(emotion.get("intensity"), "emotion.intensity")
        emotion_confidence = self._clamp01(self._coerce_float(emotion.get("confidence"), "emotion.confidence"))
        emotion_reason = str(emotion.get("reason", "")).strip()

        if sample_type == "direct":
            return JudgeResult(
                final_emotion=emotion_label,
                secondary_emotion=None,
                final_intensity=emotion_intensity,
                final_confidence=emotion_confidence,
                is_sarcasm=False,
                is_mixed=False,
                reason=f"direct 路由，直接采用 Emotion 结果。{emotion_reason}".strip(),
            )

        if sample_type == "sarcasm_suspected":
            return self._judge_sarcasm_branch(
                emotion_label=emotion_label,
                emotion_intensity=emotion_intensity,
                emotion_confidence=emotion_confidence,
                emotion_reason=emotion_reason,
                sarcasm=sarcasm,
            )

        return self._judge_mix_branch(
            emotion_label=emotion_label,
            emotion_intensity=emotion_intensity,
            emotion_confidence=emotion_confidence,
            emotion_reason=emotion_reason,
            mix=mix,
        )

    def judge_dict(self, payload: JudgeInput | dict[str, Any]) -> dict[str, Any]:
        return self.judge(payload).to_dict()

    def _should_call_llm(self, item: JudgeInput) -> bool:
        router = item.router_result
        emotion = item.emotion_result
        sarcasm = item.sarcasm_result or {}
        mix = item.mix_result or {}

        sample_type = str(router.get("sample_type", "")).strip()
        emotion_label = str(emotion.get("emotion", "")).strip()
        emotion_confidence = self._clamp01(self._coerce_float(emotion.get("confidence"), "emotion.confidence"))

        if emotion_confidence < self.emotion_confidence_threshold:
            return True

        if sample_type == "sarcasm_suspected":
            if not sarcasm:
                return True
            sarcasm_confidence = self._clamp01(
                self._coerce_float(sarcasm.get("confidence", 0), "sarcasm.confidence")
            )
            true_emotion = str(sarcasm.get("true_emotion", "")).strip()
            confidence_gap = abs(sarcasm_confidence - emotion_confidence)
            return (
                self._is_near_threshold(sarcasm_confidence, self.sarcasm_confidence_threshold)
                or confidence_gap <= self.review_confidence_margin
                or (bool(sarcasm.get("is_sarcasm")) and true_emotion and true_emotion != emotion_label)
            )

        if sample_type == "mix":
            if not mix:
                return True
            mix_confidence = self._clamp01(self._coerce_float(mix.get("confidence", 0), "mix.confidence"))
            primary_emotion = str(mix.get("primary_emotion", "")).strip()
            confidence_gap = abs(mix_confidence - emotion_confidence)
            return (
                self._is_near_threshold(mix_confidence, self.mix_confidence_threshold)
                or confidence_gap <= self.review_confidence_margin
                or (bool(mix.get("is_mixed")) and primary_emotion and primary_emotion != emotion_label)
            )

        return False

    def _is_near_threshold(self, confidence: float, threshold: float) -> bool:
        return abs(confidence - threshold) <= self.review_confidence_margin

    def _build_result(self, raw_result: dict[str, Any]) -> JudgeResult:
        final_emotion = str(raw_result.get("final_emotion", "")).strip()
        if not final_emotion:
            raise ValueError("Invalid final_emotion from LLM: empty")

        secondary_emotion_value = raw_result.get("secondary_emotion")
        secondary_emotion = None
        if secondary_emotion_value is not None:
            secondary_emotion = str(secondary_emotion_value).strip() or None

        final_intensity = self._coerce_int(raw_result.get("final_intensity"), "final_intensity")
        final_confidence = self._clamp01(self._coerce_float(raw_result.get("final_confidence"), "final_confidence"))

        return JudgeResult(
            final_emotion=final_emotion,
            secondary_emotion=secondary_emotion,
            final_intensity=final_intensity,
            final_confidence=final_confidence,
            is_sarcasm=self._coerce_bool(raw_result.get("is_sarcasm"), "is_sarcasm"),
            is_mixed=self._coerce_bool(raw_result.get("is_mixed"), "is_mixed"),
            reason=str(raw_result.get("reason", "")).strip(),
        )

    def _judge_sarcasm_branch(
        self,
        emotion_label: str,
        emotion_intensity: int,
        emotion_confidence: float,
        emotion_reason: str,
        sarcasm: dict[str, Any],
    ) -> JudgeResult:
        is_sarcasm = bool(sarcasm.get("is_sarcasm"))
        sarcasm_confidence = self._clamp01(self._coerce_float(sarcasm.get("confidence", 0), "sarcasm.confidence"))

        if is_sarcasm and sarcasm_confidence >= self.sarcasm_confidence_threshold:
            true_emotion = str(sarcasm.get("true_emotion", "")).strip()
            revised_intensity = self._coerce_int(
                sarcasm.get("revised_intensity"),
                "sarcasm.revised_intensity",
            )
            final_confidence = self._clamp01(emotion_confidence*0.3 + sarcasm_confidence*0.7)
            return JudgeResult(
                final_emotion=true_emotion,
                secondary_emotion=None,
                final_intensity=revised_intensity,
                final_confidence=final_confidence,
                is_sarcasm=True,
                is_mixed=False,
                reason=str(sarcasm.get("reason", "")).strip() or "反讽成立，采用 Sarcasm 修正结果。",
            )

        if is_sarcasm and sarcasm_confidence < self.sarcasm_confidence_threshold:
            return JudgeResult(
                final_emotion=emotion_label,
                secondary_emotion=None,
                final_intensity=emotion_intensity,
                final_confidence=self._clamp01(emotion_confidence * 0.8),
                is_sarcasm=False,
                is_mixed=False,
                reason="Sarcasm 置信度偏低，回退 Emotion 结果并下调总置信度。",
            )

        return JudgeResult(
            final_emotion=emotion_label,
            secondary_emotion=None,
            final_intensity=emotion_intensity,
            final_confidence=self._clamp01(emotion_confidence * 0.9),
            is_sarcasm=False,
            is_mixed=False,
            reason="反讽未成立，采用 Emotion 结果。",
        )

    def _judge_mix_branch(
        self,
        emotion_label: str,
        emotion_intensity: int,
        emotion_confidence: float,
        emotion_reason: str,
        mix: dict[str, Any],
    ) -> JudgeResult:
        is_mixed = bool(mix.get("is_mixed"))
        mix_confidence = self._clamp01(self._coerce_float(mix.get("confidence", 0), "mix.confidence"))

        if is_mixed and mix_confidence >= self.mix_confidence_threshold:
            primary_emotion = str(mix.get("primary_emotion", "")).strip()
            secondary_emotion = str(mix.get("secondary_emotion", "")).strip() or None
            revised_intensity = self._coerce_int(mix.get("revised_intensity"), "mix.revised_intensity")
            final_confidence = self._clamp01(emotion_confidence*0.3 + mix_confidence*0.7)
            return JudgeResult(
                final_emotion=primary_emotion,
                secondary_emotion=secondary_emotion,
                final_intensity=revised_intensity,
                final_confidence=final_confidence,
                is_sarcasm=False,
                is_mixed=True,
                reason=str(mix.get("reason", "")).strip() or "混合情绪成立，采用 Mix 结果。",
            )

        if is_mixed and mix_confidence < self.mix_confidence_threshold:
            return JudgeResult(
                final_emotion=emotion_label,
                secondary_emotion=None,
                final_intensity=emotion_intensity,
                final_confidence=self._clamp01(emotion_confidence * 0.8),
                is_sarcasm=False,
                is_mixed=False,
                reason="Mix 置信度偏低，回退 Emotion 结果并下调总置信度。",
            )

        return JudgeResult(
            final_emotion=emotion_label,
            secondary_emotion=None,
            final_intensity=emotion_intensity,
            final_confidence=self._clamp01(emotion_confidence * 0.9),
            is_sarcasm=False,
            is_mixed=False,
            reason=f"未识别到稳定混合情绪，采用 Emotion 结果。{emotion_reason}".strip(),
        )

    def _coerce_int(self, value: Any, field_name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"Invalid {field_name}: {value!r}")
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name}: {value!r}") from exc
        if out < 0 or out > 100:
            raise ValueError(f"Invalid {field_name}: {out!r}")
        return out

    def _coerce_float(self, value: Any, field_name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"Invalid {field_name}: {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name}: {value!r}") from exc

    def _coerce_bool(self, value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"Invalid {field_name}: {value!r}")

    def _clamp01(self, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return round(value, 4)

    def build_messages(
        self,
        payload: JudgeInput | dict[str, Any],
        rule_result: JudgeResult | dict[str, Any],
    ) -> list[dict[str, str]]:
        item = payload if isinstance(payload, JudgeInput) else JudgeInput(**payload)
        rule = rule_result if isinstance(rule_result, JudgeResult) else JudgeResult(**rule_result)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(item, rule)},
        ]

    def _build_user_prompt(self, payload: JudgeInput, rule_result: JudgeResult) -> str:
        body = {
            "payload": asdict(payload),
            "rule_result": rule_result.to_dict(),
        }
        return (
            "请审阅下面的情绪分析流水线结果，并返回最终的 JudgeResult JSON。\n\n"
            f"{json.dumps(body, ensure_ascii=False, indent=2)}"
        )

from __future__ import annotations

from typing import Any

from .schemas import JudgeInput, JudgeResult


class JudgeAgent:
    """Rule-based final judge that fuses outputs from upstream agents."""

    def __init__(
        self,
        sarcasm_confidence_threshold: float = 0.65,
        mix_confidence_threshold: float = 0.65,
    ) -> None:
        self.sarcasm_confidence_threshold = sarcasm_confidence_threshold
        self.mix_confidence_threshold = mix_confidence_threshold

    def judge(self, payload: JudgeInput | dict[str, Any]) -> JudgeResult:
        item = payload if isinstance(payload, JudgeInput) else JudgeInput(**payload)

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
            final_confidence = self._clamp01((emotion_confidence + sarcasm_confidence) / 2)
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
            final_confidence = self._clamp01((emotion_confidence + mix_confidence) / 2)
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

    def _clamp01(self, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return round(value, 4)

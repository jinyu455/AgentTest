"""Judge Agent 的核心逻辑实现。

负责情绪分析流水线的最终裁决，采用"规则优先 + 大模型兜底"的混合策略：
- 对于确定性高的场景（如 direct 路由），直接通过规则裁决
- 对于模糊或矛盾的场景，调用大模型进行辅助裁决
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any
from typing import Protocol

from base.coerce import CoercionMixin
from .schemas import JudgeInput, JudgeResult

# Judge Agent 的系统提示词，定义了大模型的裁决原则和输出格式
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


def build_judge_user_prompt(payload: JudgeInput, rule_result: JudgeResult) -> str:
    """构造 Judge Agent 的用户提示词，将流水线输入和规则裁决结果打包为 JSON。

    同时供 client.py 和 Agent.build_messages() 复用，避免重复构建。
    """
    body = {
        "payload": asdict(payload),
        "rule_result": rule_result.to_dict(),
    }
    return (
        "请审阅下面的情绪分析流水线结果，并返回最终的 JudgeResult JSON。\n\n"
        f"{json.dumps(body, ensure_ascii=False, indent=2)}"
    )


class JudgeLLMClient(Protocol):
    """Judge Agent 的大模型客户端协议接口。

    定义了 arbitrate 方法的签名，任何实现该协议的客户端都可以
    用于 JudgeAgent 的最终裁决。
    """

    def arbitrate(self, payload: JudgeInput, rule_result: JudgeResult) -> dict[str, Any]:
        """发送流水线输入和规则裁决结果到大模型，返回解析后的 JSON 结果。"""


class JudgeAgent(CoercionMixin):
    """混合裁决 Agent，优先使用规则进行裁决，对模糊场景调用大模型辅助。

    这是情绪分析流水线的最终决策者，负责整合 Router、Emotion、
    Sarcasm、Mix 等上游 Agent 的分析结果，输出最终的情绪判定。
    """

    def __init__(
        self,
        client: JudgeLLMClient | None = None,
        sarcasm_confidence_threshold: float = 0.65,
        mix_confidence_threshold: float = 0.65,
        emotion_confidence_threshold: float = 0.65,
        review_confidence_margin: float = 0.15,
    ) -> None:
        """初始化 JudgeAgent。

        Args:
            client: 大模型客户端实例，为 None 时仅使用规则裁决（纯规则模式）。
            sarcasm_confidence_threshold: 反讽置信度阈值，超过此值才认为反讽成立。
            mix_confidence_threshold: 混合情绪置信度阈值，超过此值才认为混合情绪成立。
            emotion_confidence_threshold: 情感置信度阈值，低于此值时会调用大模型复核。
            review_confidence_margin: 复核置信度差值阈值，当两个 Agent 的置信度差小于此值时需要大模型介入。
        """
        self.client = client
        self.sarcasm_confidence_threshold = sarcasm_confidence_threshold
        self.mix_confidence_threshold = mix_confidence_threshold
        self.emotion_confidence_threshold = emotion_confidence_threshold
        self.review_confidence_margin = review_confidence_margin

    def judge(self, payload: JudgeInput | dict[str, Any]) -> JudgeResult:
        """执行最终裁决，返回结构化的 JudgeResult 对象。

        裁决流程：
        1. 先通过规则进行确定性裁决
        2. 判断是否需要调用大模型复核
        3. 如果需要，调用大模型获取最终结果

        Args:
            payload: 上游各 Agent 分析结果的聚合输入。

        Returns:
            最终裁决结果。
        """
        item = payload if isinstance(payload, JudgeInput) else JudgeInput(**payload)
        # 第一步：基于规则进行确定性裁决
        rule_result = self._judge_by_rules(item)

        # 如果没有大模型客户端，或者规则裁决已足够确定，则直接返回规则结果
        if self.client is None or not self._should_call_llm(item):
            return rule_result

        # 需要大模型辅助裁决时，调用大模型
        raw_result = self.client.arbitrate(item, rule_result)
        return self._build_result(raw_result)

    def _judge_by_rules(self, item: JudgeInput) -> JudgeResult:
        """基于规则进行确定性裁决。

        根据路由结果的 sample_type 分发到不同的裁决分支：
        - direct: 直接采用 Emotion Agent 的结果
        - sarcasm_suspected: 进入反讽裁决分支
        - mix: 进入混合情绪裁决分支

        Args:
            item: 上游各 Agent 分析结果的聚合输入。

        Returns:
            基于规则的裁决结果。

        Raises:
            ValueError: 当 sample_type 不合法时抛出。
        """
        router = item.router_result
        emotion = item.emotion_result
        sarcasm = item.sarcasm_result or {}
        mix = item.mix_result or {}

        # 校验路由类型
        sample_type = str(router.get("sample_type", "")).strip()
        if sample_type not in {"direct", "sarcasm_suspected", "mix"}:
            raise ValueError(f"Invalid sample_type from router_result: {sample_type!r}")

        # 提取情感 Agent 的结果
        emotion_label = str(emotion.get("emotion", "")).strip()
        emotion_intensity = self._coerce_int(emotion.get("intensity"), "emotion.intensity")
        emotion_confidence = self._clamp01(self._coerce_float(emotion.get("confidence"), "emotion.confidence"))
        emotion_reason = str(emotion.get("reason", "")).strip()

        # direct 路由：直接采用 Emotion Agent 的结果
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

        # 反讽疑似：进入反讽裁决分支
        if sample_type == "sarcasm_suspected":
            return self._judge_sarcasm_branch(
                emotion_label=emotion_label,
                emotion_intensity=emotion_intensity,
                emotion_confidence=emotion_confidence,
                emotion_reason=emotion_reason,
                sarcasm=sarcasm,
            )

        # 混合情绪：进入混合情绪裁决分支
        return self._judge_mix_branch(
            emotion_label=emotion_label,
            emotion_intensity=emotion_intensity,
            emotion_confidence=emotion_confidence,
            emotion_reason=emotion_reason,
            mix=mix,
        )

    def judge_dict(self, payload: JudgeInput | dict[str, Any]) -> dict[str, Any]:
        """执行最终裁决，返回字典格式的结果。

        是 judge 的便捷版本，直接返回可序列化的字典。
        """
        return self.judge(payload).to_dict()

    def _should_call_llm(self, item: JudgeInput) -> bool:
        """判断是否需要调用大模型进行辅助裁决。

        在以下情况下需要调用大模型：
        1. Emotion Agent 的置信度低于阈值
        2. 反讽场景下，Sarcasm Agent 缺失、置信度接近阈值、
           或与 Emotion Agent 的置信度差值较小
        3. 混合情绪场景下类似的情况

        Args:
            item: 上游各 Agent 分析结果的聚合输入。

        Returns:
            True 表示需要调用大模型，False 表示规则裁决已足够。
        """
        router = item.router_result
        emotion = item.emotion_result
        sarcasm = item.sarcasm_result or {}
        mix = item.mix_result or {}

        sample_type = str(router.get("sample_type", "")).strip()
        emotion_label = str(emotion.get("emotion", "")).strip()
        emotion_confidence = self._clamp01(self._coerce_float(emotion.get("confidence"), "emotion.confidence"))

        # Emotion 置信度过低时，需要大模型复核
        if emotion_confidence < self.emotion_confidence_threshold:
            return True

        # 反讽疑似分支的判断逻辑
        if sample_type == "sarcasm_suspected":
            # Sarcasm Agent 缺失，必须调用大模型
            if not sarcasm:
                return True
            sarcasm_confidence = self._clamp01(
                self._coerce_float(sarcasm.get("confidence", 0), "sarcasm.confidence")
            )
            true_emotion = str(sarcasm.get("true_emotion", "")).strip()
            confidence_gap = abs(sarcasm_confidence - emotion_confidence)
            # 以下任一条件满足则需要大模型介入：
            # 1. 反讽置信度低于阈值（结果不可靠）
            # 2. 反讽和情感 Agent 的置信度差距小（难以确定哪个更可信）
            # 3. 反讽成立且识别出的真实情绪与情感 Agent 不同（存在矛盾）
            return (
                sarcasm_confidence < self.sarcasm_confidence_threshold
                or confidence_gap <= self.review_confidence_margin
                or (bool(sarcasm.get("is_sarcasm")) and true_emotion and true_emotion != emotion_label)
            )

        # 混合情绪分支的判断逻辑（与反讽分支类似）
        if sample_type == "mix":
            if not mix:
                return True
            mix_confidence = self._clamp01(self._coerce_float(mix.get("confidence", 0), "mix.confidence"))
            primary_emotion = str(mix.get("primary_emotion", "")).strip()
            confidence_gap = abs(mix_confidence - emotion_confidence)
            return (
                mix_confidence < self.mix_confidence_threshold
                or confidence_gap <= self.review_confidence_margin
                or (bool(mix.get("is_mixed")) and primary_emotion and primary_emotion != emotion_label)
            )

        return False

    def _build_result(self, raw_result: dict[str, Any]) -> JudgeResult:
        """将大模型返回的原始字典校验并构建为 JudgeResult 对象。

        对每个字段进行类型强转和范围校验，不合法时抛出 ValueError。

        Args:
            raw_result: 大模型返回的原始字典。

        Returns:
            校验通过的 JudgeResult 对象。

        Raises:
            ValueError: 当字段值不符合约束条件时抛出。
        """
        # 最终情绪不能为空
        final_emotion = str(raw_result.get("final_emotion", "")).strip()
        if not final_emotion:
            raise ValueError("Invalid final_emotion from LLM: empty")

        # 次情绪可选，空字符串等同于 None
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
        """反讽分支的规则裁决逻辑。

        根据反讽检测结果进行裁决，分三种情况：
        1. 反讽成立且置信度高：采用 Sarcasm Agent 修正后的真实情绪
        2. 反讽成立但置信度低：回退到 Emotion Agent 的结果并下调置信度
        3. 反讽不成立：采用 Emotion Agent 的结果并轻微下调置信度

        Args:
            emotion_label: Emotion Agent 判定的情绪标签。
            emotion_intensity: Emotion Agent 判定的情绪强度。
            emotion_confidence: Emotion Agent 的置信度。
            emotion_reason: Emotion Agent 的判定理由。
            sarcasm: Sarcasm Agent 的检测结果字典。

        Returns:
            反讽分支的裁决结果。
        """
        is_sarcasm = bool(sarcasm.get("is_sarcasm"))
        sarcasm_confidence = self._clamp01(self._coerce_float(sarcasm.get("confidence", 0), "sarcasm.confidence"))

        # 反讽成立且置信度足够高：采用反讽 Agent 修正后的真实情绪
        # 最终置信度按 3:7 加权（情感 30% + 反讽 70%），因为反讽修正是关键信息
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

        # 反讽成立但置信度不足：回退到 Emotion 结果，下调 20% 置信度
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

        # 反讽不成立：采用 Emotion 结果，轻微下调 10% 置信度
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
        """混合情绪分支的规则裁决逻辑。

        根据混合情绪检测结果进行裁决，分三种情况：
        1. 混合情绪成立且置信度高：采用 Mix Agent 的主次情绪及修正强度
        2. 混合情绪成立但置信度低：回退到 Emotion Agent 的结果并下调置信度
        3. 混合情绪不成立：采用 Emotion Agent 的结果并轻微下调置信度

        Args:
            emotion_label: Emotion Agent 判定的情绪标签。
            emotion_intensity: Emotion Agent 判定的情绪强度。
            emotion_confidence: Emotion Agent 的置信度。
            emotion_reason: Emotion Agent 的判定理由。
            mix: Mix Agent 的检测结果字典。

        Returns:
            混合情绪分支的裁决结果。
        """
        is_mixed = bool(mix.get("is_mixed"))
        mix_confidence = self._clamp01(self._coerce_float(mix.get("confidence", 0), "mix.confidence"))

        # 混合情绪成立且置信度足够高：采用 Mix Agent 的结果
        # 最终置信度按 3:7 加权（情感 30% + 混合 70%）
        if is_mixed and mix_confidence >= self.mix_confidence_threshold:
            primary_emotion = str(mix.get("primary_emotion", "")).strip()
            secondary_emotion = str(mix.get("secondary_emotion", "")).strip() or None
            adjusted_intensity = self._coerce_int(mix.get("adjusted_intensity"), "mix.adjusted_intensity")
            final_confidence = self._clamp01(emotion_confidence*0.3 + mix_confidence*0.7)
            return JudgeResult(
                final_emotion=primary_emotion,
                secondary_emotion=secondary_emotion,
                final_intensity=adjusted_intensity,
                final_confidence=final_confidence,
                is_sarcasm=False,
                is_mixed=True,
                reason=str(mix.get("reason", "")).strip() or "混合情绪成立，采用 Mix 结果。",
            )

        # 混合情绪成立但置信度不足：回退到 Emotion 结果，下调 20% 置信度
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

        # 混合情绪不成立：采用 Emotion 结果，轻微下调 10% 置信度
        return JudgeResult(
            final_emotion=emotion_label,
            secondary_emotion=None,
            final_intensity=emotion_intensity,
            final_confidence=self._clamp01(emotion_confidence * 0.9),
            is_sarcasm=False,
            is_mixed=False,
            reason=f"未识别到稳定混合情绪，采用 Emotion 结果。{emotion_reason}".strip(),
        )

    # def build_messages(
    #     self,
    #     payload: JudgeInput | dict[str, Any],
    #     rule_result: JudgeResult | dict[str, Any],
    # ) -> list[dict[str, str]]:
    #     """构建发送给大模型的消息列表。

    #     包含系统提示词和用户提示词，用户提示词中包含流水线输入和规则裁决结果。

    #     Args:
    #         payload: 上游各 Agent 分析结果的聚合输入。
    #         rule_result: 基于规则的裁决结果。

    #     Returns:
    #         符合 OpenAI 对话格式的消息列表。
    #     """
    #     item = payload if isinstance(payload, JudgeInput) else JudgeInput(**payload)
    #     rule = rule_result if isinstance(rule_result, JudgeResult) else JudgeResult(**rule_result)
    #     return [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": build_judge_user_prompt(item, rule)},
    #     ]

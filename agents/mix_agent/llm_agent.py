"""Mix Agent 的核心逻辑实现。

负责混合情绪分析的业务逻辑，包括系统提示词定义、LLM 协议接口定义、
以及对大模型返回结果的校验和类型强转。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

from base.coerce import CoercionMixin
from base.schemas import EMOTION_LABELS
from .schemas import MixInput, MixResult


# Mix Agent 的系统提示词，定义了大模型的角色、任务、输出格式和约束规则
SYSTEM_PROMPT = """你是情绪识别系统中的 Mix Agent。

你的任务是处理"单标签难以表达"的复杂文本，该模块通常在 Router 给出 need_mix_check=true 时被调用。重点关注：
1. 是否混合情绪 is_mixed
2. 主情绪 primary_emotion
3. 次情绪 secondary_emotion
4. 情绪比例 mix_ratio
5. 混合场景主情绪强度 adjusted_intensity
6. 置信度 confidence
7. 简短解释 reason

你需要重点识别：
- 转折结构（但、但是、不过、然而、就是、只是）
- 模糊低能量表达（提不起劲、说不上来、还好但空）
- 同句中的双向情绪（轻松但空、开心但累）

情绪标签建议从以下集合中选择：
- 开心
- 悲伤
- 愤怒
- 焦虑
- 厌烦
- 中性
- 疲惫
- 失落
- 无奈

输出规则：
- is_mixed 为布尔值
- primary_emotion / secondary_emotion 必须是单个标签
- mix_ratio 为对象，至少包含 primary_emotion 与 secondary_emotion 两个键
- mix_ratio 的值为 0 到 1 的小数，整体和接近 1（允许轻微浮动，控制在 0.05 以内）
- adjusted_intensity 是 0 到 100 的整数
- confidence 是 0 到 1 的小数
- reason 用一句中文说明，不超过 100 字

输出要求：
- 只返回 JSON
- 不要输出 markdown
- 字段必须完整
- 不要增加额外字段

输出格式：
{
  "is_mixed": true,
  "primary_emotion": "疲惫",
  "secondary_emotion": "开心",
  "mix_ratio": {
    "疲惫": 0.58,
    "开心": 0.42
  },
  "adjusted_intensity": 57,
  "confidence": 0.79,
  "reason": "句子存在转折结构"但"，前半句偏正向，后半句突出疲惫感，属于混合情绪"
}
"""


def build_mix_user_prompt(payload: MixInput) -> str:
    """构造 Mix Agent 的用户提示词，将输入数据序列化为可读 JSON。

    同时供 client.py 和 Agent.build_messages() 复用，避免重复构建。
    """
    return (
        "请判断下面这条消息是否属于混合情绪，并返回 JSON 结果。\n\n"
        f"{json.dumps(asdict(payload), ensure_ascii=False, indent=2)}"
    )


class MixLLMClient(Protocol):
    """Mix Agent 的大模型客户端协议接口。

    定义了 analyze 方法的签名，任何实现该协议的客户端都可以
    用于 MixAgent 的混合情绪分析。
    """

    def analyze(self, payload: MixInput) -> dict[str, Any]:
        """发送输入到大模型并返回解析后的 JSON 结果。"""


class MixAgent(CoercionMixin):
    """基于大模型的混合情绪分析 Agent。

    负责调用大模型进行混合情绪判断，并对返回结果进行严格的
    校验和类型强转，确保输出符合 MixResult 数据结构的要求。
    """

    def __init__(self, client: MixLLMClient) -> None:
        """初始化 MixAgent。

        Args:
            client: 实现 MixLLMClient 协议的大模型客户端实例。
        """
        self.client = client

    def mixRe(self, payload: MixInput | dict[str, Any]) -> MixResult:
        """执行混合情绪分析，返回结构化的 MixResult 对象。

        如果传入的是字典，会先转换为 MixInput 数据类。

        Args:
            payload: 混合情绪分析的输入数据，可以是 MixInput 或字典。

        Returns:
            结构化的混合情绪分析结果。
        """
        # 支持传入字典或 MixInput 对象
        item = payload if isinstance(payload, MixInput) else MixInput(**payload)
        # 调用大模型获取原始结果
        raw_result = self.client.analyze(item)
        # 将原始结果校验并构建为 MixResult
        return self._build_result(raw_result)

    def mixRe_dict(self, payload: MixInput | dict[str, Any]) -> dict[str, Any]:
        """执行混合情绪分析，返回字典格式的结果。

        是 mixRe 的便捷版本，直接返回可序列化的字典。
        """
        return self.mixRe(payload).to_dict()

    def _build_result(self, raw_result: dict[str, Any]) -> MixResult:
        """将大模型返回的原始字典校验并构建为 MixResult 对象。

        对每个字段进行类型强转和范围校验，不合法时抛出 ValueError。

        Args:
            raw_result: 大模型返回的原始字典。

        Returns:
            校验通过的 MixResult 对象。

        Raises:
            ValueError: 当字段值不符合约束条件时抛出。
        """
        # 提取并清洗主情绪和次情绪标签
        primary_emotion = str(raw_result.get("primary_emotion", "")).strip()
        secondary_emotion = str(raw_result.get("secondary_emotion", "")).strip()

        # 校验情绪标签是否在合法集合中
        if primary_emotion not in EMOTION_LABELS:
            raise ValueError(f"Invalid primary_emotion from LLM: {primary_emotion!r}")
        if secondary_emotion not in EMOTION_LABELS:
            raise ValueError(f"Invalid secondary_emotion from LLM: {secondary_emotion!r}")

        # 校验和转换混合比例
        mix_ratio = self._coerce_mix_ratio(raw_result.get("mix_ratio"), primary_emotion, secondary_emotion)
        # 校验混合场景主情绪强度，范围 0~100
        adjusted_intensity = self._coerce_int(raw_result.get("adjusted_intensity"), "adjusted_intensity")
        if not 0 <= adjusted_intensity <= 100:
            raise ValueError(f"Invalid adjusted_intensity from LLM: {adjusted_intensity!r}")

        # 校验置信度，范围 0~1
        confidence = self._coerce_float(raw_result.get("confidence"), "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence from LLM: {confidence!r}")

        return MixResult(
            is_mixed=self._coerce_bool(raw_result.get("is_mixed"), "is_mixed"),
            primary_emotion=primary_emotion,
            secondary_emotion=secondary_emotion,
            mix_ratio=mix_ratio,
            adjusted_intensity=adjusted_intensity,
            confidence=confidence,
            reason=str(raw_result.get("reason", "")).strip(),
        )

    def _coerce_mix_ratio(
        self,
        value: Any,
        primary_emotion: str,
        secondary_emotion: str,
    ) -> dict[str, float]:
        """校验和转换混合比例字典。

        确保 mix_ratio 是一个非空字典，所有键为合法情绪标签，
        所有值为 0~1 的浮点数，且总和在 0.95~1.05 范围内。

        Args:
            value: 大模型返回的 mix_ratio 原始值。
            primary_emotion: 主情绪标签。
            secondary_emotion: 次情绪标签。

        Returns:
            校验通过的混合比例字典。

        Raises:
            ValueError: 当 mix_ratio 格式或值不符合约束条件时抛出。
        """
        if not isinstance(value, dict) or not value:
            raise ValueError("Invalid mix_ratio from LLM: expected non-empty dict")

        ratio: dict[str, float] = {}
        for emotion, amount in value.items():
            key = str(emotion).strip()
            # 每个键必须是合法的情绪标签
            if key not in EMOTION_LABELS:
                raise ValueError(f"Invalid mix_ratio emotion from LLM: {key!r}")
            ratio[key] = self._coerce_float(amount, "mix_ratio value")
            # 每个比例值必须在 0~1 之间
            if ratio[key] < 0 or ratio[key] > 1:
                raise ValueError(f"Invalid mix_ratio value from LLM: {ratio[key]!r}")

        # 主情绪和次情绪必须存在于比例字典中
        if primary_emotion not in ratio or secondary_emotion not in ratio:
            raise ValueError("Invalid mix_ratio from LLM: missing primary/secondary emotion keys")

        # 比例总和应接近 1，允许 0.05 的浮动
        total = sum(ratio.values())
        if not 0.95 <= total <= 1.05:
            raise ValueError(f"Invalid mix_ratio sum from LLM: {total!r}")

        return ratio

    # def build_messages(self, payload: MixInput | dict[str, Any]) -> list[dict[str, str]]:
    #     """构建发送给大模型的消息列表。

    #     包含系统提示词和用户提示词，用于需要手动管理对话格式的场景。

    #     Args:
    #         payload: 混合情绪分析的输入数据。

    #     Returns:
    #         符合 OpenAI 对话格式的消息列表。
    #     """
    #     item = payload if isinstance(payload, MixInput) else MixInput(**payload)
    #     return [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": build_mix_user_prompt(item)},
    #     ]

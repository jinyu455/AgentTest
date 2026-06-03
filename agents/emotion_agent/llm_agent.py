"""情绪代理的核心逻辑模块。

定义了情绪代理的系统提示词（SYSTEM_PROMPT）和 EmotionAgent 类。
情绪代理负责进行"表层情绪判断"，不做反讽修正和混合情绪融合，
仅根据文本表面表达输出结果，后续交给 Sarcasm Agent 或 Mix Agent 处理。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

from base.coerce import CoercionMixin
from base.schemas import EMOTION_LABELS, BaseTextInput
from .schemas import EmotionInput, EmotionResult


# 情绪代理的系统提示词，定义了大模型的角色、任务、判断规则和输出格式
SYSTEM_PROMPT = """你是情绪识别系统中的 Emotion Agent。

你的任务是做"表层情绪判断"，不要负责反讽修正，也不要负责复杂混合情绪融合。
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

主情绪标签只能从以下 9 类中选择：
- 开心
- 悲伤
- 愤怒
- 焦虑
- 厌烦
- 中性
- 疲惫
- 失落
- 无奈

判断规则：
- emotion 只能输出上述标签之一。
- intensity 是 0 到 100 的整数。中性通常为 0 到 30；明显但不强烈的情绪通常为 40 到 65；强烈情绪通常为 66 到 100。
- confidence 是 0 到 1 的小数。
- tokens 应尽量保留中文词语、短语、标点外的语义片段。
- emotion_words 只放直接表达情绪或明显情绪方向的词语。
- degree_words 放"很、特别、太、稍微、有点、极其、非常"等程度修饰。
- negation_words 放"不、没、没有、别、无、并非"等否定表达。
- contrast_words 放"但、但是、不过、然而、却、只是、可、虽然"等转折表达。
- reason 用一句中文解释表层判断依据，不要超过 100 字。

输出要求：
- 只返回 JSON
- 不要输出 markdown
- 字段必须完整
- 不要增加额外字段

输出格式示例：
{
  "tokens": ["太好了", "周末", "又", "能", "继续", "改", "需求"],
  "emotion_words": ["太好了"],
  "degree_words": [],
  "negation_words": [],
  "contrast_words": [],
  "emotion": "开心",
  "intensity": 62,
  "confidence": 0.61,
  "reason": "文本表面存在明显正向表达"太好了"，情绪方向初步判为正向"
}
"""


def build_emotion_user_prompt(payload: EmotionInput) -> str:
    """构造情绪代理的用户提示词，将输入数据序列化为可读 JSON。

    同时供 client.py 和 Agent.build_messages() 复用，避免重复构建。
    """
    return (
        "请对下面这条消息做表层情绪识别，并严格返回 JSON 结果。\n\n"
        f"{json.dumps(asdict(payload), ensure_ascii=False, indent=2)}"
    )


class EmotionLLMClient(Protocol):
    """情绪代理 LLM 客户端的协议接口。

    定义了 analyze 方法的签名，任何情绪代理使用的 LLM 客户端
    都需要实现此接口，确保可替换性和类型安全。
    """
    def analyze(self, payload: EmotionInput) -> dict[str, Any]:
        """发送情绪分析输入到大模型并返回解析后的 JSON 结果。"""


class EmotionAgent(CoercionMixin):
    """基于大语言模型的表层情绪代理。

    负责接收输入消息，通过 LLM 进行表层情绪判断，并将原始结果
    校验和清洗为结构化的 EmotionResult 对象。
    继承 CoercionMixin 以获得类型强转工具方法（如 _coerce_int、_coerce_float 等）。
    """

    def __init__(self, client: EmotionLLMClient) -> None:
        """初始化情绪代理。

        Args:
            client: 实现了 EmotionLLMClient 协议的 LLM 客户端实例。
        """
        self.client = client

    def emotionRe(self, payload: EmotionInput | dict[str, Any]) -> EmotionResult:
        """对输入消息执行表层情绪分析。

        Args:
            payload: 情绪分析输入数据，可以是 EmotionInput 实例或字典。

        Returns:
            校验后的情绪分析结果。
        """
        # 支持字典输入，自动转换为 EmotionInput 数据类
        item = payload if isinstance(payload, EmotionInput) else EmotionInput(**payload)
        # 调用大模型客户端获取原始结果
        raw_result = self.client.analyze(item)
        # 对原始结果进行校验和类型转换
        return self._build_result(raw_result)

    def emotionRe_dict(self, payload: EmotionInput | dict[str, Any]) -> dict[str, Any]:
        """对输入消息执行表层情绪分析，返回字典格式结果。

        适用于需要将结果进行 JSON 序列化或传递给非 Python 上下文的场景。
        """
        return self.emotionRe(payload).to_dict()

    def _build_result(self, raw_result: dict[str, Any]) -> EmotionResult:
        """将大模型的原始返回结果校验并构建为 EmotionResult。

        对 emotion 标签进行合法性校验，确保其在预定义的 9 类标签中；
        对 intensity 和 confidence 进行范围校验；
        对所有列表字段进行类型强转。

        Args:
            raw_result: 大模型返回的原始字典。

        Returns:
            校验通过的 EmotionResult 对象。

        Raises:
            ValueError: 当 emotion 标签不在允许范围，或数值超出有效区间时抛出。
        """
        # 校验主情绪标签是否在 9 类预定义标签中
        emotion = str(raw_result.get("emotion", "")).strip()
        if emotion not in EMOTION_LABELS:
            raise ValueError(f"Invalid emotion from LLM: {emotion!r}")

        # 校验情绪强度在 0-100 范围内
        intensity = self._coerce_int(raw_result.get("intensity"), "intensity")
        if not 0 <= intensity <= 100:
            raise ValueError(f"Invalid intensity from LLM: {intensity!r}")

        # 校验置信度在 0-1 范围内
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

    # def build_messages(self, payload: EmotionInput | dict[str, Any]) -> list[dict[str, str]]:
    #     """构建发送给大模型的完整消息列表（调试/预览用）。

    #     将系统提示词和用户提示词组合为标准的聊天消息格式，
    #     便于开发者查看实际发送给大模型的内容。
    #     """
    #     item = payload if isinstance(payload, EmotionInput) else EmotionInput(**payload)
    #     return [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": build_emotion_user_prompt(item)},
    #     ]

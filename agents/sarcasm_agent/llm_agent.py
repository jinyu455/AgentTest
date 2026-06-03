"""反讽代理的核心逻辑模块。

定义了反讽代理的系统提示词（SYSTEM_PROMPT）和 SarcasmAgent 类。
反讽代理负责专门识别"反讽表达"，当路由代理判断 need_sarcasm_check=true
时被调用，对情绪代理的表层判断结果进行反讽修正。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

from base.coerce import CoercionMixin
from base.schemas import EMOTION_LABELS
from .schemas import SarcasmInput, SarcasmResult


# 反讽代理的系统提示词，定义了大模型的角色、任务、反讽判断规则和输出格式
SYSTEM_PROMPT = """你是情绪识别系统中的 Sarcasm Agent。

你的任务是专门识别"反讽表达"，不要输出与任务无关的信息。
该模块通常在 Router 给出 need_sarcasm_check=true 时被调用。

你需要一次性完成：
1. 判断是否反讽 is_sarcasm
2. 给出句面情绪 surface_emotion（按表层词面判断）
3. 给出真实情绪 true_emotion（结合语境修正后）
4. 给出修正后的强度 revised_intensity
5. 给出置信度 confidence
6. 给出简短解释 reason

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


反讽判断重点：
- 正向词 + 负向事件
- 夸张赞美 + 抱怨语境
- 重复受害信号（如"又"）
- 负面场景（加班、改需求、被催、深夜开会等）

输出规则：
- surface_emotion / true_emotion 只能从上述标签中选择
- revised_intensity 是 0 到 100 的整数
- confidence 是 0 到 1 的小数
- reason 用一句中文解释，不超过 90 字

输出要求：
- 只返回 JSON
- 不要输出 markdown
- 字段必须完整
- 不要增加额外字段

输出格式：
{
  "is_sarcasm": true,
  "surface_emotion": "开心",
  "true_emotion": "厌烦",
  "revised_intensity": 74,
  "confidence": 0.85,
  "reason": "表面正向词与负面工作场景形成反差，真实情绪更偏厌烦"
}
"""


def build_sarcasm_user_prompt(payload: SarcasmInput) -> str:
    """构造反讽代理的用户提示词，将输入数据序列化为可读 JSON。

    同时供 client.py 和 Agent.build_messages() 复用，避免重复构建。
    """
    return (
        "请判断下面这条消息是否反讽，并返回 JSON 结果。\n\n"
        f"{json.dumps(asdict(payload), ensure_ascii=False, indent=2)}"
    )


class SarcasmLLMClient(Protocol):
    """反讽代理 LLM 客户端的协议接口。

    定义了 analyze 方法的签名，任何反讽代理使用的 LLM 客户端
    都需要实现此接口，确保可替换性和类型安全。
    """
    def analyze(self, payload: SarcasmInput) -> dict[str, Any]:
        """发送反讽检测输入到大模型并返回解析后的 JSON 结果。"""


class SarcasmAgent(CoercionMixin):
    """基于大语言模型的反讽检测代理。

    负责接收输入消息，通过 LLM 进行反讽识别和情绪修正，
    并将原始结果校验和清洗为结构化的 SarcasmResult 对象。
    继承 CoercionMixin 以获得类型强转工具方法。
    """

    def __init__(self, client: SarcasmLLMClient) -> None:
        """初始化反讽代理。

        Args:
            client: 实现了 SarcasmLLMClient 协议的 LLM 客户端实例。
        """
        self.client = client

    def detect(self, payload: SarcasmInput | dict[str, Any]) -> SarcasmResult:
        """对输入消息执行反讽检测。

        Args:
            payload: 反讽检测输入数据，可以是 SarcasmInput 实例或字典。

        Returns:
            校验后的反讽检测结果。
        """
        # 支持字典输入，自动转换为 SarcasmInput 数据类
        item = payload if isinstance(payload, SarcasmInput) else SarcasmInput(**payload)
        # 调用大模型客户端获取原始结果
        raw_result = self.client.analyze(item)
        # 对原始结果进行校验和类型转换
        return self._build_result(raw_result)

    def detect_dict(self, payload: SarcasmInput | dict[str, Any]) -> dict[str, Any]:
        """对输入消息执行反讽检测，返回字典格式结果。

        适用于需要将结果进行 JSON 序列化或传递给非 Python 上下文的场景。
        """
        return self.detect(payload).to_dict()

    def _build_result(self, raw_result: dict[str, Any]) -> SarcasmResult:
        """将大模型的原始返回结果校验并构建为 SarcasmResult。

        对 surface_emotion 和 true_emotion 进行合法性校验，
        确保其在预定义的 9 类情绪标签中；
        对 revised_intensity 和 confidence 进行范围校验。

        Args:
            raw_result: 大模型返回的原始字典。

        Returns:
            校验通过的 SarcasmResult 对象。

        Raises:
            ValueError: 当情绪标签不在允许范围，或数值超出有效区间时抛出。
        """
        # 校验句面情绪标签是否合法
        surface_emotion = str(raw_result.get("surface_emotion", "")).strip()
        true_emotion = str(raw_result.get("true_emotion", "")).strip()
        if surface_emotion not in EMOTION_LABELS:
            raise ValueError(f"Invalid surface_emotion from LLM: {surface_emotion!r}")
        if true_emotion not in EMOTION_LABELS:
            raise ValueError(f"Invalid true_emotion from LLM: {true_emotion!r}")

        # 校验修正后强度在 0-100 范围内
        revised_intensity = self._coerce_int(raw_result.get("revised_intensity"), "revised_intensity")
        if not 0 <= revised_intensity <= 100:
            raise ValueError(f"Invalid revised_intensity from LLM: {revised_intensity!r}")

        # 校验置信度在 0-1 范围内
        confidence = self._coerce_float(raw_result.get("confidence"), "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence from LLM: {confidence!r}")

        return SarcasmResult(
            is_sarcasm=self._coerce_bool(raw_result.get("is_sarcasm"), "is_sarcasm"),
            surface_emotion=surface_emotion,
            true_emotion=true_emotion,
            revised_intensity=revised_intensity,
            confidence=confidence,
            reason=str(raw_result.get("reason", "")).strip(),
        )

    # def build_messages(self, payload: SarcasmInput | dict[str, Any]) -> list[dict[str, str]]:
    #     """构建发送给大模型的完整消息列表（调试/预览用）。

    #     将系统提示词和用户提示词组合为标准的聊天消息格式，
    #     便于开发者查看实际发送给大模型的内容。
    #     """
    #     item = payload if isinstance(payload, SarcasmInput) else SarcasmInput(**payload)
    #     return [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": build_sarcasm_user_prompt(item)},
    #     ]

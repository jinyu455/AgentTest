"""Mix Agent 的 HTTP 客户端实现。

通过 OpenAI 兼容的 HTTP 接口调用大语言模型，进行混合情绪分析。
"""

from __future__ import annotations

from typing import Any

from base.base_client import BaseHTTPLLMClient
from .llm_agent import SYSTEM_PROMPT, build_mix_user_prompt
from .schemas import MixInput


class HTTPMixLLMClient(BaseHTTPLLMClient):
    """基于 HTTP 的 Mix Agent 大模型客户端。

    继承 BaseHTTPLLMClient，使用 OpenAI 兼容协议与大模型通信，
    专门用于混合情绪（mix）分析场景。
    """

    def analyze(self, payload: MixInput) -> dict[str, Any]:
        """对输入文本进行混合情绪分析。

        将 MixInput 序列化为 JSON 作为用户提示词，配合系统提示词
        调用大模型，返回原始 JSON 结果。

        Args:
            payload: 混合情绪分析的输入数据。

        Returns:
            大模型返回的原始字典结果，包含 is_mixed、primary_emotion 等字段。
        """
        # 复用 llm_agent 中的 prompt 构建函数
        user_prompt = build_mix_user_prompt(payload)
        # 使用较低的 temperature 以获得更稳定的输出
        return self._call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.1)

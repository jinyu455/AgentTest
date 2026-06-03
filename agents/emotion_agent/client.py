"""情绪代理的 HTTP LLM 客户端。

封装了与 OpenAI 兼容 API 的通信逻辑，负责将情绪分析输入
序列化为提示词并发送给大语言模型，获取表层情绪识别结果。
"""

from __future__ import annotations

from typing import Any

from base.base_client import BaseHTTPLLMClient
from .llm_agent import SYSTEM_PROMPT, build_emotion_user_prompt
from .schemas import EmotionInput


class HTTPEmotionLLMClient(BaseHTTPLLMClient):
    """基于 HTTP 的情绪代理大模型客户端。

    继承自 BaseHTTPLLMClient，复用其 LLM 连接管理能力，
    提供 analyze 方法将消息发送给大模型进行情绪分析。
    """

    def analyze(self, payload: EmotionInput) -> dict[str, Any]:
        """将情绪分析输入发送给大模型，返回原始 JSON 结果。

        Args:
            payload: 待分析的情绪输入数据。

        Returns:
            大模型返回的原始字典结果，包含 emotion、intensity 等字段。
        """
        # 复用 llm_agent 中的 prompt 构建函数
        user_prompt = build_emotion_user_prompt(payload)
        # 使用较低的 temperature 确保情绪判断结果稳定
        return self._call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.1)

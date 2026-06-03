"""Chat Agent 的 HTTP 客户端实现。

通过 OpenAI 兼容的 HTTP 接口调用大语言模型，生成情绪聊天回复。
"""

from __future__ import annotations

from typing import Any

from base.base_client import BaseHTTPLLMClient
from .llm_agent import SYSTEM_PROMPT, build_chat_user_prompt
from .schemas import ChatInput


class HTTPChatLLMClient(BaseHTTPLLMClient):
    """基于 HTTP 的 Chat Agent 大模型客户端。

    继承 BaseHTTPLLMClient，使用 OpenAI 兼容协议与大模型通信，
    专门用于情绪聊天回复的生成场景。
    """

    def generate(self, payload: ChatInput) -> dict[str, Any]:
        """根据聊天输入生成回复。

        使用 build_chat_user_prompt 构建用户提示词，配合系统提示词
        调用大模型，返回原始 JSON 结果。

        Args:
            payload: 聊天输入数据，包含用户消息、历史记录和情绪分析结果。

        Returns:
            大模型返回的原始字典结果，包含 reply、tone、risk_hint 等字段。
        """
        # 使用模块级函数构建用户提示词，便于在 Agent 中复用
        user_prompt = build_chat_user_prompt(payload)
        # 使用稍高的 temperature（0.4）以获得更自然、多样化的回复
        return self._call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.4)

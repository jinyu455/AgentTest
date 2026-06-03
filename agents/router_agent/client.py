"""路由代理的 HTTP LLM 客户端。

封装了与 OpenAI 兼容 API 的通信逻辑，负责将路由输入
序列化为提示词并发送给大语言模型，获取分类结果。
"""

from __future__ import annotations

from typing import Any

from base.base_client import BaseHTTPLLMClient
from .llm_agent import SYSTEM_PROMPT, build_router_user_prompt
from .schemas import RouterInput


class HTTPRouterLLMClient(BaseHTTPLLMClient):
    """基于 HTTP 的路由代理大模型客户端。

    继承自 BaseHTTPLLMClient，复用其 LLM 连接管理能力，
    提供 classify 方法将消息发送给大模型进行路由分类。
    """

    def classify(self, payload: RouterInput) -> dict:
        """将路由输入发送给大模型，返回原始 JSON 分类结果。

        Args:
            payload: 待分类的路由输入数据。

        Returns:
            大模型返回的原始字典结果，包含 sample_type 等字段。
        """
        # 复用 llm_agent 中的 prompt 构建函数
        user_prompt = build_router_user_prompt(payload)
        # 使用较低的 temperature 确保分类结果稳定一致
        return self._call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.1)

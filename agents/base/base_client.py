"""HTTP 客户端基类模块。

提供面向 LLM API 的基础客户端类和响应解析工具函数。
所有需要调用 LLM 服务的 Agent 客户端应继承 BaseHTTPLLMClient。
"""

from __future__ import annotations

import json
from typing import Any

from llm_http import post_json_with_retries
from .llm_config import LLMConfig


def extract_llm_result(raw_text: str) -> dict[str, Any]:
    """从 LLM 的原始响应文本中提取并解析 JSON 结果。

    处理 OpenAI 兼容格式的响应结构，支持两种 content 格式：
    - 纯字符串格式：直接解析为 JSON
    - 多模态列表格式：提取其中 type=="text" 的部分拼接后解析

    Args:
        raw_text: LLM 返回的原始 HTTP 响应体文本

    Returns:
        解析后的字典对象，通常是 LLM 结构化输出的 JSON

    Raises:
        json.JSONDecodeError: 当响应内容无法解析为有效 JSON 时
        KeyError: 当响应结构不符合预期格式时（如缺少 choices/message 等字段）
    """
    data = json.loads(raw_text)
    content = data["choices"][0]["message"]["content"]
    # 处理多模态响应格式：content 可能是文本片段列表而非纯字符串
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
        content = "".join(text_parts)
    return json.loads(content)


class BaseHTTPLLMClient:
    """基于 HTTP 的 LLM 客户端基类。

    封装了与 LLM API 交互的通用逻辑，子类只需继承并实现
    具体的业务方法（如情绪分析、讽刺检测等）。

    该类负责：
    - 持有 LLM 配置信息（URL、密钥、模型等）
    - 构造标准的 Chat Completions 请求体
    - 调用 HTTP 工具函数发送请求
    - 解析并返回结构化的 JSON 结果
    """

    def __init__(self, config: LLMConfig) -> None:
        """初始化 LLM 客户端。

        Args:
            config: LLM 连接配置对象，包含 API 端点、密钥等信息
        """
        self.config = config

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """向 LLM 发送一次对话请求并返回解析后的 JSON 结果。

        构造 OpenAI 兼容格式的 Chat Completions 请求体，使用
        response_format 强制要求 LLM 返回 JSON 格式内容。

        Args:
            system_prompt: 系统提示词，用于设定 LLM 的角色和行为约束
            user_prompt: 用户提示词，包含需要处理的具体内容
            temperature: 生成温度参数，值越低输出越确定性，默认 0.1

        Returns:
            解析后的 JSON 字典，结构取决于具体的 system_prompt 设计

        Raises:
            HTTPError: 当 API 请求返回非可重试错误码时
            json.JSONDecodeError: 当 LLM 返回的内容无法解析为 JSON 时
        """
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        # 使用带重试机制的 HTTP POST 请求发送 API 调用
        raw_text = post_json_with_retries(
            self.config.base_url,
            body,
            self.config.api_key,
            self.config.timeout_seconds,
        )
        return extract_llm_result(raw_text)

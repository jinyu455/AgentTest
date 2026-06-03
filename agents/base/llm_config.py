"""LLM 配置数据类模块。

使用 dataclass 定义大语言模型服务的连接配置，
包括 API 端点地址、认证密钥、模型名称和超时时间。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LLMConfig:
    """LLM 服务连接配置数据类。

    集中管理调用大语言模型 API 所需的各项参数，
    通过 dataclass 自动生成 __init__、__eq__ 等方法。

    Attributes:
        base_url: LLM API 的完整端点 URL（Chat Completions 接口）
        api_key: API 认证密钥，用于 Bearer Token 认证
        model: 模型名称标识符，如 "gpt-4"、"deepseek-chat" 等
        timeout_seconds: 单次 HTTP 请求的超时时间（秒），默认 30 秒
    """
    base_url: str = "https://your-llm-service.example.com/v1/chat/completions"
    api_key: str = "YOUR_API_KEY"
    model: str = "YOUR_MODEL_NAME"
    timeout_seconds: int = 30

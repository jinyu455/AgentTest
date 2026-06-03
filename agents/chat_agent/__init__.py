"""Chat Agent 包的初始化模块。

导出聊天回复生成相关的核心类和配置，包括 HTTP 客户端、
Agent 实例、输入输出数据结构。
"""

from .client import HTTPChatLLMClient
from .llm_agent import ChatAgent
from .schemas import ChatInput, ChatMessage, ChatResult

__all__ = [
    "HTTPChatLLMClient",   # 基于 HTTP 的大模型客户端
    "ChatAgent",           # 聊天回复生成 Agent
    "ChatInput",           # 输入数据结构
    "ChatMessage",         # 单条对话消息结构
    "ChatResult",          # 输出结果数据结构
]

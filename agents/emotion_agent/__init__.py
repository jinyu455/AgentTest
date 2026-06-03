"""情绪代理（Emotion Agent）模块的公开接口。

导出情绪代理所需的客户端、Agent 类和数据模型，
供上层模块直接从本包导入使用。
"""

from .client import HTTPEmotionLLMClient
from .llm_agent import EmotionAgent
from .schemas import EmotionInput, EmotionResult

__all__ = [
    "HTTPEmotionLLMClient",
    "EmotionAgent",
    "EmotionInput",
    "EmotionResult",
]

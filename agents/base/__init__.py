"""基础模块导出包。

统一导出 agents/base 下的核心组件，包括：
- BaseHTTPLLMClient: HTTP 客户端基类
- CoercionMixin: 类型强制转换混入类
- LLMConfig: LLM 配置数据类
- EMOTION_LABELS: 情绪标签常量集合
- BaseTextInput: 统一输入 schema 数据类
"""

from .base_client import BaseHTTPLLMClient
from .coerce import CoercionMixin
from .llm_config import LLMConfig
from .schemas import EMOTION_LABELS, BaseTextInput

# 公开接口列表，控制 from agents.base import * 的导出内容
__all__ = [
    "BaseHTTPLLMClient",
    "CoercionMixin",
    "LLMConfig",
    "EMOTION_LABELS",
    "BaseTextInput",
]

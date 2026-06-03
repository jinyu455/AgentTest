"""反讽代理（Sarcasm Agent）模块的公开接口。

导出反讽代理所需的客户端、Agent 类和数据模型，
供上层模块直接从本包导入使用。
"""

from .client import HTTPSarcasmLLMClient
from .llm_agent import SarcasmAgent
from .schemas import SarcasmInput, SarcasmResult

__all__ = [
    "HTTPSarcasmLLMClient",
    "SarcasmAgent",
    "SarcasmInput",
    "SarcasmResult",
]

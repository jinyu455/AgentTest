from __future__ import annotations

from typing import Any

from base.base_client import BaseHTTPLLMClient
from .llm_agent import SYSTEM_PROMPT, build_profile_user_prompt


class HTTPProfileLLMClient(BaseHTTPLLMClient):
    """基于 OpenAI 兼容 API 的 Profile Agent HTTP 客户端。

    继承 BaseHTTPLLMClient，复用 _call_llm() 的 HTTP 调用和重试逻辑。
    """

    def analyze(self, features: dict[str, Any], chat_summary: str) -> dict[str, Any]:
        """调用 LLM 生成用户画像。

        参数:
            features:     extract_features() 提取的统计特征字典
            chat_summary: 格式化后的对话历史摘要

        返回:
            LLM 返回的 JSON 字典，包含 personality_traits, communication_style,
            emotional_patterns, mbti, summary 五个字段。
        """
        # 复用 llm_agent 中的 prompt 构建函数
        user_prompt = build_profile_user_prompt(features, chat_summary)
        # temperature=0.3：画像生成需要一定创造性，但不宜过于随机
        return self._call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.3)

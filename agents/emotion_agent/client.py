from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from llm_http import post_json_with_retries
from .llm_agent import SYSTEM_PROMPT
from .schemas import EmotionInput


@dataclass(slots=True)
class LLMConfig:
    base_url: str = "https://your-llm-service.example.com/v1/chat/completions"
    api_key: str = "YOUR_API_KEY"
    model: str = "YOUR_MODEL_NAME"
    timeout_seconds: int = 30


class HTTPEmotionLLMClient:
    """Generic OpenAI-compatible client with placeholder config."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def analyze(self, payload: EmotionInput) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(payload)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        raw_text = post_json_with_retries(
            self.config.base_url,
            body,
            self.config.api_key,
            self.config.timeout_seconds,
        )

        return self._extract_result(raw_text)

    def _build_user_prompt(self, payload: EmotionInput) -> str:
        return (
            "请对下面这条消息做表层情绪识别，并严格返回 JSON 结果。\n\n"
            f"{json.dumps(asdict(payload), ensure_ascii=False, indent=2)}"
        )

    def _extract_result(self, raw_text: str) -> dict[str, Any]:
        data = json.loads(raw_text)
        content = data["choices"][0]["message"]["content"]

        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
            content = "".join(text_parts)

        return json.loads(content)

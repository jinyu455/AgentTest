from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any
from urllib import request

from .llm_agent import SYSTEM_PROMPT
from .schemas import JudgeInput, JudgeResult


@dataclass(slots=True)
class LLMConfig:
    base_url: str = "https://your-llm-service.example.com/v1/chat/completions"
    api_key: str = "YOUR_API_KEY"
    model: str = "YOUR_MODEL_NAME"
    timeout_seconds: int = 30


class HTTPJudgeLLMClient:
    """Generic OpenAI-compatible client for Judge Agent arbitration."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def arbitrate(self, payload: JudgeInput, rule_result: JudgeResult) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(payload, rule_result)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            url=self.config.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )

        with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
            raw_text = response.read().decode("utf-8")

        return self._extract_result(raw_text)

    def _build_user_prompt(self, payload: JudgeInput, rule_result: JudgeResult) -> str:
        body = {
            "payload": asdict(payload),
            "rule_result": rule_result.to_dict(),
        }
        return (
            "请审核以下情感流水线分析结果，并返回最终的判定结果JSON 数据。\n\n"
            f"{json.dumps(body, ensure_ascii=False, indent=2)}"
        )

    def _extract_result(self, raw_text: str) -> dict[str, Any]:
        data = json.loads(raw_text)
        content = data["choices"][0]["message"]["content"]

        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
            content = "".join(text_parts)

        return json.loads(content)

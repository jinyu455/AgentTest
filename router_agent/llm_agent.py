from __future__ import annotations

import json
from dataclasses import asdict
from typing import Protocol

from .schemas import RouterInput, RouterResult


SYSTEM_PROMPT = """你是情绪识别系统中的 Router Agent。

你的任务只有两个：
1. 判断输入句子的表达类型，只能输出以下三类之一：
- direct
- sarcasm_suspected
- mix
2. 决定是否需要调用后续模块：
- need_sarcasm_check
- need_mix_check

分类规则：
1. direct
- 明显直接情绪表达
- 没有明显转折
- 没有明显反讽结构
- 没有明显复合情绪

2. sarcasm_suspected
- 句面正向，语境负向
- 或夸张正向词和明显糟糕事件并存
- 常见触发：又、还真是、真棒、太好了

3. mix
- 有转折词或复合表达
- 有两个情绪方向
- 情绪模糊，不适合单标签
- 低能量、压抑、说不上来、提不起劲等表达

输出要求：
- 只能返回 JSON
- 不要输出 markdown
- 字段必须完整
- evidence 是字符串数组，列出支持判断的线索

输出格式：
{
  "sample_type": "direct | sarcasm_suspected | mix",
  "need_sarcasm_check": true,
  "need_mix_check": false,
  "routing_reason": "简洁说明原因",
  "evidence": ["线索1", "线索2"]
}
"""


class RouterLLMClient(Protocol):
    def classify(self, payload: RouterInput) -> dict:
        """Send payload to an LLM and return the parsed JSON result."""


class RouterAgent:
    """LLM-based router agent."""

    def __init__(self, client: RouterLLMClient) -> None:
        self.client = client

    def route(self, payload: RouterInput | dict) -> RouterResult:
        item = payload if isinstance(payload, RouterInput) else RouterInput(**payload)
        raw_result = self.client.classify(item)
        return self._build_result(raw_result)

    def route_dict(self, payload: RouterInput | dict) -> dict:
        return self.route(payload).to_dict()

    def _build_result(self, raw_result: dict) -> RouterResult:
        sample_type = str(raw_result.get("sample_type", "")).strip()
        if sample_type not in {"direct", "sarcasm_suspected", "mix"}:
            raise ValueError(f"Invalid sample_type from LLM: {sample_type!r}")

        return RouterResult(
            sample_type=sample_type,
            need_sarcasm_check=bool(raw_result.get("need_sarcasm_check")),
            need_mix_check=bool(raw_result.get("need_mix_check")),
            routing_reason=str(raw_result.get("routing_reason", "")).strip(),
            evidence=[str(item) for item in raw_result.get("evidence", [])],
        )
 
    #以下的函数只是为了让我们看到发给大模型的message
    def build_messages(self, payload: RouterInput | dict) -> list[dict[str, str]]:
        item = payload if isinstance(payload, RouterInput) else RouterInput(**payload)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(item)},
        ]

    def _build_user_prompt(self, payload: RouterInput) -> str:
        return (
            "请判断下面这条消息的路由类型，并给出 JSON 结果。\n\n"
            f"{json.dumps(asdict(payload), ensure_ascii=False, indent=2)}"
        )
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from judge_agent import HTTPJudgeLLMClient, JudgeAgent, LLMConfig


def load_api_key() -> str:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "API_KEY":
                api_key = value.strip().strip('"').strip("'")
                if api_key:
                    return api_key

    api_key = os.getenv("API_KEY", "").strip()
    if api_key:
        return api_key

    raise RuntimeError("API_KEY not found. Please set it in .env or environment variables.")

def main() -> None:

    payload = {
        "text": "太好了，周末又能继续改需求了。",
        "router_result": {
            "sample_type": "sarcasm_suspected",
            "need_sarcasm_check": True,
            "need_mix_check": False,
            "routing_reason": "表面正向，但工作语境偏负向，疑似反讽。",
            "evidence": ["太好了", "周末继续改需求"],
        },
        "emotion_result": {
            "emotion": "开心",
            "intensity": 62,
            "confidence": 0.72,
            "reason": "文本表面包含明显正向表达。",
        },
        "sarcasm_result": {
            "is_sarcasm": True,
            "surface_emotion": "开心",
            "true_emotion": "厌烦",
            "revised_intensity": 74,
            "confidence": 0.86,
            "reason": "正向词与负面工作场景形成反差，真实情绪更偏厌烦。",
        },
        "mix_result": None,
    }

    config = LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1/chat/completions"),
        api_key=load_api_key(),
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
    )

    agent = JudgeAgent(client=HTTPJudgeLLMClient(config))

    print("【Judge 输入】")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    print("\n【将要发送给 Judge LLM 的 messages】")
    rule_only_agent = JudgeAgent()
    rule_result = rule_only_agent.judge(payload)
    print(json.dumps(agent.build_messages(payload, rule_result), ensure_ascii=False, indent=2))

    print("\n" + "=" * 50)
    print("正在执行混合 Judge：" )
    print("=" * 50)

    result = agent.judge_dict(payload)

    print("\n【Judge 输出】")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from router_agent import HTTPRouterLLMClient, LLMConfig, RouterAgent


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
        "id": "msg_001",
        "user_id": "u_1001",
        "text": "太好了，周末又能继续改需求了。",
        "source": "chat",
        "created_at": "2026-03-24T14:00:00",
    }

    config = LLMConfig(
        base_url="https://api.deepseek.com/v1/chat/completions",
        api_key=load_api_key(),
        model="deepseek-chat",
    )
    client = HTTPRouterLLMClient(config)
    agent = RouterAgent(client=client)

    print("下面是将要发送给大模型的 messages：")
    print(json.dumps(agent.build_messages(payload), ensure_ascii=False, indent=2))

    print("\n" + "=" * 50)
    print("正在调用大模型，请稍候...")
    print("=" * 50)

    result = agent.route_dict(payload)

    print("\n【大模型返回的router结果】")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

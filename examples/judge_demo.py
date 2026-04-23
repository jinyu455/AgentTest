from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from judge_agent import JudgeAgent


def main() -> None:
    payload = {
        "router_result": {
            "sample_type": "sarcasm_suspected",
            "need_sarcasm_check": True,
            "need_mix_check": False,
            "routing_reason": "表面正向，语境负向，疑似反讽。",
            "evidence": ["太好了", "周末继续改需求"],
        },
        "emotion_result": {
            "emotion": "开心",
            "intensity": 62,
            "confidence": 0.61,
            "reason": "文本表面存在明显正向表达“太好了”。",
        },
        "sarcasm_result": {
            "is_sarcasm": True,
            "surface_emotion": "开心",
            "true_emotion": "厌烦",
            "revised_intensity": 74,
            "confidence": 0.85,
            "reason": "正向词与负面工作场景形成反差，真实情绪更偏厌烦。",
        },
        "mix_result": None,
    }

    agent = JudgeAgent()
    result = agent.judge_dict(payload)

    print("【Judge 输入】")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("\n【Judge 输出】")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

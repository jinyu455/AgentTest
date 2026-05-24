from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS_ROOT))

from chat_agent import ChatAgent
from chat_agent.schemas import ChatInput


class FakeChatLLMClient:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.last_payload: ChatInput | None = None

    def generate(self, payload: ChatInput) -> dict:
        self.last_payload = payload
        return self.result


class ChatAgentTestCase(unittest.TestCase):
    def test_chat_returns_expected_shape(self) -> None:
        client = FakeChatLLMClient(
            {
                "reply": "听起来你其实挺疲惫，也有点无奈。周末还被需求占着，确实会让人烦。",
                "tone": "supportive",
                "risk_hint": "none",
                "suggested_actions": ["先把最急的事列出来", "给自己留一点休息时间"],
                "reason": "用户文本包含反讽和工作压力，适合支持性回应。",
            }
        )
        agent = ChatAgent(client=client)

        result = agent.chat_dict(
            {
                "text": "太好了，周末又能继续改需求了。",
                "user_id": "u_1001",
                "judge_result": {
                    "final_emotion": "厌烦",
                    "final_intensity": 74,
                    "final_confidence": 0.86,
                    "is_sarcasm": True,
                    "is_mixed": False,
                },
            }
        )

        self.assertEqual(result["tone"], "supportive")
        self.assertEqual(result["risk_hint"], "none")
        self.assertIn("疲惫", result["reply"])
        self.assertEqual(client.last_payload.text, "太好了，周末又能继续改需求了。")

    def test_build_messages_contains_text_and_judge_result(self) -> None:
        client = FakeChatLLMClient(
            {
                "reply": "我在。",
                "tone": "calm",
                "risk_hint": "none",
                "suggested_actions": [],
                "reason": "简短陪伴。",
            }
        )
        agent = ChatAgent(client=client)

        messages = agent.build_messages(
            {
                "text": "今天有点累。",
                "judge_result": {"final_emotion": "疲惫", "final_intensity": 60},
            }
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("今天有点累", messages[1]["content"])
        self.assertIn("judge_result", messages[1]["content"])

    def test_build_messages_formats_history_for_context(self) -> None:
        client = FakeChatLLMClient(
            {
                "reply": "可以，我们先把今晚的代码收个尾。",
                "tone": "supportive",
                "risk_hint": "none",
                "suggested_actions": [],
                "reason": "结合上一轮疲惫和当前建议请求。",
            }
        )
        agent = ChatAgent(client=client)

        messages = agent.build_messages(
            {
                "text": "能给我一些建议吗",
                "history": [
                    {"role": "user", "content": "太忙了，又改了一天的代码"},
                    {"role": "assistant", "content": "忙了一天改代码确实很累，今晚早点休息。"},
                ],
                "judge_result": {"final_emotion": "疲惫", "final_intensity": 65},
            }
        )

        prompt = messages[1]["content"]
        self.assertIn("最近对话历史", prompt)
        self.assertIn("用户: 太忙了，又改了一天的代码", prompt)
        self.assertIn("当前用户消息", prompt)
        self.assertIn("能给我一些建议吗", prompt)

    def test_invalid_tone_raises(self) -> None:
        client = FakeChatLLMClient(
            {
                "reply": "我懂。",
                "tone": "funny",
                "risk_hint": "none",
                "suggested_actions": [],
                "reason": "",
            }
        )
        agent = ChatAgent(client=client)

        with self.assertRaises(ValueError):
            agent.chat({"text": "今天好累。"})


if __name__ == "__main__":
    unittest.main()

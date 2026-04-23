from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mix_agent import MixAgent
from mix_agent.schemas import MixInput


class FakeMixLLMClient:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.last_payload: MixInput | None = None

    def analyze(self, payload: MixInput) -> dict:
        self.last_payload = payload
        return self.result


class MixAgentTestCase(unittest.TestCase):
    def _payload(self, text: str) -> dict:
        return {
            "id": "msg_001",
            "user_id": "u_1001",
            "text": text,
            "source": "chat",
            "created_at": "2026-03-24T14:00:00",
        }

    def test_analyze_mix_emotion_from_llm_result(self) -> None:
        client = FakeMixLLMClient(
            {
                "is_mixed": True,
                "primary_emotion": "疲惫",
                "secondary_emotion": "开心",
                "mix_ratio": {"疲惫": 0.58, "开心": 0.42},
                "revised_intensity": 57,
                "confidence": 0.79,
                "reason": "句子存在转折结构，后半句突出疲惫感。",
            }
        )
        agent = MixAgent(client=client)

        result = agent.mixRe(self._payload("开心是开心，但也挺累"))

        self.assertTrue(result.is_mixed)
        self.assertEqual(result.primary_emotion, "疲惫")
        self.assertEqual(result.secondary_emotion, "开心")
        self.assertEqual(result.revised_intensity, 57)
        self.assertAlmostEqual(result.mix_ratio["疲惫"], 0.58)
        self.assertEqual(client.last_payload.text, "开心是开心，但也挺累")

    def test_analyze_dict_returns_expected_shape(self) -> None:
        client = FakeMixLLMClient(
            {
                "is_mixed": True,
                "primary_emotion": "空虚",
                "secondary_emotion": "开心",
                "mix_ratio": {"空虚": 0.61, "开心": 0.39},
                "revised_intensity": 48,
                "confidence": 0.72,
                "reason": "结束后出现轻松与空落并存。",
            }
        )
        agent = MixAgent(client=client)

        result = agent.mixRe_dict(self._payload("终于结束了，轻松，但有点空"))

        self.assertTrue(result["is_mixed"])
        self.assertEqual(result["primary_emotion"], "空虚")
        self.assertIn("mix_ratio", result)

    def test_build_messages_contains_schema_and_text(self) -> None:
        client = FakeMixLLMClient(
            {
                "is_mixed": False,
                "primary_emotion": "中性",
                "secondary_emotion": "中性",
                "mix_ratio": {"中性": 1.0},
                "revised_intensity": 20,
                "confidence": 0.52,
                "reason": "无明显复合情绪。",
            }
        )
        agent = MixAgent(client=client)

        messages = agent.build_messages(self._payload("今天状态还行。"))

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("今天状态还行。", messages[1]["content"])
        self.assertIn("is_mixed", messages[0]["content"])

    def test_invalid_emotion_raises(self) -> None:
        client = FakeMixLLMClient(
            {
                "is_mixed": True,
                "primary_emotion": "激动",
                "secondary_emotion": "开心",
                "mix_ratio": {"激动": 0.5, "开心": 0.5},
                "revised_intensity": 50,
                "confidence": 0.7,
                "reason": "非法标签。",
            }
        )
        agent = MixAgent(client=client)

        with self.assertRaises(ValueError):
            agent.mixRe(self._payload("还不错，但有点上头"))

    def test_invalid_mix_ratio_sum_raises(self) -> None:
        client = FakeMixLLMClient(
            {
                "is_mixed": True,
                "primary_emotion": "疲惫",
                "secondary_emotion": "开心",
                "mix_ratio": {"疲惫": 0.7, "开心": 0.5},
                "revised_intensity": 60,
                "confidence": 0.8,
                "reason": "比例总和异常。",
            }
        )
        agent = MixAgent(client=client)

        with self.assertRaises(ValueError):
            agent.mixRe(self._payload("开心是开心，但也挺累"))

    def test_invalid_is_mixed_type_raises(self) -> None:
        client = FakeMixLLMClient(
            {
                "is_mixed": "true",
                "primary_emotion": "疲惫",
                "secondary_emotion": "开心",
                "mix_ratio": {"疲惫": 0.58, "开心": 0.42},
                "revised_intensity": 57,
                "confidence": 0.79,
                "reason": "布尔类型错误。",
            }
        )
        agent = MixAgent(client=client)

        with self.assertRaises(ValueError):
            agent.mixRe(self._payload("开心是开心，但也挺累"))


if __name__ == "__main__":
    unittest.main()

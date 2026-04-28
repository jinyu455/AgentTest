from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from judge_agent import JudgeAgent


class FakeJudgeClient:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.calls = 0
        self.result = result or {
            "final_emotion": "negative",
            "secondary_emotion": None,
            "final_intensity": 76,
            "final_confidence": 0.82,
            "is_sarcasm": True,
            "is_mixed": False,
            "reason": "mock llm review",
        }

    def arbitrate(self, payload: Any, rule_result: Any) -> dict[str, Any]:
        self.calls += 1
        return self.result


class JudgeAgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = JudgeAgent(sarcasm_confidence_threshold=0.65, mix_confidence_threshold=0.65)

    def test_direct_uses_emotion_result(self) -> None:
        payload = {
            "router_result": {"sample_type": "direct"},
            "emotion_result": {
                "emotion": "anxiety",
                "intensity": 68,
                "confidence": 0.84,
                "reason": "direct anxiety signal",
            },
        }

        result = self.agent.judge_dict(payload)

        self.assertEqual(result["final_emotion"], "anxiety")
        self.assertIsNone(result["secondary_emotion"])
        self.assertEqual(result["final_intensity"], 68)
        self.assertEqual(result["final_confidence"], 0.84)
        self.assertFalse(result["is_sarcasm"])
        self.assertFalse(result["is_mixed"])

    def test_sarcasm_suspected_uses_sarcasm_when_confident(self) -> None:
        payload = {
            "router_result": {"sample_type": "sarcasm_suspected"},
            "emotion_result": {
                "emotion": "happy",
                "intensity": 61,
                "confidence": 0.72,
                "reason": "surface positive words",
            },
            "sarcasm_result": {
                "is_sarcasm": True,
                "surface_emotion": "happy",
                "true_emotion": "annoyed",
                "revised_intensity": 74,
                "confidence": 0.86,
                "reason": "positive wording conflicts with negative context",
            },
        }

        result = self.agent.judge_dict(payload)

        self.assertEqual(result["final_emotion"], "annoyed")
        self.assertEqual(result["final_intensity"], 74)
        self.assertTrue(result["is_sarcasm"])
        self.assertFalse(result["is_mixed"])

    def test_sarcasm_suspected_falls_back_when_low_confidence(self) -> None:
        payload = {
            "router_result": {"sample_type": "sarcasm_suspected"},
            "emotion_result": {
                "emotion": "happy",
                "intensity": 58,
                "confidence": 0.8,
                "reason": "surface positive words",
            },
            "sarcasm_result": {
                "is_sarcasm": True,
                "surface_emotion": "happy",
                "true_emotion": "annoyed",
                "revised_intensity": 70,
                "confidence": 0.52,
                "reason": "weak evidence",
            },
        }

        result = self.agent.judge_dict(payload)

        self.assertEqual(result["final_emotion"], "happy")
        self.assertEqual(result["final_intensity"], 58)
        self.assertEqual(result["final_confidence"], 0.64)
        self.assertFalse(result["is_sarcasm"])

    def test_mix_uses_mix_result_when_confident(self) -> None:
        payload = {
            "router_result": {"sample_type": "mix"},
            "emotion_result": {
                "emotion": "happy",
                "intensity": 62,
                "confidence": 0.77,
                "reason": "surface positive signal",
            },
            "mix_result": {
                "is_mixed": True,
                "primary_emotion": "tired",
                "secondary_emotion": "happy",
                "mix_ratio": {"tired": 0.58, "happy": 0.42},
                "revised_intensity": 57,
                "confidence": 0.79,
                "reason": "contrast makes tiredness dominant",
            },
        }

        result = self.agent.judge_dict(payload)

        self.assertEqual(result["final_emotion"], "tired")
        self.assertEqual(result["secondary_emotion"], "happy")
        self.assertEqual(result["final_intensity"], 57)
        self.assertTrue(result["is_mixed"])

    def test_mix_falls_back_when_low_confidence(self) -> None:
        payload = {
            "router_result": {"sample_type": "mix"},
            "emotion_result": {
                "emotion": "anxiety",
                "intensity": 55,
                "confidence": 0.75,
                "reason": "negative tone",
            },
            "mix_result": {
                "is_mixed": True,
                "primary_emotion": "tired",
                "secondary_emotion": "anxiety",
                "mix_ratio": {"tired": 0.51, "anxiety": 0.49},
                "revised_intensity": 53,
                "confidence": 0.41,
                "reason": "not stable enough",
            },
        }

        result = self.agent.judge_dict(payload)

        self.assertEqual(result["final_emotion"], "anxiety")
        self.assertIsNone(result["secondary_emotion"])
        self.assertEqual(result["final_intensity"], 55)
        self.assertEqual(result["final_confidence"], 0.6)
        self.assertFalse(result["is_mixed"])

    def test_direct_high_confidence_skips_llm_review(self) -> None:
        client = FakeJudgeClient()
        agent = JudgeAgent(client=client)
        payload = {
            "router_result": {"sample_type": "direct"},
            "emotion_result": {
                "emotion": "anxiety",
                "intensity": 68,
                "confidence": 0.84,
                "reason": "direct anxiety signal",
            },
        }

        result = agent.judge_dict(payload)

        self.assertEqual(client.calls, 0)
        self.assertEqual(result["final_emotion"], "anxiety")

    def test_conflicting_sarcasm_result_uses_llm_review(self) -> None:
        client = FakeJudgeClient()
        agent = JudgeAgent(client=client)
        payload = {
            "text": "Great, another weekend of work.",
            "router_result": {"sample_type": "sarcasm_suspected"},
            "emotion_result": {
                "emotion": "happy",
                "intensity": 61,
                "confidence": 0.72,
                "reason": "surface positive words",
            },
            "sarcasm_result": {
                "is_sarcasm": True,
                "surface_emotion": "happy",
                "true_emotion": "annoyed",
                "revised_intensity": 74,
                "confidence": 0.86,
                "reason": "positive wording conflicts with negative context",
            },
        }

        result = agent.judge_dict(payload)

        self.assertEqual(client.calls, 1)
        self.assertEqual(result["final_emotion"], "negative")
        self.assertEqual(result["final_confidence"], 0.82)
        self.assertTrue(result["is_sarcasm"])


if __name__ == "__main__":
    unittest.main()

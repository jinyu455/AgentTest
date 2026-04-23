from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from judge_agent import JudgeAgent


class JudgeAgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = JudgeAgent(sarcasm_confidence_threshold=0.65, mix_confidence_threshold=0.65)

    def test_direct_uses_emotion_result(self) -> None:
        payload = {
            "router_result": {"sample_type": "direct"},
            "emotion_result": {
                "emotion": "焦虑",
                "intensity": 68,
                "confidence": 0.84,
                "reason": "直接表达焦虑。",
            },
        }

        result = self.agent.judge_dict(payload)

        self.assertEqual(result["final_emotion"], "焦虑")
        self.assertIsNone(result["secondary_emotion"])
        self.assertEqual(result["final_intensity"], 68)
        self.assertEqual(result["final_confidence"], 0.84)
        self.assertFalse(result["is_sarcasm"])
        self.assertFalse(result["is_mixed"])

    def test_sarcasm_suspected_uses_sarcasm_when_confident(self) -> None:
        payload = {
            "router_result": {"sample_type": "sarcasm_suspected"},
            "emotion_result": {
                "emotion": "开心",
                "intensity": 61,
                "confidence": 0.72,
                "reason": "表面正向词明显。",
            },
            "sarcasm_result": {
                "is_sarcasm": True,
                "surface_emotion": "开心",
                "true_emotion": "厌烦",
                "revised_intensity": 74,
                "confidence": 0.86,
                "reason": "正向词与负面工作语境冲突，反讽成立。",
            },
        }

        result = self.agent.judge_dict(payload)

        self.assertEqual(result["final_emotion"], "厌烦")
        self.assertEqual(result["final_intensity"], 74)
        self.assertTrue(result["is_sarcasm"])
        self.assertFalse(result["is_mixed"])

    def test_sarcasm_suspected_falls_back_when_low_confidence(self) -> None:
        payload = {
            "router_result": {"sample_type": "sarcasm_suspected"},
            "emotion_result": {
                "emotion": "开心",
                "intensity": 58,
                "confidence": 0.8,
                "reason": "表面正向。",
            },
            "sarcasm_result": {
                "is_sarcasm": True,
                "surface_emotion": "开心",
                "true_emotion": "厌烦",
                "revised_intensity": 70,
                "confidence": 0.52,
                "reason": "证据不足。",
            },
        }

        result = self.agent.judge_dict(payload)

        self.assertEqual(result["final_emotion"], "开心")
        self.assertEqual(result["final_intensity"], 58)
        self.assertEqual(result["final_confidence"], 0.64)
        self.assertFalse(result["is_sarcasm"])

    def test_mix_uses_mix_result_when_confident(self) -> None:
        payload = {
            "router_result": {"sample_type": "mix"},
            "emotion_result": {
                "emotion": "开心",
                "intensity": 62,
                "confidence": 0.77,
                "reason": "主观正向。",
            },
            "mix_result": {
                "is_mixed": True,
                "primary_emotion": "疲惫",
                "secondary_emotion": "开心",
                "mix_ratio": {"疲惫": 0.58, "开心": 0.42},
                "revised_intensity": 57,
                "confidence": 0.79,
                "reason": "转折后疲惫占主导，属于混合情绪。",
            },
        }

        result = self.agent.judge_dict(payload)

        self.assertEqual(result["final_emotion"], "疲惫")
        self.assertEqual(result["secondary_emotion"], "开心")
        self.assertEqual(result["final_intensity"], 57)
        self.assertTrue(result["is_mixed"])

    def test_mix_falls_back_when_low_confidence(self) -> None:
        payload = {
            "router_result": {"sample_type": "mix"},
            "emotion_result": {
                "emotion": "焦虑",
                "intensity": 55,
                "confidence": 0.75,
                "reason": "有明显负向色彩。",
            },
            "mix_result": {
                "is_mixed": True,
                "primary_emotion": "疲惫",
                "secondary_emotion": "焦虑",
                "mix_ratio": {"疲惫": 0.51, "焦虑": 0.49},
                "revised_intensity": 53,
                "confidence": 0.41,
                "reason": "结构不够稳定。",
            },
        }

        result = self.agent.judge_dict(payload)

        self.assertEqual(result["final_emotion"], "焦虑")
        self.assertIsNone(result["secondary_emotion"])
        self.assertEqual(result["final_intensity"], 55)
        self.assertEqual(result["final_confidence"], 0.6)
        self.assertFalse(result["is_mixed"])


if __name__ == "__main__":
    unittest.main()

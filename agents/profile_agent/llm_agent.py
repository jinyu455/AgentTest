from __future__ import annotations

import json
from typing import Any, Protocol

from base.coerce import CoercionMixin
from .feature_extractor import extract_features
from .schemas import ProfileInput, ProfileResult
from .visualizer import build_visualization_data


# LLM 系统提示词：指导大模型基于统计数据生成用户画像
SYSTEM_PROMPT = """你是 EmoAgent 中的 Profile Agent，负责根据用户的历史情绪统计数据和对话记录，分析并生成用户的情绪画像。

你将收到：
1. 从历史数据中提取的统计特征（emotion_features）
2. 用户最近的聊天记录摘要（chat_summary）

你需要基于这些信息，生成结构化的用户画像，包括：

1. personality_traits（性格特征）：基于情绪模式推断 2-4 个核心性格特征，每个用简短中文描述（不超过 15 字）
2. communication_style（沟通风格）：用 1-2 句中文描述用户在对话中的沟通倾向（不超过 80 字）
3. emotional_patterns（情绪模式）：用 2-3 句中文描述用户的情绪变化规律和触发因素（不超过 120 字）
4. mbti（MBTI 人格类型）：基于情绪模式、沟通风格和行为特征推断 MBTI 四字母类型（如 INFP、ESTJ），数据不足时输出"UNKNOWN"
5. summary（综合评价）：用 1-2 句中文给出整体用户画像总结（不超过 80 字）

分析原则：
- 基于数据说话，不要凭空臆断
- 关注情绪分布、强度趋势、反讽频率等统计特征
- 如果数据量太少（少于 10 条记录），在 summary 中说明数据不足，画像仅供参考
- 注意区分高频出现的中性情绪和真正的情绪波动
- 如果反讽率较高，说明用户可能倾向于用反话表达真实情绪

输出要求：
- 只返回 JSON
- 不要输出 markdown
- 字段必须完整
- 不要增加额外字段

输出格式：
{
  "personality_traits": ["特征1", "特征2", "特征3"],
  "communication_style": "描述用户的沟通风格",
  "emotional_patterns": "描述用户的情绪模式",
  "mbti": "INFP",
  "summary": "综合评价"
}
"""


def build_profile_user_prompt(features: dict[str, Any], chat_summary: str) -> str:
    """构造 Profile Agent 的用户提示词，包含统计特征和对话摘要。

    同时供 client.py 和 Agent.build_messages() 复用，避免重复构建。
    """
    return (
        "请根据以下用户历史情绪统计数据和对话摘要，生成用户情绪画像 JSON。\n\n"
        f"emotion_features:\n{json.dumps(features, ensure_ascii=False, indent=2)}\n\n"
        f"chat_summary:\n{chat_summary}"
    )


class ProfileLLMClient(Protocol):
    """LLM 客户端接口：接收统计特征和对话摘要，返回画像 JSON。"""

    def analyze(self, features: dict[str, Any], chat_summary: str) -> dict[str, Any]:
        """发送特征和摘要到 LLM，返回解析后的 JSON 结果。"""


def _build_chat_summary(history: list[dict[str, Any]]) -> str:
    """将对话历史格式化为 LLM 可读的摘要文本。

    最多保留最近 20 条消息，过滤无效条目。
    格式示例:
        用户: 今天好累
        助手: 忙了一天确实辛苦...
    """
    if not history:
        return "（无历史对话）"
    lines: list[str] = []
    for item in history[-20:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else "（无历史对话）"


class ProfileAgent(CoercionMixin):
    """用户画像 Agent：提供统计特征计算和可选的 LLM 画像生成。

    两个入口方法：
    - profile()     → 仅返回统计特征 + 可视化数据（不调 LLM）
    - generate()    → 完整画像：统计 + LLM 生成的性格/沟通/情绪描述
    """

    def __init__(self, client: ProfileLLMClient) -> None:
        self.client = client


    def profile(self, payload: ProfileInput | dict[str, Any]) -> ProfileResult:
        """仅计算统计特征和可视化数据，不调用 LLM。

        对应 POST /profile 端点，纯计算、零延迟、零成本。
        """
        item = payload if isinstance(payload, ProfileInput) else ProfileInput(**payload)
        features = extract_features(item.emotion_records)
        viz = build_visualization_data(features, item.emotion_records)
        return self._build_stats_result(features, viz)

    def profile_dict(self, payload: ProfileInput | dict[str, Any]) -> dict[str, Any]:
        """profile() 的字典版本，用于 FastAPI 端点直接返回。"""
        return self.profile(payload).to_dict()

    def generate(self, payload: ProfileInput | dict[str, Any]) -> ProfileResult:
        """完整画像生成：统计特征 + LLM 生成的性格/沟通风格/情绪模式。

        对应 POST /profile/generate 端点，需要调用 LLM。
        """
        item = payload if isinstance(payload, ProfileInput) else ProfileInput(**payload)
        features = extract_features(item.emotion_records)
        viz = build_visualization_data(features, item.emotion_records)
        chat_summary = _build_chat_summary(item.chat_history)
        # 调用 LLM 生成画像描述
        llm_result = self.client.analyze(features, chat_summary)
        return self._build_full_result(llm_result, features, viz)

    def generate_dict(self, payload: ProfileInput | dict[str, Any]) -> dict[str, Any]:
        """generate() 的字典版本，用于 FastAPI 端点直接返回。"""
        return self.generate(payload).to_dict()


    def _build_stats_result(
        self,
        features: dict[str, Any],
        viz: dict[str, Any],
    ) -> ProfileResult:
        """仅用统计特征和可视化数据构建 ProfileResult，LLM 字段留空。"""
        return ProfileResult(
            total_records=features.get("total_records", 0),
            emotion_distribution=features.get("emotion_distribution", {}),
            avg_intensity=features.get("avg_intensity", 0.0),
            avg_confidence=features.get("avg_confidence", 0.0),
            sarcasm_rate=features.get("sarcasm_rate", 0.0),
            mixed_rate=features.get("mixed_rate", 0.0),
            dominant_emotion=features.get("dominant_emotion", "中性"),
            intensity_trend=features.get("intensity_trend", "平稳"),
            activity_pattern=features.get("activity_pattern", {}),
            radar_chart=viz.get("radar_chart", {}),
            timeline=viz.get("timeline", {}),
            intensity_distribution=viz.get("intensity_distribution", {}),
        )

    def _build_full_result(
        self,
        raw_result: dict[str, Any],
        features: dict[str, Any],
        viz: dict[str, Any],
    ) -> ProfileResult:
        """将 LLM 返回的结果与统计特征合并，构建完整的 ProfileResult。

        对 LLM 输出做类型强制转换和空值校验。
        """
        # 从 LLM 结果中提取并校验各字段
        personality_traits = self._coerce_str_list(
            raw_result.get("personality_traits", []), "personality_traits"
        )
        communication_style = str(raw_result.get("communication_style", "")).strip()
        emotional_patterns = str(raw_result.get("emotional_patterns", "")).strip()
        mbti = str(raw_result.get("mbti", "")).strip().upper()
        summary = str(raw_result.get("summary", "")).strip()

        # LLM 返回的字段不能为空
        if not personality_traits:
            raise ValueError("Invalid personality_traits from LLM: empty")
        if not communication_style:
            raise ValueError("Invalid communication_style from LLM: empty")
        if not emotional_patterns:
            raise ValueError("Invalid emotional_patterns from LLM: empty")
        if not summary:
            raise ValueError("Invalid summary from LLM: empty")

        # MBTI 校验：4 个位置各有 2 个合法选项，或 UNKNOWN
        _MBTI_VALID = ({"E", "I"}, {"S", "N"}, {"T", "F"}, {"J", "P"})
        if mbti and mbti != "UNKNOWN":
            if len(mbti) != 4 or not all(
                c in pair for c, pair in zip(mbti, _MBTI_VALID)
            ):
                raise ValueError(f"Invalid mbti from LLM: {mbti!r}")

        return ProfileResult(
            # 统计特征（来自纯计算）
            total_records=features.get("total_records", 0),
            emotion_distribution=features.get("emotion_distribution", {}),
            avg_intensity=features.get("avg_intensity", 0.0),
            avg_confidence=features.get("avg_confidence", 0.0),
            sarcasm_rate=features.get("sarcasm_rate", 0.0),
            mixed_rate=features.get("mixed_rate", 0.0),
            dominant_emotion=features.get("dominant_emotion", "中性"),
            intensity_trend=features.get("intensity_trend", "平稳"),
            activity_pattern=features.get("activity_pattern", {}),
            # LLM 生成的画像描述
            personality_traits=personality_traits,
            communication_style=communication_style,
            emotional_patterns=emotional_patterns,
            mbti=mbti,
            summary=summary,
            # 可视化数据
            radar_chart=viz.get("radar_chart", {}),
            timeline=viz.get("timeline", {}),
            intensity_distribution=viz.get("intensity_distribution", {}),
        )


    # def build_messages(self, payload: ProfileInput | dict[str, Any]) -> list[dict[str, str]]:
    #     """构建发给 LLM 的 system + user 消息列表，用于调试和审查。"""
    #     item = payload if isinstance(payload, ProfileInput) else ProfileInput(**payload)
    #     features = extract_features(item.emotion_records)
    #     chat_summary = _build_chat_summary(item.chat_history)
    #     return [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": build_profile_user_prompt(features, chat_summary)},
    #     ]

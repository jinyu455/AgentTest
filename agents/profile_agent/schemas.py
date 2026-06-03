from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EmotionRecordEntry:
    """对应 Java 后端 EmotionRecord 实体的字段结构。

    每条记录代表一次情绪分析的结果，由 judge_agent 产出并持久化到数据库。
    """
    id: str                           # 记录唯一 ID
    conversation_id: str              # 所属会话 ID
    message_id: str                   # 关联的消息 ID
    final_emotion: str                # 最终情绪标签（如"焦虑""疲惫"）
    secondary_emotion: str | None     # 次要情绪（仅混合情绪时有值）
    final_intensity: int              # 情绪强度 0-100
    final_confidence: float           # 模型置信度 0-1
    is_sarcasm: bool                  # 是否检测到反讽
    is_mixed: bool                    # 是否为混合情绪
    raw_analysis_json: str            # 原始分析结果的 JSON 字符串
    created_at: str                   # ISO 格式时间戳


@dataclass(slots=True)
class ProfileInput:
    """用户画像生成的输入参数。

    接收该用户的所有历史情绪记录和可选的对话历史，
    由前端或 Java 后端组装后传入。
    """
    user_id: str                                              # 用户 ID
    emotion_records: list[dict[str, Any]] = field(default_factory=list)  # 历史情绪记录列表
    chat_history: list[dict[str, Any]] = field(default_factory=list)     # 对话历史（role+content）
    metadata: dict[str, Any] = field(default_factory=dict)               # 额外元数据


@dataclass(slots=True)
class ProfileResult:
    """用户画像结果，包含三层数据：

    1. 统计特征 — 从 emotion_records 纯计算得出，不依赖 LLM
    2. LLM 画像 — 由 LLM 生成的性格/沟通/情绪描述（仅 /profile/generate 返回）
    3. 可视化数据 — 前端图表所需的结构化数据
    """

    # ---- 统计特征（纯计算，/profile 即返回） ----
    total_records: int = 0                          # 总记录数
    emotion_distribution: dict[str, float] = field(default_factory=dict)  # 情绪分布 {"焦虑": 0.4, ...}
    avg_intensity: float = 0.0                      # 平均情绪强度
    avg_confidence: float = 0.0                     # 平均置信度
    sarcasm_rate: float = 0.0                       # 反讽出现比例
    mixed_rate: float = 0.0                         # 混合情绪出现比例
    dominant_emotion: str = "中性"                   # 出现最多的情绪
    intensity_trend: str = "平稳"                    # 强度趋势："上升"|"下降"|"平稳"
    activity_pattern: dict[str, int] = field(default_factory=dict)  # 活跃时段 {"上午": 3, "晚上": 7}

    # ---- LLM 画像（仅 /profile/generate 调用 LLM 后填充） ----
    personality_traits: list[str] = field(default_factory=list)  # 性格特征列表 2-4 个
    communication_style: str = ""                   # 沟通风格描述
    emotional_patterns: str = ""                    # 情绪模式描述
    mbti: str = ""                                  # MBTI 人格类型（如 "INFP"、"ESTJ"）
    summary: str = ""                               # 综合评价

    # ---- 可视化数据（前端图表用） ----
    radar_chart: dict[str, Any] = field(default_factory=dict)              # 情绪雷达图数据
    timeline: dict[str, Any] = field(default_factory=dict)                 # 情绪时间线数据
    intensity_distribution: dict[str, Any] = field(default_factory=dict)   # 强度直方图数据

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，用于 JSON 序列化返回给前端。"""
        return {
            "total_records": self.total_records,
            "emotion_distribution": self.emotion_distribution,
            "avg_intensity": self.avg_intensity,
            "avg_confidence": self.avg_confidence,
            "sarcasm_rate": self.sarcasm_rate,
            "mixed_rate": self.mixed_rate,
            "dominant_emotion": self.dominant_emotion,
            "intensity_trend": self.intensity_trend,
            "activity_pattern": self.activity_pattern,
            "personality_traits": self.personality_traits,
            "communication_style": self.communication_style,
            "emotional_patterns": self.emotional_patterns,
            "mbti": self.mbti,
            "summary": self.summary,
            "radar_chart": self.radar_chart,
            "timeline": self.timeline,
            "intensity_distribution": self.intensity_distribution,
        }

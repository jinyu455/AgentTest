"""共享 schema 模块。

定义情绪分析系统中各模块共用的数据结构和常量，
包括情绪标签集合和统一的输入数据模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 系统支持的全部情绪标签，用于情绪分类任务中的标签验证和映射
EMOTION_LABELS = {"开心", "悲伤", "愤怒", "焦虑", "厌烦", "中性", "疲惫", "失落", "无奈"}


@dataclass(slots=True)
class BaseTextInput:
    """路由、情绪、讽刺和混合 Agent 的通用输入 schema。

    所有下游 Agent 统一接收该结构作为输入，确保数据格式一致性。
    metadata 字段为可选的扩展字典，用于携带额外上下文信息。

    Attributes:
        id: 当前会话或消息的唯一标识符
        user_id: 用户唯一标识符
        text: 用户输入的文本内容
        source: 消息来源渠道标识（如 "web"、"app" 等）
        created_at: 消息创建时间戳字符串
        metadata: 可选的扩展元数据字典，默认为空字典
    """
    id: str
    user_id: str
    text: str
    source: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

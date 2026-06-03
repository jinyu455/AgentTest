"""Chat Agent 的数据模型定义。

包含聊天响应生成所需的输入数据结构 ChatInput、
对话消息结构 ChatMessage 和输出结果数据结构 ChatResult。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChatMessage:
    """单条对话消息。

    用于表示对话历史中的每一条消息，包含角色和内容。
    """

    role: str       # 消息角色："user"（用户）或 "assistant"（助手）
    content: str    # 消息内容


@dataclass(slots=True)
class ChatInput:
    """聊天响应生成的输入数据。

    包含用户当前消息、对话上下文和情绪分析结果，
    供 Chat Agent 生成合适的回复。
    """

    text: str                                                        # 当前用户输入的文本
    user_id: str | None = None                                       # 用户 ID（可选）
    conversation_id: str | None = None                               # 对话 ID（可选，用于关联会话）
    judge_result: dict[str, Any] | None = None                       # Judge Agent 的最终裁决结果
    history: list[dict[str, Any]] = field(default_factory=list)      # 历史对话记录列表
    metadata: dict[str, Any] = field(default_factory=dict)           # 附加元数据


@dataclass(slots=True)
class ChatResult:
    """聊天响应生成的输出结果。

    包含生成的回复文本、语气、风险提示等信息。
    """

    reply: str                                           # 给用户的回复文本
    tone: str                                            # 回复语气（如 supportive、calm 等）
    risk_hint: str                                       # 风险提示（none 或 possible_crisis）
    suggested_actions: list[str] = field(default_factory=list)  # 建议的行动项列表
    reason: str = ""                                     # 生成依据的简短说明

    def to_dict(self) -> dict[str, Any]:
        """将结果转换为字典格式，便于序列化传输。"""
        return {
            "reply": self.reply,
            "tone": self.tone,
            "risk_hint": self.risk_hint,
            "suggested_actions": self.suggested_actions,
            "reason": self.reason,
        }

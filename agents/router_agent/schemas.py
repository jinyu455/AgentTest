"""路由代理的输入输出数据模型。

定义了 RouterAgent 接收的输入结构（RouterInput）和
输出的路由判断结果结构（RouterResult）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from base.schemas import BaseTextInput

# RouterInput 与 BaseTextInput 字段完全一致，直接复用
RouterInput = BaseTextInput


@dataclass(slots=True)
class RouterResult:
    """路由代理的输出数据结构。

    包含对输入消息的分类结果以及后续处理的路由建议，
    供下游代理（情绪代理、反讽代理、混合代理）决定是否参与处理。

    Attributes:
        sample_type: 消息的表达类型，取值为 "direct"、
            "sarcasm_suspected" 或 "mix" 之一。
        need_sarcasm_check: 是否需要反讽代理进一步判断。
        need_mix_check: 是否需要混合情绪代理进一步判断。
        routing_reason: 路由判断的简短文字理由。
        evidence: 支持路由判断的线索列表。
    """
    sample_type: str
    need_sarcasm_check: bool
    need_mix_check: bool
    routing_reason: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """将结果转换为普通字典，便于序列化或日志记录。"""
        return {
            "sample_type": self.sample_type,
            "need_sarcasm_check": self.need_sarcasm_check,
            "need_mix_check": self.need_mix_check,
            "routing_reason": self.routing_reason,
            "evidence": self.evidence,
        }

"""Chat Agent 的核心逻辑实现。

负责生成情绪聊天助手的回复，包括系统提示词定义、LLM 协议接口定义、
用户提示词构建、对话历史格式化，以及对大模型返回结果的校验和类型强转。
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from base.coerce import CoercionMixin
from .schemas import ChatInput, ChatResult


# 合法的回复语气标签集合
TONE_LABELS = {"supportive", "calm", "encouraging", "reflective", "crisis_support"}
# 合法的风险提示标签集合
RISK_HINTS = {"none", "possible_crisis"}

# Chat Agent 的系统提示词，定义了回复原则、语气要求和输出格式
SYSTEM_PROMPT = """你是 EmoAgent 中的 Chat Agent，负责生成情绪聊天助手的回复。
你的目标不是给用户贴标签，也不是做机械的心理咨询式追问，而是基于情绪分析结果和最近对话历史，用温和、尊重、具体、有用的方式回应用户。

回复原则：
- 优先理解最近对话历史，尤其是最近 3 轮。如果当前用户说"能给我一些建议吗""那怎么办""这个怎么处理""我该咋办"等，必须结合上文直接回答。
- 当用户请求建议、办法、下一步、安慰或陪伴时，先给出可执行帮助，不要先追问。信息不足时，也要先给通用可行建议，再在结尾只问 1 个必要问题。
- 回复要抓住用户说过的具体事实，不要只说"听起来你很难受""能具体说说吗"这类空泛模板。
- 可以先简短回应感受，但不要停在共情上；普通场景下至少给出 2 个具体做法或下一步。
- 建议要轻量、具体、可执行，例如"先列清单""先确认对方要求""把任务拆成 3 块""休息 10 分钟再处理最小一项"等。
- 不要说教，不要夸大判断，不要假装自己能替代专业帮助。
- 如果 judge_result 中存在较高 safety_score，或文本出现自伤、自杀、伤害他人、极端危机倾向，risk_hint 必须为 possible_crisis，tone 使用 crisis_support。
- 危机场景中，不提供危险方法，不鼓励危险行为，应建议联系可信任的人、当地紧急服务或专业支持。
- 普通场景下，回复控制在 2 到 5 句话，像一个可靠、有行动感的聊天助手，而不是分析报告。
- 不要在 reply 里展示情绪标签、置信度、强度分数或 JSON 字段名。

只返回 JSON，不要输出 markdown，不要增加额外字段。
输出格式：
{
  "reply": "给用户的中文回复",
  "tone": "supportive | calm | encouraging | reflective | crisis_support",
  "risk_hint": "none | possible_crisis",
  "suggested_actions": ["可选的简短行动建议"],
  "reason": "简短说明生成依据"
}
"""


class ChatLLMClient(Protocol):
    """Chat Agent 的大模型客户端协议接口。

    定义了 generate 方法的签名，任何实现该协议的客户端都可以
    用于 ChatAgent 的回复生成。
    """

    def generate(self, payload: ChatInput) -> dict[str, Any]:
        """发送输入到大模型并返回解析后的 JSON 结果。"""


class ChatAgent(CoercionMixin):
    """基于大模型的情绪聊天回复 Agent。

    负责根据用户消息、对话历史和情绪分析结果，
    生成温和、尊重、具体的聊天回复。
    """

    def __init__(self, client: ChatLLMClient) -> None:
        """初始化 ChatAgent。

        Args:
            client: 实现 ChatLLMClient 协议的大模型客户端实例。
        """
        self.client = client

    def chat(self, payload: ChatInput | dict[str, Any]) -> ChatResult:
        """生成聊天回复，返回结构化的 ChatResult 对象。

        如果传入的是字典，会先转换为 ChatInput 数据类。

        Args:
            payload: 聊天输入数据，可以是 ChatInput 或字典。

        Returns:
            结构化的聊天回复结果。
        """
        # 支持传入字典或 ChatInput 对象
        item = payload if isinstance(payload, ChatInput) else ChatInput(**payload)
        # 调用大模型获取原始结果
        raw_result = self.client.generate(item)
        # 将原始结果校验并构建为 ChatResult
        return self._build_result(raw_result)

    def chat_dict(self, payload: ChatInput | dict[str, Any]) -> dict[str, Any]:
        """生成聊天回复，返回字典格式的结果。

        是 chat 的便捷版本，直接返回可序列化的字典。
        """
        return self.chat(payload).to_dict()

    def _build_result(self, raw_result: dict[str, Any]) -> ChatResult:
        """将大模型返回的原始字典校验并构建为 ChatResult 对象。

        对每个字段进行类型强转和合法性校验，不合法时抛出 ValueError。

        Args:
            raw_result: 大模型返回的原始字典。

        Returns:
            校验通过的 ChatResult 对象。

        Raises:
            ValueError: 当字段值不符合约束条件时抛出。
        """
        # 回复内容不能为空
        reply = str(raw_result.get("reply", "")).strip()
        if not reply:
            raise ValueError("Invalid reply from LLM: empty")

        # 校验语气标签是否合法
        tone = str(raw_result.get("tone", "")).strip()
        if tone not in TONE_LABELS:
            raise ValueError(f"Invalid tone from LLM: {tone!r}")

        # 校验风险提示标签是否合法
        risk_hint = str(raw_result.get("risk_hint", "")).strip()
        if risk_hint not in RISK_HINTS:
            raise ValueError(f"Invalid risk_hint from LLM: {risk_hint!r}")

        return ChatResult(
            reply=reply,
            tone=tone,
            risk_hint=risk_hint,
            suggested_actions=self._coerce_str_list(raw_result.get("suggested_actions", []), "suggested_actions"),
            reason=str(raw_result.get("reason", "")).strip(),
        )

    # def build_messages(self, payload: ChatInput | dict[str, Any]) -> list[dict[str, str]]:
    #     """构建发送给大模型的消息列表。

    #     包含系统提示词和用户提示词，用于需要手动管理对话格式的场景。

    #     Args:
    #         payload: 聊天输入数据。

    #     Returns:
    #         符合 OpenAI 对话格式的消息列表。
    #     """
    #     item = payload if isinstance(payload, ChatInput) else ChatInput(**payload)
    #     return [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": build_chat_user_prompt(item)},
    #     ]


def build_chat_user_prompt(payload: ChatInput) -> str:
    """构建聊天回复生成的用户提示词。

    将对话历史、当前用户消息、情绪分析结果和元数据
    组织为结构化的提示词文本。

    Args:
        payload: 聊天输入数据。

    Returns:
        格式化的用户提示词字符串。
    """
    # 格式化对话历史为可读文本
    history = format_chat_history(payload.history)
    # 将情绪分析结果和元数据序列化为 JSON
    judge_result = json.dumps(payload.judge_result or {}, ensure_ascii=False, indent=2)
    metadata = json.dumps(payload.metadata or {}, ensure_ascii=False, indent=2)
    return (
        "请根据最近对话历史、当前用户消息和情绪分析结果生成回复。\n"
        "如果当前用户消息里出现\u201c这/这个/刚才/它/建议/怎么办\u201d等依赖上下文的表达，"
        "必须优先结合最近对话历史理解指代，不要把它当成全新的泛泛问题。\n\n"
        f"conversation_id: {payload.conversation_id or ''}\n"
        f"user_id: {payload.user_id or ''}\n\n"
        "最近对话历史（按时间从旧到新）：\n"
        f"{history}\n\n"
        "当前用户消息：\n"
        f"{payload.text}\n\n"
        "当前消息的情绪分析 judge_result：\n"
        f"{judge_result}\n\n"
        "metadata：\n"
        f"{metadata}"
    )


def format_chat_history(history: list[dict[str, Any]]) -> str:
    """将对话历史列表格式化为可读文本。

    只保留最近 20 条有效消息，每条消息标注角色（用户/助手）。
    过滤掉无效的角色和空内容。

    Args:
        history: 对话历史记录列表，每条记录包含 role 和 content 字段。

    Returns:
        格式化后的对话历史文本，无历史时返回"（无历史）"。
    """
    if not history:
        return "（无历史）"

    lines: list[str] = []
    # 只取最近 20 条消息，避免提示词过长
    for item in history[-20:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        # 只接受 user 和 assistant 角色
        if role not in {"user", "assistant"} or not content:
            continue
        # 将英文角色名转为中文标签
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}: {content}")

    return "\n".join(lines) if lines else "（无历史）"

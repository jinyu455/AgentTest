"""路由代理的核心逻辑模块。

定义了路由代理的系统提示词（SYSTEM_PROMPT）和 RouterAgent 类。
路由代理负责将输入消息分为三类：直接表达（direct）、
疑似反讽（sarcasm_suspected）和混合情绪（mix），
并决定后续是否需要调用反讽代理或混合情绪代理。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Protocol

from base.coerce import CoercionMixin
from base.schemas import BaseTextInput
from .schemas import RouterInput, RouterResult


# 路由代理的系统提示词，定义了大模型的角色、任务、分类规则和输出格式
SYSTEM_PROMPT = """你是情绪识别系统中的 Router Agent。

你的任务只有两个：
1. 判断输入句子的表达类型，只能输出以下三类之一：
- direct
- sarcasm_suspected
- mix
2. 决定是否需要调用后续模块：
- need_sarcasm_check
- need_mix_check

分类规则：
1. direct
- 明显直接情绪表达
- 没有明显转折
- 没有明显反讽结构
- 没有明显复合情绪

2. sarcasm_suspected
- 句面正向，语境负向
- 或夸张正向词和明显糟糕事件并存
- 常见触发：又、还真是、真棒、太好了

3. mix
- 有转折词或复合表达
- 有两个情绪方向
- 情绪模糊，不适合单标签
- 低能量、压抑、说不上来、提不起劲等表达

输出要求：
- 只能返回 JSON
- 不要输出 markdown
- 字段必须完整
- evidence 是字符串数组，列出支持判断的线索

输出格式：
{
  "sample_type": "direct | sarcasm_suspected | mix",
  "need_sarcasm_check": true,
  "need_mix_check": false,
  "routing_reason": "简洁说明原因",
  "evidence": ["线索1", "线索2"]
}
"""


def build_router_user_prompt(payload: RouterInput) -> str:
    """构造路由代理的用户提示词，将输入数据序列化为可读 JSON。

    同时供 client.py 和 Agent.build_messages() 复用，避免重复构建。
    """
    return (
        "请判断下面这条消息的路由类型，并给出 JSON 结果。\n\n"
        f"{json.dumps(asdict(payload), ensure_ascii=False, indent=2)}"
    )


class RouterLLMClient(Protocol):
    """路由代理 LLM 客户端的协议接口。

    定义了 classify 方法的签名，任何路由代理使用的 LLM 客户端
    都需要实现此接口，确保可替换性和类型安全。
    """
    def classify(self, payload: RouterInput) -> dict:
        """发送路由输入到大模型并返回解析后的 JSON 结果。"""


class RouterAgent(CoercionMixin):
    """基于大语言模型的路由代理。

    负责接收输入消息，通过 LLM 进行路由分类，并将原始结果
    校验和清洗为结构化的 RouterResult 对象。
    继承 CoercionMixin 以获得类型强转工具方法。
    """

    def __init__(self, client: RouterLLMClient) -> None:
        """初始化路由代理。

        Args:
            client: 实现了 RouterLLMClient 协议的 LLM 客户端实例。
        """
        self.client = client

    def route(self, payload: RouterInput | dict) -> RouterResult:
        """对输入消息执行路由分类。

        Args:
            payload: 路由输入数据，可以是 RouterInput 实例或字典。

        Returns:
            校验后的路由判断结果。
        """
        # 支持字典输入，自动转换为 RouterInput 数据类
        item = payload if isinstance(payload, RouterInput) else RouterInput(**payload)
        # 调用大模型客户端获取原始结果
        raw_result = self.client.classify(item)
        # 对原始结果进行校验和类型转换
        return self._build_result(raw_result)

    def route_dict(self, payload: RouterInput | dict) -> dict:
        """对输入消息执行路由分类，返回字典格式结果。

        适用于需要将结果进行 JSON 序列化或传递给非 Python 上下文的场景。
        """
        return self.route(payload).to_dict()

    def _build_result(self, raw_result: dict) -> RouterResult:
        """将大模型的原始返回结果校验并构建为 RouterResult。

        对 sample_type 进行合法性校验，确保其为三个有效值之一；
        对布尔字段和字符串字段进行基本清洗。

        Args:
            raw_result: 大模型返回的原始字典。

        Returns:
            校验通过的 RouterResult 对象。

        Raises:
            ValueError: 当 sample_type 不在允许的枚举值中时抛出。
        """
        sample_type = str(raw_result.get("sample_type", "")).strip()
        # 校验 sample_type 必须是三种合法类型之一
        if sample_type not in {"direct", "sarcasm_suspected", "mix"}:
            raise ValueError(f"Invalid sample_type from LLM: {sample_type!r}")

        return RouterResult(
            sample_type=sample_type,
            need_sarcasm_check=bool(raw_result.get("need_sarcasm_check")),
            need_mix_check=bool(raw_result.get("need_mix_check")),
            routing_reason=str(raw_result.get("routing_reason", "")).strip(),
            evidence=[str(item) for item in raw_result.get("evidence", [])],
        )

    # #以下的函数只是为了让我们看到发给大模型的message
    # def build_messages(self, payload: RouterInput | dict) -> list[dict[str, str]]:
    #     """构建发送给大模型的完整消息列表（调试/预览用）。

    #     将系统提示词和用户提示词组合为标准的聊天消息格式，
    #     便于开发者查看实际发送给大模型的内容。
    #     """
    #     item = payload if isinstance(payload, RouterInput) else RouterInput(**payload)
    #     return [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": build_router_user_prompt(item)},
    #     ]

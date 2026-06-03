"""EmoAgent 服务层的工具函数模块。

包含 Agent 构建、启动初始化、统一执行器、异常处理和 fallback 降级函数。
app.py 只保留 FastAPI 路由定义和端点，非主要逻辑全部在此模块。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from fastapi import HTTPException

# 动态将 agents 目录加入 Python 路径，确保各 Agent 包可被导入
AGENTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTS_ROOT))

from chat_agent import ChatAgent, HTTPChatLLMClient  # noqa: E402
from emotion_agent import EmotionAgent, HTTPEmotionLLMClient  # noqa: E402
from judge_agent import HTTPJudgeLLMClient, JudgeAgent  # noqa: E402
from mix_agent import HTTPMixLLMClient, MixAgent  # noqa: E402
from router_agent import HTTPRouterLLMClient, RouterAgent  # noqa: E402
from sarcasm_agent import HTTPSarcasmLLMClient, SarcasmAgent  # noqa: E402
from profile_agent import ProfileAgent, HTTPProfileLLMClient  # noqa: E402
from profile_agent import extract_features, build_visualization_data  # noqa: E402
from base.llm_config import LLMConfig  # noqa: E402

def _load_api_key() -> str:
    """加载 API 密钥。

    按以下优先级查找 API_KEY：
    1. 项目根目录下的 .env 文件
    2. 系统环境变量

    Returns:
        API 密钥字符串。

    Raises:
        RuntimeError: 当 API_KEY 未找到时抛出。
    """
    # 优先从 .env 文件读取
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "API_KEY":
                api_key = value.strip().strip('"').strip("'")
                if api_key:
                    return api_key

    # 其次从环境变量读取
    api_key = os.getenv("API_KEY", "").strip()
    if api_key:
        return api_key

    raise RuntimeError("API_KEY not found. Please set it in .env or environment variables.")

def _build_router_agent() -> RouterAgent:
    """构建 Router Agent 实例。"""
    config = LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-flash"),
    )
    return RouterAgent(client=HTTPRouterLLMClient(config))


def _build_emotion_agent() -> EmotionAgent:
    """构建 Emotion Agent 实例。"""
    config = LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-flash"),
    )
    return EmotionAgent(client=HTTPEmotionLLMClient(config))


def _build_sarcasm_agent() -> SarcasmAgent:
    """构建 Sarcasm Agent 实例。"""
    config = LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-flash"),
    )
    return SarcasmAgent(client=HTTPSarcasmLLMClient(config))

def _build_mix_agent() -> MixAgent:
    """构建 Mix Agent 实例。"""
    config = LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-flash"),
    )
    return MixAgent(client=HTTPMixLLMClient(config))


def _build_judge_agent() -> JudgeAgent:
    """构建 Judge Agent 实例。"""
    config = LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-flash"),
    )
    return JudgeAgent(client=HTTPJudgeLLMClient(config))


def _build_chat_agent() -> ChatAgent:
    """构建 Chat Agent 实例。Chat Agent 默认使用 qwen-plus 模型。"""
    config = LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-plus"),
    )
    return ChatAgent(client=HTTPChatLLMClient(config))


def _build_profile_agent() -> ProfileAgent:
    """构建 Profile Agent 实例。"""
    config = LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-flash"),
    )
    return ProfileAgent(client=HTTPProfileLLMClient(config))


try:
    router_agent = _build_router_agent()
    emotion_agent = _build_emotion_agent()
    sarcasm_agent = _build_sarcasm_agent()
    mix_agent = _build_mix_agent()
    judge_agent = _build_judge_agent()
    chat_agent = _build_chat_agent()
    profile_agent = _build_profile_agent()
except RuntimeError as exc:
    # 启动失败时的降级处理：所有 Agent 设为 None，JudgeAgent 以纯规则模式运行
    router_agent = None
    emotion_agent = None
    sarcasm_agent = None
    mix_agent = None
    judge_agent = JudgeAgent()  # 无客户端，纯规则模式
    chat_agent = None
    profile_agent = None
    startup_error = str(exc)
else:
    startup_error = ""


def _ensure_ready() -> None:
    """检查服务是否已就绪，未就绪时抛出 503。"""
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Service not ready: {startup_error}")


def _execute(callable_fn: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """统一的 Agent 方法执行器，将异常映射为 HTTP 状态码。

    - ValueError -> 400
    - HTTPError -> 502
    - URLError -> 502
    - TimeoutError -> 504
    - 其他 -> 500
    """
    _ensure_ready()
    try:
        return callable_fn(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPError as exc:
        raise HTTPException(status_code=502, detail=_http_error_detail(exc)) from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"LLM network error: {exc}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"LLM timeout: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


def _execute_agent(agent: Any, method_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """执行 Agent 的指定方法，无 fallback。"""
    _ensure_ready()
    return _execute(getattr(agent, method_name), payload)


def _execute_agent_or_fallback(
    agent: Any,
    method_name: str,
    payload: dict[str, Any],
    fallback_fn: Any,
) -> dict[str, Any]:
    """执行 Agent 的指定方法，失败时调用 fallback 降级处理。"""
    _ensure_ready()
    try:
        return getattr(agent, method_name)(payload)
    except Exception as exc:
        return fallback_fn(payload, _exception_detail(exc))


def _exception_detail(exc: Exception) -> str:
    """提取异常详情，HTTPError 会读取响应体。"""
    if isinstance(exc, HTTPError):
        return _http_error_detail(exc)
    return str(exc)


def _http_error_detail(exc: HTTPError) -> str:
    """提取 HTTP 错误详情，包含响应体前 1000 字符。"""
    body = ""
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""

    if body:
        return f"LLM HTTP error {exc.code}: {body[:1000]}"
    return f"LLM HTTP error {exc.code}: {exc.reason}"


def _fallback_router(payload: dict[str, Any], detail: str) -> dict[str, Any]:
    """Router Agent 降级：返回 direct 路由。"""
    return {
        "sample_type": "direct",
        "need_sarcasm_check": False,
        "need_mix_check": False,
        "routing_reason": f"Router LLM 调用失败，已降级为 direct：{detail}",
        "evidence": ["fallback"],
        "fallback": True,
        "fallback_detail": detail,
    }


def _fallback_emotion(payload: dict[str, Any], detail: str) -> dict[str, Any]:
    """Emotion Agent 降级：返回低置信度中性结果。"""
    text = str(payload.get("text", "")).strip()
    return {
        "tokens": [text] if text else [],
        "emotion_words": [],
        "degree_words": [],
        "negation_words": [],
        "contrast_words": [],
        "emotion": "中性",
        "intensity": 30,
        "confidence": 0.2,
        "reason": f"Emotion LLM 调用失败，已使用低置信度中性兜底：{detail}",
        "fallback": True,
        "fallback_detail": detail,
    }


def _fallback_sarcasm(payload: dict[str, Any], detail: str) -> dict[str, Any]:
    """Sarcasm Agent 降级：判定为无反讽。"""
    return {
        "is_sarcasm": False,
        "surface_emotion": "中性",
        "true_emotion": "中性",
        "revised_intensity": 30,
        "confidence": 0.2,
        "reason": f"Sarcasm LLM 调用失败，已降级为无反讽：{detail}",
        "fallback": True,
        "fallback_detail": detail,
    }


def _fallback_mix(payload: dict[str, Any], detail: str) -> dict[str, Any]:
    """Mix Agent 降级：判定为非混合情绪。"""
    return {
        "is_mixed": False,
        "primary_emotion": "中性",
        "secondary_emotion": "",
        "mix_ratio": {},
        "adjusted_intensity": 30,
        "confidence": 0.2,
        "reason": f"Mix LLM 调用失败，已降级为非混合情绪：{detail}",
        "fallback": True,
        "fallback_detail": detail,
    }


def _fallback_judge(payload: dict[str, Any], detail: str) -> dict[str, Any]:
    """Judge Agent 降级：尝试纯规则裁决，失败则用 emotion_result 兜底。"""
    try:
        result = JudgeAgent().judge_dict(payload)
    except Exception:
        emotion_result = payload.get("emotion_result") or {}
        result = {
            "final_emotion": str(emotion_result.get("emotion", "中性")),
            "secondary_emotion": None,
            "final_intensity": int(emotion_result.get("intensity", 30)),
            "final_confidence": float(emotion_result.get("confidence", 0.2)),
            "is_sarcasm": False,
            "is_mixed": False,
            "reason": "Judge LLM 调用失败，已使用 emotion_result 兜底。",
        }

    result["fallback"] = True
    result["fallback_detail"] = detail
    return result


def _fallback_profile(payload: dict[str, Any], detail: str) -> dict[str, Any]:
    """Profile Agent 降级：提取统计数据，LLM 字段留空。"""
    try:
        records = payload.get("emotion_records", [])
        features = extract_features(records)
        viz = build_visualization_data(features, records)
    except Exception:
        features = {}
        viz = {"radar_chart": {}, "timeline": {}, "intensity_distribution": {}}

    return {
        "total_records": features.get("total_records", 0),
        "emotion_distribution": features.get("emotion_distribution", {}),
        "avg_intensity": features.get("avg_intensity", 0.0),
        "avg_confidence": features.get("avg_confidence", 0.0),
        "sarcasm_rate": features.get("sarcasm_rate", 0.0),
        "mixed_rate": features.get("mixed_rate", 0.0),
        "dominant_emotion": features.get("dominant_emotion", "中性"),
        "intensity_trend": features.get("intensity_trend", "平稳"),
        "activity_pattern": features.get("activity_pattern", {}),
        "personality_traits": [],
        "communication_style": f"LLM 调用失败，仅提供统计数据：{detail}",
        "emotional_patterns": "",
        "summary": f"数据统计可用，但 LLM 画像生成失败：{detail}",
        "radar_chart": viz.get("radar_chart", {}),
        "timeline": viz.get("timeline", {}),
        "intensity_distribution": viz.get("intensity_distribution", {}),
        "fallback": True,
        "fallback_detail": detail,
    }


# Agent 实例
__all__ = [
    "router_agent", "emotion_agent", "sarcasm_agent", "mix_agent",
    "judge_agent", "chat_agent", "profile_agent",
    "startup_error",
    "execute", "execute_agent", "execute_agent_or_fallback",
    "fallback_router", "fallback_emotion", "fallback_sarcasm",
    "fallback_mix", "fallback_judge", "fallback_profile",
]

# 执行器
execute = _execute
execute_agent = _execute_agent
execute_agent_or_fallback = _execute_agent_or_fallback

# Fallback
fallback_router = _fallback_router
fallback_emotion = _fallback_emotion
fallback_sarcasm = _fallback_sarcasm
fallback_mix = _fallback_mix
fallback_judge = _fallback_judge
fallback_profile = _fallback_profile

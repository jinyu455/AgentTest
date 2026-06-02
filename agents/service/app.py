from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

AGENTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTS_ROOT))

from chat_agent import ChatAgent, HTTPChatLLMClient  # noqa: E402
from chat_agent import LLMConfig as ChatLLMConfig  # noqa: E402
from emotion_agent import EmotionAgent, HTTPEmotionLLMClient  # noqa: E402
from emotion_agent import LLMConfig as EmotionLLMConfig  # noqa: E402
from judge_agent import HTTPJudgeLLMClient, JudgeAgent  # noqa: E402
from judge_agent import LLMConfig as JudgeLLMConfig  # noqa: E402
from mix_agent import HTTPMixLLMClient, MixAgent  # noqa: E402
from mix_agent import LLMConfig as MixLLMConfig  # noqa: E402
from router_agent import HTTPRouterLLMClient, RouterAgent  # noqa: E402
from router_agent import LLMConfig as RouterLLMConfig  # noqa: E402
from sarcasm_agent import HTTPSarcasmLLMClient, SarcasmAgent  # noqa: E402
from sarcasm_agent import LLMConfig as SarcasmLLMConfig  # noqa: E402


class TextInput(BaseModel):
    id: str
    user_id: str
    text: str
    source: str
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeInputPayload(BaseModel):
    router_result: dict[str, Any]
    emotion_result: dict[str, Any]
    sarcasm_result: dict[str, Any] | None = None
    mix_result: dict[str, Any] | None = None
    text: str | None = None


class ChatInputPayload(BaseModel):
    text: str
    user_id: str | None = None
    conversation_id: str | None = None
    judge_result: dict[str, Any] | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _load_api_key() -> str:
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

    api_key = os.getenv("API_KEY", "").strip()
    if api_key:
        return api_key

    raise RuntimeError("API_KEY not found. Please set it in .env or environment variables.")


def _build_router_agent() -> RouterAgent:
    config = RouterLLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-flash"),
    )
    return RouterAgent(client=HTTPRouterLLMClient(config))


def _build_emotion_agent() -> EmotionAgent:
    config = EmotionLLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-flash"),
    )
    return EmotionAgent(client=HTTPEmotionLLMClient(config))


def _build_sarcasm_agent() -> SarcasmAgent:
    config = SarcasmLLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-flash"),
    )
    return SarcasmAgent(client=HTTPSarcasmLLMClient(config))


def _build_mix_agent() -> MixAgent:
    config = MixLLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-flash"),
    )
    return MixAgent(client=HTTPMixLLMClient(config))


def _build_judge_agent() -> JudgeAgent:
    config = JudgeLLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-flash"),
    )
    return JudgeAgent(client=HTTPJudgeLLMClient(config))


def _build_chat_agent() -> ChatAgent:
    config = ChatLLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "qwen-plus"),
    )
    return ChatAgent(client=HTTPChatLLMClient(config))


app = FastAPI(title="EmoAgent Service", version="1.0.0")

try:
    router_agent = _build_router_agent()
    emotion_agent = _build_emotion_agent()
    sarcasm_agent = _build_sarcasm_agent()
    mix_agent = _build_mix_agent()
    judge_agent = _build_judge_agent()
    chat_agent = _build_chat_agent()
except RuntimeError as exc:
    router_agent = None
    emotion_agent = None
    sarcasm_agent = None
    mix_agent = None
    judge_agent = JudgeAgent()
    chat_agent = None
    startup_error = str(exc)
else:
    startup_error = ""


def _ensure_ready() -> None:
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Service not ready: {startup_error}")


def _execute(callable_fn: Any, payload: dict[str, Any]) -> dict[str, Any]:
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
    _ensure_ready()
    return _execute(getattr(agent, method_name), payload)


def _execute_agent_or_fallback(
    agent: Any,
    method_name: str,
    payload: dict[str, Any],
    fallback_fn: Any,
) -> dict[str, Any]:
    _ensure_ready()
    try:
        return getattr(agent, method_name)(payload)
    except Exception as exc:
        return fallback_fn(payload, _exception_detail(exc))


def _exception_detail(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        return _http_error_detail(exc)
    return str(exc)


def _http_error_detail(exc: HTTPError) -> str:
    body = ""
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""

    if body:
        return f"LLM HTTP error {exc.code}: {body[:1000]}"
    return f"LLM HTTP error {exc.code}: {exc.reason}"


def _fallback_router(payload: dict[str, Any], detail: str) -> dict[str, Any]:
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
    return {
        "is_mixed": False,
        "primary_emotion": "中性",
        "secondary_emotion": "",
        "mix_ratio": {},
        "revised_intensity": 30,
        "confidence": 0.2,
        "reason": f"Mix LLM 调用失败，已降级为非混合情绪：{detail}",
        "fallback": True,
        "fallback_detail": detail,
    }


def _fallback_judge(payload: dict[str, Any], detail: str) -> dict[str, Any]:
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


@app.get("/health")
def health() -> dict[str, Any]:
    if startup_error:
        return {"status": "degraded", "ready": False, "reason": startup_error}
    return {"status": "ok", "ready": True}


@app.post("/router")
def route(payload: TextInput) -> dict[str, Any]:
    return _execute_agent_or_fallback(router_agent, "route_dict", payload.model_dump(), _fallback_router)


@app.post("/emotion")
def emotion(payload: TextInput) -> dict[str, Any]:
    return _execute_agent_or_fallback(emotion_agent, "emotionRe_dict", payload.model_dump(), _fallback_emotion)


@app.post("/sarcasm")
def sarcasm(payload: TextInput) -> dict[str, Any]:
    return _execute_agent_or_fallback(sarcasm_agent, "detect_dict", payload.model_dump(), _fallback_sarcasm)


@app.post("/mix")
def mix(payload: TextInput) -> dict[str, Any]:
    return _execute_agent_or_fallback(mix_agent, "mixRe_dict", payload.model_dump(), _fallback_mix)


@app.post("/judge")
def judge(payload: JudgeInputPayload) -> dict[str, Any]:
    return _execute_agent_or_fallback(judge_agent, "judge_dict", payload.model_dump(), _fallback_judge)


@app.post("/chat")
def chat(payload: ChatInputPayload) -> dict[str, Any]:
    return _execute_agent(chat_agent, "chat_dict", payload.model_dump())

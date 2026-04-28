from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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


def _load_api_key() -> str:
    env_path = PROJECT_ROOT / ".env"
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
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
    )
    return RouterAgent(client=HTTPRouterLLMClient(config))


def _build_emotion_agent() -> EmotionAgent:
    config = EmotionLLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
    )
    return EmotionAgent(client=HTTPEmotionLLMClient(config))


def _build_sarcasm_agent() -> SarcasmAgent:
    config = SarcasmLLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
    )
    return SarcasmAgent(client=HTTPSarcasmLLMClient(config))


def _build_mix_agent() -> MixAgent:
    config = MixLLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
    )
    return MixAgent(client=HTTPMixLLMClient(config))


def _build_judge_agent() -> JudgeAgent:
    config = JudgeLLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1/chat/completions"),
        api_key=_load_api_key(),
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
    )
    return JudgeAgent(client=HTTPJudgeLLMClient(config))


app = FastAPI(title="EmoAgent Service", version="1.0.0")

try:
    router_agent = _build_router_agent()
    emotion_agent = _build_emotion_agent()
    sarcasm_agent = _build_sarcasm_agent()
    mix_agent = _build_mix_agent()
    judge_agent = _build_judge_agent()
except RuntimeError as exc:
    router_agent = None
    emotion_agent = None
    sarcasm_agent = None
    mix_agent = None
    judge_agent = JudgeAgent()
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
        raise HTTPException(status_code=502, detail=f"LLM HTTP error: {exc}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"LLM network error: {exc}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"LLM timeout: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    if startup_error:
        return {"status": "degraded", "ready": False, "reason": startup_error}
    return {"status": "ok", "ready": True}


@app.post("/router")
def route(payload: TextInput) -> dict[str, Any]:
    return _execute(router_agent.route_dict, payload.model_dump())


@app.post("/emotion")
def emotion(payload: TextInput) -> dict[str, Any]:
    return _execute(emotion_agent.emotionRe_dict, payload.model_dump())


@app.post("/sarcasm")
def sarcasm(payload: TextInput) -> dict[str, Any]:
    return _execute(sarcasm_agent.detect_dict, payload.model_dump())


@app.post("/mix")
def mix(payload: TextInput) -> dict[str, Any]:
    return _execute(mix_agent.mixRe_dict, payload.model_dump())


@app.post("/judge")
def judge(payload: JudgeInputPayload) -> dict[str, Any]:
    return _execute(judge_agent.judge_dict, payload.model_dump())

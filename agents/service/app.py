"""EmoAgent 情绪分析服务的 FastAPI 应用入口。

本模块提供基于 FastAPI 的 HTTP API 服务，整合了情绪分析流水线中的
所有 Agent（Router、Emotion、Sarcasm、Mix、Judge、Chat、Profile），
对外暴露 RESTful 接口供前端或外部系统调用。

主要接口：
- POST /router    - 文本路由分类
- POST /emotion   - 情感识别
- POST /sarcasm   - 反讽检测
- POST /mix       - 混合情绪分析
- POST /judge     - 最终裁决
- POST /chat      - 情绪聊天回复生成
- POST /profile   - 用户情绪画像（纯统计）
- POST /profile/generate - 用户情绪画像（含 LLM 生成）
- GET  /health    - 健康检查

容错机制：
- 所有 Agent 均支持 fallback 降级，当大模型调用失败时返回保守的默认结果
- 启动时如果 API_KEY 缺失，JudgeAgent 会以纯规则模式运行，其余 Agent 降级为 None
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from dataclasses import asdict

from fastapi import FastAPI, HTTPException

# 动态将 agents 目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.schemas import BaseTextInput  # noqa: E402
from chat_agent.schemas import ChatInput  # noqa: E402
from judge_agent.schemas import JudgeInput  # noqa: E402
from profile_agent.schemas import ProfileInput  # noqa: E402
from .utils import (  # noqa: E402
    router_agent, emotion_agent, sarcasm_agent, mix_agent,
    judge_agent, chat_agent, profile_agent,
    startup_error,
    execute, execute_agent, execute_agent_or_fallback,
    fallback_router, fallback_emotion, fallback_sarcasm,
    fallback_mix, fallback_judge, fallback_profile,
)
from profile_agent import extract_features, build_visualization_data  # noqa: E402

app = FastAPI(title="EmoAgent Service", version="1.0.0")

@app.get("/health")
def health() -> dict[str, Any]:
    """健康检查接口。返回服务运行状态。"""
    if startup_error:
        return {"status": "degraded", "ready": False, "reason": startup_error}
    return {"status": "ok", "ready": True}


@app.post("/router")
def route(payload: BaseTextInput) -> dict[str, Any]:
    """文本路由分类接口。由 Router Agent 判断文本类型。"""
    return execute_agent_or_fallback(router_agent, "route_dict", asdict(payload), fallback_router)


@app.post("/emotion")
def emotion(payload: BaseTextInput) -> dict[str, Any]:
    """情感识别接口。由 Emotion Agent 进行情感识别。"""
    return execute_agent_or_fallback(emotion_agent, "emotionRe_dict", asdict(payload), fallback_emotion)


@app.post("/sarcasm")
def sarcasm(payload: BaseTextInput) -> dict[str, Any]:
    """反讽检测接口。由 Sarcasm Agent 检测反讽。"""
    return execute_agent_or_fallback(sarcasm_agent, "detect_dict", asdict(payload), fallback_sarcasm)


@app.post("/mix")
def mix(payload: BaseTextInput) -> dict[str, Any]:
    """混合情绪分析接口。由 Mix Agent 判断是否包含混合情绪。"""
    return execute_agent_or_fallback(mix_agent, "mixRe_dict", asdict(payload), fallback_mix)


@app.post("/judge")
def judge(payload: JudgeInput) -> dict[str, Any]:
    """最终裁决接口。由 Judge Agent 综合裁决。"""
    return execute_agent_or_fallback(judge_agent, "judge_dict", asdict(payload), fallback_judge)


@app.post("/chat")
def chat(payload: ChatInput) -> dict[str, Any]:
    """聊天回复生成接口。由 Chat Agent 生成回复。"""
    return execute_agent(chat_agent, "chat_dict", asdict(payload))


@app.post("/profile")
def profile(payload: ProfileInput) -> dict[str, Any]:
    """用户情绪画像接口（纯统计，不调用 LLM）。"""
    try:
        records = payload.emotion_records
        features = extract_features(records)
        viz = build_visualization_data(features, records)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        "radar_chart": viz.get("radar_chart", {}),
        "timeline": viz.get("timeline", {}),
        "intensity_distribution": viz.get("intensity_distribution", {}),
    }


@app.post("/profile/generate")
def profile_generate(payload: ProfileInput) -> dict[str, Any]:
    """用户情绪画像接口（含 LLM 生成）。"""
    return execute_agent_or_fallback(
        profile_agent, "generate_dict", asdict(payload), fallback_profile
    )

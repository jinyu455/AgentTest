"""用户画像模块：基于历史情绪记录提取统计特征，并可选地调用 LLM 生成人格画像。"""

from .client import HTTPProfileLLMClient
from .llm_agent import ProfileAgent
from .schemas import EmotionRecordEntry, ProfileInput, ProfileResult
from .feature_extractor import extract_features
from .visualizer import build_visualization_data

__all__ = [
    "HTTPProfileLLMClient",
    "ProfileAgent",
    "ProfileInput",
    "ProfileResult",
    "EmotionRecordEntry",
    "extract_features",
    "build_visualization_data",
]

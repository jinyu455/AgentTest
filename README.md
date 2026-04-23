# EmoAgent Router Agent

当前版本的 `Router Agent` 已改为通过调用大模型进行判断，不再以内置规则作为主逻辑。

它的职责仍然只有两件事：

- 判断输入句子属于 `direct`、`sarcasm_suspected`、`mix` 中的哪一类
- 决定是否需要继续调用 `Sarcasm Agent` 和 `Mix Agent`

## 输入格式

```json
{
  "id": "msg_001",
  "user_id": "u_1001",
  "text": "太好了，周末又能继续改需求了。",
  "source": "chat",
  "created_at": "2026-03-24T14:00:00"
}
```

## 输出格式

```json
{
  "sample_type": "sarcasm_suspected",
  "need_sarcasm_check": true,
  "need_mix_check": false,
  "routing_reason": "句子表面正向，但事件语境明显负向，疑似反讽。",
  "evidence": ["正向词: 太好了", "负向场景: 周末继续改需求"]
}
```

## 配置占位

在 [router_agent/client.py](d:/PracticalTraining/Agenttest/EmoAgent/router_agent/client.py) 里保留了占位配置：

- `base_url = "https://your-llm-service.example.com/v1/chat/completions"`
- `api_key = "YOUR_API_KEY"`
- `model = "YOUR_MODEL_NAME"`

你后续只需要替换成真实值即可。

## 快速使用

```python
from router_agent import HTTPRouterLLMClient, LLMConfig, RouterAgent

config = LLMConfig(
    base_url="https://your-llm-service.example.com/v1/chat/completions",
    api_key="YOUR_API_KEY",
    model="YOUR_MODEL_NAME",
)
client = HTTPRouterLLMClient(config)
agent = RouterAgent(client=client)

result = agent.route_dict(
    {
        "id": "msg_001",
        "user_id": "u_1001",
        "text": "太好了，周末又能继续改需求了。",
        "source": "chat",
        "created_at": "2026-03-24T14:00:00",
    }
)
```

## 说明

- `router_agent/llm_agent.py` 负责 Router Agent 的 prompt 和结果校验
- `router_agent/client.py` 负责通用 HTTP 调用
- 当前请求体按 OpenAI 兼容风格组织，便于后续替换不同模型服务
- 单元测试使用 fake client，不依赖真实网络

## Emotion Agent

当前已新增 `Emotion Agent`，用于第一版表层情绪识别。它的职责是：

- 识别主情绪
- 给出情绪强度分数
- 输出结构化语言特征
- 给出初步解释

第一版标签固定为 6 类：

- 开心
- 悲伤
- 愤怒
- 焦虑
- 厌烦
- 中性

输出格式：

```json
{
  "tokens": ["太好了", "周末", "又", "能", "继续", "改", "需求"],
  "emotion_words": ["太好了"],
  "degree_words": [],
  "negation_words": [],
  "contrast_words": [],
  "emotion": "开心",
  "intensity": 62,
  "confidence": 0.61,
  "reason": "文本表面存在明显正向表达“太好了”，情绪方向初步判为正向"
}
```

快速使用：

```python
from emotion_agent import EmotionAgent, HTTPEmotionLLMClient, LLMConfig

config = LLMConfig(
    base_url="https://your-llm-service.example.com/v1/chat/completions",
    api_key="YOUR_API_KEY",
    model="YOUR_MODEL_NAME",
)
client = HTTPEmotionLLMClient(config)
agent = EmotionAgent(client=client)

result = agent.analyze_dict(
    {
        "id": "msg_001",
        "user_id": "u_1001",
        "text": "太好了，周末又能继续改需求了。",
        "source": "chat",
        "created_at": "2026-03-24T14:00:00",
    }
)
```

说明：

- `emotion_agent/llm_agent.py` 负责 Emotion Agent 的 prompt 和结果校验
- `emotion_agent/client.py` 负责通用 HTTP 调用，占位配置与 Router Agent 保持一致
- `examples/emotion_demo.py` 可以打印将要发送给大模型的 messages
- 单元测试使用 fake client，不依赖真实网络

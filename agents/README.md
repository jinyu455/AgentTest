# EmoAgent Agents

这里放 Python 侧的 Agent 实现、FastAPI 服务、示例和测试。

## 目录

```text
agents/
  chat_agent/
  emotion_agent/
  judge_agent/
  mix_agent/
  router_agent/
  sarcasm_agent/
  service/
    app.py
  examples/
  tests/
  requirements.txt
```

## 首次拉取项目

在仓库根目录准备 `.env`：

```env
API_KEY=你的deepseek服务密钥
```

如需更换大模型，可通过环境变量配置：

```env
LLM_BASE_URL=https://api.deepseek.com/v1/chat/completions
LLM_MODEL=deepseek-chat
```

然后进入 `agents` 目录，创建虚拟环境并安装依赖：

```powershell
cd agents
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 后续启动服务

后续不用重复创建虚拟环境，只需要进入 `agents` 目录、激活环境并启动 FastAPI：

```powershell
cd agents
.\.venv\Scripts\Activate.ps1
uvicorn service.app:app --reload
```

默认服务地址：

```text
http://127.0.0.1:8000
```

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

## 可用接口

- `GET /health`
- `POST /router`
- `POST /emotion`
- `POST /sarcasm`
- `POST /mix`
- `POST /judge`
- `POST /chat`

## 文本分析请求示例

`/router`、`/emotion`、`/sarcasm`、`/mix` 接收相同的文本请求体：

```json
{
  "id": "msg_001",
  "user_id": "u_1001",
  "text": "太好了，周末又能继续改需求了。",
  "source": "chat",
  "created_at": "2026-05-09T14:00:00",
  "metadata": {}
}
```

PowerShell 示例：

```powershell
curl -Method Post `
  -Uri http://127.0.0.1:8000/router `
  -ContentType "application/json; charset=utf-8" `
  -Body '{
    "id": "msg_001",
    "user_id": "u_1001",
    "text": "太好了，周末又能继续改需求了。",
    "source": "chat",
    "created_at": "2026-05-09T14:00:00",
    "metadata": {}
  }'
```

把 URL 换成下面任一接口即可调用对应 Agent：

```text
http://127.0.0.1:8000/emotion
http://127.0.0.1:8000/sarcasm
http://127.0.0.1:8000/mix
```

## Judge 请求示例

`/judge` 接收上游 Agent 的结构化结果：

```json
{
  "text": "太好了，周末又能继续改需求了。",
  "router_result": {
    "sample_type": "sarcasm_suspected",
    "need_sarcasm_check": true,
    "need_mix_check": false,
    "routing_reason": "句子表面正向，但事件语境明显负向，疑似反讽。",
    "evidence": ["正向词: 太好了", "负向场景: 周末继续改需求"]
  },
  "emotion_result": {
    "emotion": "开心",
    "intensity": 62,
    "confidence": 0.72,
    "reason": "文本表面包含明显正向表达。"
  },
  "sarcasm_result": {
    "is_sarcasm": true,
    "surface_emotion": "开心",
    "true_emotion": "厌烦",
    "revised_intensity": 74,
    "confidence": 0.86,
    "reason": "正向词与负向工作场景形成反差。"
  },
  "mix_result": null
}
```

## Chat 请求示例

`/chat` 用于根据用户文本、情绪分析结果和可选历史消息生成聊天回复：

```json
{
  "text": "太好了，周末又能继续改需求了。",
  "user_id": "u_1001",
  "conversation_id": "c_001",
  "judge_result": {
    "final_emotion": "厌烦",
    "secondary_emotion": null,
    "final_intensity": 74,
    "final_confidence": 0.86,
    "is_sarcasm": true,
    "is_mixed": false,
    "reason": "正向词与负向工作场景形成反差。"
  },
  "history": [
    {
      "role": "user",
      "content": "最近工作有点多。"
    },
    {
      "role": "assistant",
      "content": "听起来你这段时间一直在扛很多事情。"
    }
  ],
  "metadata": {}
}
```

PowerShell 示例：

```powershell
curl -Method Post `
  -Uri http://127.0.0.1:8000/chat `
  -ContentType "application/json; charset=utf-8" `
  -Body '{
    "text": "太好了，周末又能继续改需求了。",
    "user_id": "u_1001",
    "conversation_id": "c_001",
    "judge_result": {
      "final_emotion": "厌烦",
      "secondary_emotion": null,
      "final_intensity": 74,
      "final_confidence": 0.86,
      "is_sarcasm": true,
      "is_mixed": false,
      "reason": "正向词与负向工作场景形成反差。"
    },
    "history": [],
    "metadata": {}
  }'
```

返回示例：

```json
{
  "reply": "听起来你其实挺疲惫，也有点无奈。周末还被需求占着，确实会让人烦。",
  "tone": "supportive",
  "risk_hint": "none",
  "suggested_actions": ["先把最急的事列出来", "给自己留一点休息时间"],
  "reason": "用户文本包含反讽和工作压力，适合支持性回应。"
}
```

## 运行测试

```powershell
cd agents
.\.venv\Scripts\Activate.ps1
python -m unittest discover tests
```

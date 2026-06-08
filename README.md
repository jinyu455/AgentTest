# EmoAgent

EmoAgent 按运行边界拆成两个子项目：

```text
EmoAgent/
  agents/     Python Agents、FastAPI Agent 服务、示例和测试
  backend/    Spring Boot 后端服务
```

`agents` 负责调用大模型，并提供 `/router`、`/emotion`、`/sarcasm`、`/mix`、`/judge` 等 FastAPI 接口。

`backend` 负责对外提供统一后端接口，并编排调用 Python Agent 服务。

## 首次拉取项目

在仓库根目录准备 `.env`：

```env
API_KEY=你的deepseek服务密钥(如需更换其他大模型，更换agents/servic/app.py里的base_url与model)
```

初始化 Python Agent 环境：

```powershell
cd agents
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 后续启动

先启动 Python Agent 服务：

```powershell
cd agents
.\.venv\Scripts\Activate.ps1
uvicorn service.app:app --reload
```

再启动 Spring Boot 后端：

```powershell
cd backend
.\mvnw.cmd spring-boot:run
```

再启动 Web 前端：
```powershell
cd frontend
python -m http.server 5173
```

默认地址：

- Agent 服务: `http://127.0.0.1:8000`
- 后端服务: `http://127.0.0.1:8080`

## 配置

Agent 服务会优先读取仓库根目录的 `.env`。

可选环境变量：

- `LLM_BASE_URL`
- `LLM_MODEL`

## 更多说明

- [agents/README.md](agents/README.md)
- [backend/README.md](backend/README.md)

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

前端默认使用当前页面同源地址作为 API 基础地址，这是为了配合服务器部署时的 Nginx 转发：线上由 Nginx 接收同源 `/api/...` 请求，并代理到 Spring Boot 后端。
本地使用 Python 静态服务器访问 `http://localhost:5173` 时，`python -m http.server` 不会代理 `/api` 到后端，因此需要在浏览器控制台执行一次：

```javascript
localStorage.setItem("emoagent.api.base", "http://127.0.0.1:8080")
location.reload()
```

这样本地前端会请求 `http://127.0.0.1:8080/api/...`。如需恢复默认同源地址，可执行：

```javascript
localStorage.removeItem("emoagent.api.base")
location.reload()
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

# EmoAgent Frontend

轻量静态聊天前端，默认请求 `http://localhost:8080/api/emotion`。

## 启动

```powershell
cd frontend
python -m http.server 5173
```

然后打开 `http://localhost:5173`。

后端和 Agent 仍按项目根目录 `README.md` 的方式启动：

```powershell
cd agents
.\.venv\Scripts\Activate.ps1
uvicorn service.app:app --reload
```

```powershell
cd backend
.\mvnw.cmd spring-boot:run
```

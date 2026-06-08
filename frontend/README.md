# EmoAgent Frontend

轻量静态聊天前端，默认 API 地址为 `http://localhost:8080`，接口路径由前端统一拼接。

## 目录结构

```text
frontend/
  index.html              # 页面入口，只负责加载 partial 和脚本
  app.js                  # 业务交互逻辑
  scripts/
    partials.js           # 静态 HTML 片段加载器
  partials/
    auth.html             # 登录 / 注册页结构
    workspace.html        # 聊天工作台结构
  styles.css              # 样式入口，集中 import 分层样式
  styles/
    tokens.css            # 变量、基础 reset
    auth.css              # 登录 / 注册页样式
    layout.css            # 三栏布局和通用组件样式
    sidebar.css           # 左侧栏、历史会话、设置和状态样式
    chat.css              # 聊天区、消息、输入框样式
    insights.css          # 情绪洞察和画像卡片样式
    responsive.css        # 响应式规则
```

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

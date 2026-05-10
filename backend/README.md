# EmoAgent Backend

`backend` 是 EmoAgent 项目的 Spring Boot 后端服务，负责接收前端请求，并调用 Python FastAPI Agent 服务完成情绪识别流程。

## 技术栈

- Java 17
- Spring Boot
- Spring WebMVC
- Spring Validation
- Spring Boot Actuator
- Maven Wrapper

## 服务职责

后端对前端提供统一接口，并在内部编排多个 Agent 的调用流程。

调用链路：

```text
前端
  -> Spring Boot Backend
  -> Python FastAPI Agent Service
  -> LLM
```

当前后端主要做以下事情：

- 接收文本情绪分析请求
- 调用 Python FastAPI Agent 服务
- 编排 Router、Emotion、Sarcasm、Mix、Judge 流程
- 返回完整情绪分析结果

## 目录结构

```text
backend/
  pom.xml
  mvnw.cmd
  src/
    main/
      java/
        com/emoagent/backend/
          BackendApplication.java
          client/
            AgentClient.java
          config/
            RestClientConfig.java
          controller/
            EmotionAnalysisController.java
          dto/
            AnalyzeResponse.java
            JudgeRequest.java
            TextAnalyzeRequest.java
          exception/
            GlobalExceptionHandler.java
          service/
            EmotionAnalysisService.java
      resources/
        application.yml
    test/
      java/
        com/emoagent/backend/
          BackendApplicationTests.java
```

目录说明：

- `controller`：接收前端 HTTP 请求
- `service`：编排业务流程
- `client`：调用 Python FastAPI Agent 服务
- `dto`：定义请求和响应数据结构
- `config`：放置 Spring 配置类
- `exception`：统一异常处理
- `resources/application.yml`：后端配置文件

## 启动前准备

需要先启动 Python Agent 服务。

进入项目根目录下的 `agents` 目录：

```powershell
cd D:\PracticalTraining\Agenttest\EmoAgent\agents
```

启动 FastAPI：

```powershell
uvicorn service.app:app --reload
```

默认服务地址：

```text
http://127.0.0.1:8000
```

可以先检查 Python Agent 服务：

```text
http://127.0.0.1:8000/health
```

## 启动后端

进入 `backend` 目录：

```powershell
cd D:\PracticalTraining\Agenttest\EmoAgent\backend
```

使用 Maven Wrapper 启动：

```powershell
.\mvnw.cmd spring-boot:run
```

默认服务地址：

```text
http://127.0.0.1:8080
```

检查 Spring Boot 健康状态：

```text
http://127.0.0.1:8080/actuator/health
```

## 配置说明

配置文件：

```text
src/main/resources/application.yml
```

当前配置：

```yaml
server:
  port: 8080

spring:
  servlet:
    encoding:
      charset: UTF-8
      enabled: true
      force: true

emo-agent:
  base-url: http://127.0.0.1:8000
```

配置含义：

- `server.port`：Spring Boot 后端端口
- `server.servlet.encoding`：强制使用 UTF-8 编码
- `emo-agent.base-url`：Python FastAPI Agent 服务地址

如果 8080 端口被占用，可以改成：

```yaml
server:
  port: 8081
```

## 接口说明

### Agent 服务健康检查

```http
GET /api/emotion/health
```

该接口会转调 Python FastAPI 的 `/health`，用于确认 Spring Boot 后端能否连接 Python Agent 服务。

请求示例：

```text
http://127.0.0.1:8080/api/emotion/health
```

返回示例：

```json
{
  "status": "ok",
  "ready": true
}
```

### 情绪分析

```http
POST /api/emotion/analyze
```

该接口是前端主要调用的统一情绪分析接口。

后端内部调用流程：

```text
1. 调用 /router 判断文本类型
2. 调用 /emotion 获取表层情绪
3. 如果需要反讽检测，调用 /sarcasm
4. 如果需要混合情绪检测，调用 /mix
5. 调用 /judge 输出最终裁决
```

请求头：

```http
Content-Type: application/json; charset=utf-8
```

请求体示例：

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

返回体示例：

```json
{
  "text": "太好了，周末又能继续改需求了。",
  "router_result": {
    "sample_type": "sarcasm_suspected",
    "need_sarcasm_check": true,
    "need_mix_check": false,
    "routing_reason": "...",
    "evidence": []
  },
  "emotion_result": {
    "emotion": "开心",
    "intensity": 62,
    "confidence": 0.61,
    "reason": "..."
  },
  "sarcasm_result": {
    "is_sarcasm": true,
    "surface_emotion": "开心",
    "true_emotion": "厌烦",
    "revised_intensity": 75,
    "confidence": 0.9,
    "reason": "..."
  },
  "mix_result": null,
  "judge_result": {
    "final_emotion": "厌烦",
    "secondary_emotion": null,
    "final_intensity": 75,
    "final_confidence": 0.9,
    "is_sarcasm": true,
    "is_mixed": false,
    "reason": "..."
  }
}
```

## 使用 Apifox 测试

推荐使用 Apifox 或 Postman 测试中文接口，避免 PowerShell 终端编码导致中文显示乱码。

测试完整分析接口：

- Method：`POST`
- URL：`http://127.0.0.1:8080/api/emotion/analyze`
- Header：`Content-Type: application/json; charset=utf-8`
- Body 类型：`JSON`

Body：

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

## 运行测试

在 `backend` 目录执行：

```powershell
.\mvnw.cmd test
```

看到以下结果表示测试通过：

```text
BUILD SUCCESS
```

## 常见问题

### mvn 命令不存在

本项目使用 Maven Wrapper，不要求全局安装 Maven。

请使用：

```powershell
.\mvnw.cmd spring-boot:run
```

而不是：

```powershell
mvn spring-boot:run
```

### 8080 端口被占用

查看 8080 端口占用：

```powershell
netstat -ano | findstr :8080
```

查看进程：

```powershell
tasklist | findstr 进程ID
```

结束进程：

```powershell
taskkill /PID 进程ID /F
```

也可以修改 `application.yml` 中的端口：

```yaml
server:
  port: 8081
```

### PowerShell 返回中文乱码

PowerShell 可能因为终端编码导致中文显示乱码。建议优先使用 Apifox 或 Postman 测试接口。

如果必须使用 PowerShell，可以先执行：

```powershell
chcp 65001
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

然后请求时显式指定 UTF-8：

```powershell
-ContentType "application/json; charset=utf-8"
```

### Python Agent 服务未启动

如果调用 `/api/emotion/health` 或 `/api/emotion/analyze` 返回 502，通常是 Python Agent 服务没有启动。

请先启动：

```powershell
cd D:\PracticalTraining\Agenttest\EmoAgent\agents
uvicorn service.app:app --reload
```

第 1 步：启动入口

  BackendApplication.java  ← 3 行代码，不用细看，知道从这里启动就行

  第 2 步：配置（决定程序怎么运行）

  application.yml           ← 读这个！所有配置都在这里
  config/WebConfig.java     ← CORS 跨域 + JWT 拦截器注册
  config/JwtConfig.java     ← JWT 密钥配置
  config/RestClientConfig.java ← 几乎不用管

  第 3 步：数据结构（程序处理什么数据）

  dto/RegisterRequest.java      ← 登录注册的请求格式（最简单的 DTO）
  dto/AuthRequest.java
  dto/AuthResponse.java
  dto/TextAnalyzeRequest.java   ← 情绪分析的请求格式
  dto/ChatRequest.java          ← 聊天的请求格式

  entity/User.java              ← 用户表（最简单的 Entity）
  entity/Conversation.java      ← 对话表
  entity/ChatMessage.java       ← 消息表
  entity/EmotionRecord.java     ← 情绪记录表

  第 4 步：数据库访问（数据怎么存取）

  repository/UserRepository.java          ← 最简单的，2 个方法
  repository/ConversationRepository.java  ← 按用户查对话
  repository/ChatMessageRepository.java   ← 按对话查消息
  repository/EmotionRecordRepository.java ← 按用户查情绪记录

  第 5 步：鉴权（请求怎么验证身份）

  filter/JwtAuthFilter.java  ← 每个请求先过这里，提取 token → userId/role
  service/AuthService.java   ← 注册、登录、JWT 生成、验证码生成
  controller/AuthController.java ← 3 个端点：register/login/captcha

  第 6 步：核心业务（最重要的逻辑）

  service/EmotionAnalysisService.java  ← 重点！完整分析流水线：
                                         router → emotion → sarcasm/mix → judge → chat
  service/ChatPersistenceService.java  ← 对话和消息的持久化逻辑
  client/AgentClient.java              ← 调用 Python Agent 的 HTTP 客户端

  第 7 步：API 端点（对外暴露什么接口）

  controller/EmotionAnalysisController.java ← 重点！所有业务端点
                                              注意 userId 从 JWT 获取的部分

  第 8 步：异常处理

  exception/GlobalExceptionHandler.java ← 全局兜底，异常转 HTTP 状态码

---

  总结阅读顺序

1. application.yml          （配置）
2. dto/*.java               （数据格式）
3. entity/*.java            （表结构）
4. repository/*.java        （数据库查询）
5. filter/JwtAuthFilter     （鉴权拦截）
6. service/AuthService      （登录注册）
7. controller/AuthController（认证端点）
8. client/AgentClient       （调 Python）
9. service/EmotionAnalysisService （核心流水线）
10. service/ChatPersistenceService（持久化）
11. controller/EmotionAnalysisController（业务端点）
12. exception/GlobalExceptionHandler（异常处理）

  按这个顺序读，每个文件都不长，2-3 小时能过一遍。

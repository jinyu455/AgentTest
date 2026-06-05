### 一、admin和user

1. 管理员账号：在application.yaml配置，不能注册

   - username:admin
   - password:admin123
2. 注册只能注册user角色

### 二、JWT

1. JWT：配置后检测token有没有过期，是不是合法，所有对私有接口的访问必须用JWT检查

- 检测token的方法：
  - JWT Token = 头部 + 内容 + 签名,用 2 个点 `.` 连起来的一串字符串:aaaaa.bbbbb.ccccc
    - 头部：告诉程序：
      - 我用的是 **JWT**
      - 加密算法是 **HS256**
    - 内容：这里放**你想存的用户信息**（不能存密码会泄露），比如：
      - userId：1001
      - username：张三
      - 登录时间
      - 过期时间
    - 签名：加密算法( 第一段Header + 第二段Payload + 你的 secret 钥匙 )计算得到

2. 具体实现：JwtAuthFilter.java：
   1. 前端发送请求，调用POST /api/emotion/chat接口，请求头带上Authorization: Bearer 你的token字符串
   2. preHandle：检查有没有token与token是不是以Bearer 开头，不是直接返回未鉴权；验证token是不是有效，然后request中贴上user_name,role,失败返回401

### 三、登录和注册（注册后直接跳转登录，不勾选自动登录token有效期5min，勾选有效期7）

1. 注册：前端必须给后端传这几个字段：用户名、密码、验证码、验证码 ID
2. 登录request：前端必须给后端传这几个字段：用户名、密码
3. 登录response:登录成功后，后端给前端发送这几个字段：jwt_token(前端以后请求都要这个token)、userid、username、role

### 四、DTO（数据传输对象）

> 所有 DTO 使用 Java record 定义（不可变、自动 getter/equals/hashCode），字段上的 `@NotBlank` 表示前端必须传，`@JsonProperty("xxx")` 表示 JSON 中的字段名映射。

#### 1. RegisterRequest — 注册请求

| 字段        | JSON 名      | 类型   | 必填 | 说明                                      |
| ----------- | ------------ | ------ | ---- | ----------------------------------------- |
| username    | username     | String | ✅   | 用户名                                    |
| password    | password     | String | ✅   | 密码                                      |
| captchaCode | captcha_code | String | ✅   | 验证码                                    |
| captchaKey  | captcha_key  | String | ✅   | 验证码 key（用于定位 Redis 中存的验证码） |

#### 2. AuthRequest — 登录请求

| 字段     | JSON 名  | 类型   | 必填 | 说明   |
| -------- | -------- | ------ | ---- | ------ |
| username | username | String | ✅   | 用户名 |
| password | password | String | ✅   | 密码   |

#### 3. AuthResponse — 登录响应

| 字段     | JSON 名  | 类型   | 说明                              |
| -------- | -------- | ------ | --------------------------------- |
| token    | token    | String | JWT Token，前端后续所有请求需携带 |
| userId   | user_id  | String | 用户 ID                           |
| username | username | String | 用户名                            |
| role     | role     | String | 角色（admin / user）              |

#### 4. TextAnalyzeRequest — 情绪分析请求

| 字段      | JSON 名    | 类型                | 必填 | 说明               |
| --------- | ---------- | ------------------- | ---- | ------------------ |
| id        | id         | String              | ✅   | 记录 ID            |
| userId    | user_id    | String              | ✅   | 用户 ID            |
| text      | text       | String              | ✅   | 要分析情绪的文本   |
| source    | source     | String              | ✅   | 来源标识           |
| createdAt | created_at | String              | ✅   | 创建时间           |
| metadata  | metadata   | Map<String, Object> | ❌   | 附加元数据（可选） |

#### 5. ChatRequest — 聊天请求

| 字段           | JSON 名         | 类型                      | 必填 | 说明                                |
| -------------- | --------------- | ------------------------- | ---- | ----------------------------------- |
| text           | text            | String                    | ✅   | 用户发送的消息                      |
| userId         | user_id         | String                    | ✅   | 用户 ID                             |
| conversationId | conversation_id | String                    | ❌   | 对话 ID（首次对话为空，后端会新建） |
| judgeResult    | judge_result    | Map<String, Object>       | ❌   | 情绪分析结果（前端可携带）          |
| history        | history         | List<Map<String, Object>> | ❌   | 聊天历史上下文                      |
| metadata       | metadata        | Map<String, Object>       | ❌   | 附加元数据（可选）                  |

### 五、Entity（数据库实体）

> 所有 Entity 使用 JPA 注解映射到数据库表，ID 均为 String 类型（UUID），使用 `Instant` 表示时间戳。

#### 1. User — 用户表 `users`

| 字段         | 数据库列名    | 类型    | 约束             | 说明                 |
| ------------ | ------------- | ------- | ---------------- | -------------------- |
| id           | id            | String  | PK, 不可修改     | 用户 ID              |
| username     | username      | String  | NOT NULL, UNIQUE | 用户名               |
| passwordHash | password_hash | String  | NOT NULL         | 密码哈希（不存明文） |
| salt         | salt          | String  | NOT NULL         | 加盐值               |
| role         | role          | String  | NOT NULL         | 角色（admin / user） |
| createdAt    | created_at    | Instant | NOT NULL         | 创建时间             |

#### 2. Conversation — 对话表 `conversations`

| 字段      | 数据库列名 | 类型    | 约束         | 说明         |
| --------- | ---------- | ------- | ------------ | ------------ |
| id        | id         | String  | PK, 不可修改 | 对话 ID      |
| userId    | user_id    | String  | NOT NULL     | 所属用户 ID  |
| title     | title      | String  | 可选         | 对话标题     |
| createdAt | created_at | Instant | NOT NULL     | 创建时间     |
| updatedAt | updated_at | Instant | NOT NULL     | 最后更新时间 |

- `touch(Instant updatedAt)` 方法用于更新 `updatedAt` 时间戳

#### 3. ChatMessage — 消息表 `chat_messages`

| 字段           | 数据库列名      | 类型         | 约束         | 说明                     |
| -------------- | --------------- | ------------ | ------------ | ------------------------ |
| id             | id              | String       | PK, 不可修改 | 消息 ID                  |
| conversationId | conversation_id | String       | NOT NULL     | 所属对话 ID              |
| role           | role            | String       | NOT NULL     | 角色（user / assistant） |
| content        | content         | String (Lob) | NOT NULL     | 消息内容（长文本）       |
| createdAt      | created_at      | Instant      | NOT NULL     | 创建时间                 |

#### 4. EmotionRecord — 情绪记录表 `emotion_records`

| 字段             | 数据库列名        | 类型         | 约束         | 说明                      |
| ---------------- | ----------------- | ------------ | ------------ | ------------------------- |
| id               | id                | String       | PK, 不可修改 | 记录 ID                   |
| conversationId   | conversation_id   | String       | NOT NULL     | 所属对话 ID               |
| messageId        | message_id        | String       | NOT NULL     | 关联消息 ID               |
| finalEmotion     | final_emotion     | String       | 可选         | 最终情绪标签              |
| secondaryEmotion | secondary_emotion | String       | 可选         | 次要情绪标签              |
| finalIntensity   | final_intensity   | Integer      | 可选         | 情绪强度（数值）          |
| finalConfidence  | final_confidence  | Double       | 可选         | 情绪置信度（0~1）         |
| sarcasm          | is_sarcasm        | Boolean      | 可选         | 是否检测到反讽            |
| mixed            | is_mixed          | Boolean      | 可选         | 是否为混合情绪            |
| rawAnalysisJson  | raw_analysis_json | String (Lob) | NOT NULL     | 大模型返回的原始分析 JSON |
| createdAt        | created_at        | Instant      | NOT NULL     | 创建时间                  |

### 六、hash和salt

1. 利用sha256加密算法，并增加随机盐值，两个人即使密码相同，加密后也不同   String hash = sha256(salt + 密码)

### 七、验证码

1. 存储在内存中，5分钟过期
2. 生成 4 位随机验证码，生成一个唯一 key，生成带干扰线、彩色、旋转文字的  SVG 图片并转成 Base64 字符串返回，前端直接放进 img src 就能显示。

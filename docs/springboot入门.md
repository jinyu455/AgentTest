# Spring Boot 基础知识（结合emoagent后端项目）

## 一、Spring Boot 是什么

Spring Boot = Spring 框架 + 自动配置 + 内嵌服务器。无需手动编写XML配置，依靠注解即可快速搭建Web服务。

类比：Spring是发动机，Spring Boot是组装完成的整车，开发者只需专注编写业务逻辑。

## 二、项目目录结构

```
backend/
├── pom.xml                    ← 依赖清单（Maven自动下载所需依赖）
├── mvnw / mvnw.cmd            ← Maven启动脚本项目自动生成，无需修改
├── src/main/java/             ← Java业务源码目录
│   └── com/emoagent/backend/
│       ├── BackendApplication.java   ← 项目启动入口
│       ├── controller/       ← ① 接收前端HTTP请求（门卫层）
│       ├── service/          ← ② 编写核心业务逻辑（大脑层）
│       ├── repository/       ← ③ 数据库CRUD操作（仓库管理员）
│       ├── entity/           ← ④ 映射数据库表结构（数据表图纸）
│       ├── dto/              ← ⑤ 封装请求、响应数据格式（数据信封）
│       ├── client/           ← ⑥ 远程调用外部服务（对接Python Agent）
│       ├── config/           ← ⑦ 全局配置类（CORS跨域、JWT、RestClient配置）
│       ├── filter/           ← ⑧ 请求过滤器（JWT登录鉴权拦截）
│       └── exception/        ← ⑨ 全局统一异常捕获处理
├── src/main/resources/
│   └── application.yml       ← 项目主配置文件（端口、数据库、密钥等）
└── src/test/                 ← 单元测试代码目录
```

## 三、核心概念：三层架构

请求流转链路：

```
前端HTTP请求
     ↓
Controller（controller/）  ← 接收请求、参数校验、调用Service
     ↓
Service（service/）         ← 实现业务逻辑，操作数据库/调用第三方服务
     ↓
Repository（repository/）  ← 执行数据库增删改查
     ↓
Entity（entity/）           ← Java实体映射数据库数据表
```

### 项目真实调用示例

前端发起：`POST /api/emotion/chat`

```
→ EmotionAnalysisController.chat()
  → EmotionAnalysisService.chat()
    → ChatPersistenceService.startTurn()     ← 存储用户提问至数据库
    → AgentClient.router() / emotion()      ← 远程调用Python智能体服务
    → ChatPersistenceService.saveAssistantMessage() ← 存储AI回复
→ 封装ChatResponse返回前端
```

## 四、代码生成/手写区分表

| 文件                    | 来源     | 说明                                    |
| ----------------------- | -------- | --------------------------------------- |
| BackendApplication.java | 固定模板 | SpringBoot项目启动类，代码固定          |
| pom.xml                 | 手写     | 管理项目所有第三方依赖包                |
| application.yml         | 手写     | 配置服务端口、数据库连接、密钥等        |
| entity/*.java           | 手写     | 定义数据表映射，@Entity注解实现自动建表 |
| repository/*.java       | 半自动   | 仅定义接口，根据方法名自动生成SQL       |
| dto/*.java              | 手写     | 自定义接口入参、出参结构                |
| service/*.java          | 手写     | 项目核心业务代码                        |
| controller/*.java       | 手写     | 定义API接口地址与请求方式               |
| config/*.java           | 手写     | 各类组件配置（跨域、过滤器等）          |
| filter/*.java           | 手写     | 自定义请求拦截逻辑                      |
| mvnw、.mvn文件夹        | 自动生成 | Maven包装脚本，无需修改                 |
| target/                 | 自动生成 | 编译打包输出目录                        |

## 五、核心注解速查表

| 注解                   | 作用                                  | 项目使用位置                              |
| ---------------------- | ------------------------------------- | ----------------------------------------- |
| @SpringBootApplication | 标识项目启动类                        | BackendApplication.java                   |
| @RestController        | 标识当前类为REST接口控制器            | AuthController、EmotionAnalysisController |
| @GetMapping("/xxx")    | 注册GET请求接口                       | AuthController.login()                    |
| @PostMapping("/xxx")   | 注册POST请求接口                      | AuthController.register()                 |
| @RequestBody           | 解析请求体JSON转为Java对象            | 接口入参接收JSON数据                      |
| @RequestParam("key")   | 从URL拼接参数中取值                   | `?user_id=123`格式参数接收              |
| @PathVariable          | 从URL路径占位符取值                   | `/conversations/{id}`路径参数           |
| @Valid                 | 开启请求参数校验，配合@NotBlank等注解 | 入参非空、长度校验                        |
| @Service               | 标识业务层Bean                        | AuthService、EmotionAnalysisService       |
| @Repository            | 标识DAO数据库访问层                   | UserRepository等仓储接口                  |
| @Entity                | 实体类映射数据库单表                  | User、Conversation实体类                  |
| @Column(name="xxx")    | 指定实体字段对应数据库列名            | User实体映射user_id字段                   |
| @Id                    | 标记实体主键字段                      | 所有Entity主键属性                        |
| @Value("${xxx}")       | 从yml配置文件读取配置项               | JwtConfig读取jwt.secret密钥               |
| @PostConstruct         | Bean初始化后自动执行一次方法          | AuthService.initAdminUser()初始化管理员   |
| @Transactional         | 声明方法受事务管理                    | ChatPersistenceService数据库事务          |

## 六、源码学习优先级（由浅入深）

### 第1步：弄懂接口接收请求

- `controller/AuthController.java`：最简登录注册控制器
- `dto/RegisterRequest.java`：注册接口请求体结构

### 第2步：弄懂业务处理逻辑

- `service/AuthService.java`：登录、注册、JWT签发、验证码全流程

### 第3步：弄懂数据持久化

- `entity/User.java`：用户数据表映射
- `repository/UserRepository.java`：用户数据查询操作

### 第4步：弄懂全局配置

- `application.yml`：项目所有配置集中管理
- `config/WebConfig.java`：跨域配置、JWT过滤器注册

### 第5步：弄懂复杂业务全链路

- `controller/EmotionAnalysisController.java`：聊天接口入口
- `service/EmotionAnalysisService.java`：情绪聊天完整业务流水线

## 七、项目启动运行命令

```bash
# 进入项目后端目录
cd backend

# 编译项目
./mvnw compile

# 直接启动项目（默认占用8080端口）
./mvnw spring-boot:run

# 打包部署方式
./mvnw package
java -jar target/backend-0.0.1-SNAPSHOT.jar
```

### 启动后可用接口地址

1. 获取验证码：`http://localhost:8080/api/auth/captcha`
2. 用户登录：`http://localhost:8080/api/auth/login`
3. AI情绪聊天（需携带JWT Token）：`http://localhost:8080/api/emotion/chat`

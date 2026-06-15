# Emotion 情绪感知 AI 对话系统 — 汇报文档

---

## 一、项目概述

Emotion 是一个基于多 Agent 架构的情绪感知 AI 对话系统。系统能够实时分析用户输入文本的情绪状态（包括讽刺、混合情绪等复杂场景），并基于情绪分析结果生成更具同理心的对话回复。

**技术栈**：

| 层级     | 技术                               |
| -------- | ---------------------------------- |
| 前端     | 原生 HTML/CSS/JavaScript（无框架） |
| 后端     | Spring Boot 3（Java 17）           |
| AI Agent | Python FastAPI + DeepSeek/Qwen LLM |
| 数据库   | SQLite                             |
| 部署     | Nginx + GitHub Actions CI/CD       |

---

## 二、Agent 流水线设计

### 2.1 整体架构

系统的核心是多 Agent 协作的情绪分析流水线，共 5 个 Agent 协同工作：

```
用户输入文本
    │
    ▼
┌──────────────────┐
│  ① Router Agent   │ ← 分流：判断文本类型
│  （每次都调用）    │    direct / sarcasm_suspected / mix
└────────┬─────────┘
         │
    ┌────┴────┐
    │ 直接路由 │──→ 只进 Emotion Agent
    │ 讽刺路由 │──→ 进 Emotion + Sarcasm Agent
    │ 混合路由 │──→ 进 Emotion + Mix Agent
    └────┬────┘
         │
         ▼
┌──────────────────┐
│  ② Emotion Agent  │ ← 表层情绪检测（每次都调用）
│  输出：情绪标签    │    9 类情绪 + 强度 0-100 + 置信度 0-1
└────────┬─────────┘
         │
    ┌────┴──────────────────┐
    │                       │
    ▼                       ▼
┌──────────────┐    ┌──────────────┐
│ ③ Sarcasm    │    │ ④ Mix        │
│    Agent     │    │    Agent     │
│ （条件调用）  │    │ （条件调用）  │
│ 讽刺检测      │    │ 混合情绪检测  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
       ┌──────────────────┐
       │  ⑤ Judge Agent    │ ← 最终仲裁
       │  规则优先 + LLM 兜底 │    加权合并，输出最终结果
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │  ⑥ Chat Agent     │ ← 基于情绪上下文生成回复
       └──────────────────┘
```

#### 部署模型配置

服务器实际部署使用**阿里云千问（Qwen）系列模型**：

| Agent              | 默认模型       | 说明                     |
| ------------------ | -------------- | ------------------------ |
| Router Agent       | `qwen-flash`   | 轻量快速，负责文本分流   |
| Emotion Agent      | `qwen-flash`   | 表层情绪检测             |
| Sarcasm Agent      | `qwen-flash`   | 讽刺检测                 |
| Mix Agent          | `qwen-flash`   | 混合情绪检测             |
| Judge Agent        | `qwen-flash`   | 规则优先 + LLM 兜底      |
| Chat Agent         | `qwen-plus`    | 对话回复生成（质量更高） |
| Profile Agent      | `qwen-flash`   | 用户画像生成             |

**选型理由**：

- `qwen-flash`：响应快、成本低，适合高频调用的情绪分析任务，解决了之前deepseek生成慢的问题
- `qwen-plus`：生成质量更高，用于需要同理心的对话回复

---

### 2.2 各 Agent 详细设计

#### ① Router Agent — 文本分流

**职责**：判断用户输入属于哪种类型，决定后续调用哪些 Agent。

| 分类                | 条件                                                                            | 调用 Sarcasm | 调用 Mix |
| ------------------- | ------------------------------------------------------------------------------- | :----------: | :------: |
| `direct`            | 情绪直接表达，无对比/转折结构                                                   |      ✗       |    ✗     |
| `sarcasm_suspected` | 正面词汇 + 负面语境；夸张赞美 + 糟糕事件；触发词："又"、"还真是"、"真棒"        |      ✓       |    ✗     |
| `mix`               | 转折词（但/但是/不过/然而）；双向情绪；模糊低能量表达（"提不起劲"、"说不上来"） |      ✗       |    ✓     |

#### ② Emotion Agent — 表层情绪检测

**职责**：对文本进行字面情绪判断，**不处理讽刺和复杂混合情绪**（由下游 Agent 修正）。

**9 类情绪标签**：开心、悲伤、愤怒、焦虑、厌烦、中性、疲惫、失落、无奈

**强度量表**：

| 范围   | 含义               |
| ------ | ------------------ |
| 0-30   | 中性               |
| 40-65  | 明显但不强烈的情绪 |
| 66-100 | 强烈情绪           |

**输出字段**：情绪标签、强度(intensity)、置信度(confidence)、情感词、程度词、否定词、转折词

#### ③ Sarcasm Agent — 讽刺检测（条件调用）

**仅当 Router 判定为 `sarcasm_suspected` 时调用。**

**检测标准**：

- 正面词汇 + 负面事件
- 夸大赞美 + 抱怨语境
- 受害者信号（"又来了"、"还真是"）
- 负面场景：加班、改需求、被催、深夜会议

**输出**：是否讽刺(is_sarcasm)、表层情绪、**真实情绪(true_emotion)**、修正后强度

#### ④ Mix Agent — 混合情绪检测（条件调用）

**仅当 Router 判定为 `mix` 时调用。**

**检测标准**：

- 转折结构（"但"、"但是"、"不过"、"然而"）
- 模糊低能量表达（"提不起劲"、"还好但空"）
- 单句双向情绪（"轻松但空"、"开心但累"）

**输出**：是否混合(is_mixed)、主情绪(primary)、次情绪(secondary)、比例(mix_ratio)、调整后强度

#### ③ vs ④ 深度对比：Sarcasm Agent vs Mix Agent

两个 Agent 虽然都是条件调用，但解决的是**完全不同类型**的复杂情绪场景：

| 维度         | Sarcasm Agent（讽刺检测）        | Mix Agent（混合情绪）      |
| ------------ | -------------------------------- | -------------------------- |
| **核心问题** | "说话人真的在表达字面意思吗？"   | "说话人同时有几种情绪？"   |
| **情绪关系** | 表面情绪与真实情绪**相反**       | 多种情绪**同时存在**       |
| **判断逻辑** | 识别"言不由衷"——正话反说         | 识别"悲喜交加"——复杂共存   |
| **输出结构** | 表层情绪 vs 真实情绪（二元对比） | 主情绪 + 次情绪 + 比例分布 |
| **强度处理** | 修正为真实情绪的强度             | 按比例加权计算综合强度     |

**Sarcasm Agent 系统提示词核心逻辑**：

识别反讽的关键信号：
1. 正向词 + 负向事件 → "太好了又要加班"
2. 夸张赞美 + 抱怨语境 → "真是个天才需求"
3. 重复受害信号（"又"） → "又改需求了"
4. 负面场景标记 → 加班、改需求、被催、深夜开会

输出：句面情绪 → 真实情绪（如：表面"开心" → 真实"厌烦"）

**Mix Agent 系统提示词核心逻辑**：

识别混合情绪的关键信号：
1. 转折结构 → "开心但累"、"轻松但空"
2. 模糊低能量表达 → "提不起劲"、"还好但空"
3. 同句双向情绪 → 前半句正向 + 后半句负向

输出：主情绪 + 次情绪 + 比例（如：疲惫 58% + 开心 42%）

**典型场景对比**：

| 场景                 | Agent   | 分析结果              |
| -------------------- | ------- | --------------------- |
| "太好了，又改需求了" | Sarcasm | 表面开心 → 真实厌烦   |
| "项目上线了，但好累" | Mix     | 开心(45%) + 疲惫(55%) |
| "还真是有意思啊"     | Sarcasm | 表面中性 → 真实无奈   |
| "有点开心又有点焦虑" | Mix     | 开心(50%) + 焦虑(50%) |

#### ⑤ Judge Agent — 最终仲裁（核心）

Judge Agent 是整个流水线的核心决策者，采用**规则优先 + LLM 兜底**的混合策略。

##### ⑤.1 决策流程

```
输入：所有上游 Agent 的结果
    │
    ▼
┌─────────────────────┐
│ Step 1: 规则判定     │ ← 始终执行，基于路由类型分发
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Step 2: 是否需要 LLM？│ ← 检查置信度阈值和冲突条件
└────────┬────────────┘
         │
    ┌────┴────┐
    │ 不需要  │──→ 直接返回规则结果
    │ 需要    │──→ 调用 LLM 辅助仲裁
    └────┬────┘
         │
         ▼
    最终结果
```

##### ⑤.2 规则判定逻辑

**直接路由（direct）**：直接使用 Emotion Agent 结果，不做任何修改。

**讽刺路由（sarcasm_suspected）** — 三种情况：

| 情况                | 条件                                             | 情绪来源                | 强度（Intensity）           | 置信度（Confidence）               |
| ------------------- | ------------------------------------------------ | ----------------------- | --------------------------- | ---------------------------------- |
| 讽刺确认 + 高置信度 | `is_sarcasm=True` 且 `sarcasm_confidence ≥ 0.65` | Sarcasm 的 true_emotion | **采用** `revised_intensity` | **加权合并** `emotion×0.3 + sarcasm×0.7` |
| 讽刺确认 + 低置信度 | `is_sarcasm=True` 且 `sarcasm_confidence < 0.65` | 回退到 Emotion Agent    | **保持** `intensity` 不变     | **降低 20%** `emotion×0.8`         |
| 未检测到讽刺        | `is_sarcasm=False`                               | Emotion Agent           | **保持** `intensity` 不变     | **降低 10%** `emotion×0.9`         |

**混合路由（mix）** — 三种情况：

| 情况                | 条件                                       | 情绪来源                   | 强度（Intensity）             | 置信度（Confidence）                 |
| ------------------- | ------------------------------------------ | -------------------------- | ----------------------------- | ------------------------------------ |
| 混合确认 + 高置信度 | `is_mixed=True` 且 `mix_confidence ≥ 0.65` | Mix 的 primary + secondary | **采用** `adjusted_intensity` | **加权合并** `emotion×0.3 + mix×0.7` |
| 混合确认 + 低置信度 | `is_mixed=True` 且 `mix_confidence < 0.65` | 回退到 Emotion Agent       | **保持** `intensity` 不变       | **降低 20%** `emotion×0.8`           |
| 未检测到混合        | `is_mixed=False`                           | Emotion Agent              | **保持** `intensity` 不变       | **降低 10%** `emotion×0.9`           |

##### ⑤.3 置信度与强度的处理规则总结

**置信度（Confidence）处理**：

| 场景                      | 处理方式                          | 公式                          | 说明                     |
| ------------------------- | --------------------------------- | ----------------------------- | ------------------------ |
| 直接路由                  | 不变                              | `emotion_confidence`          | 直接使用 Emotion 置信度  |
| 讽刺/混合确认 + 高置信度 | 加权合并                          | `emotion×0.3 + sarcasm/mix×0.7` | 专项检测器权重更高（70%） |
| 讽刺/混合确认 + 低置信度 | 降低 20%                          | `emotion×0.8`                 | 回退到表层情绪，表示不确定性 |
| 未检测到讽刺/混合        | 降低 10%                          | `emotion×0.9`                 | 排除但不确定，小幅降低   |

**强度（Intensity）处理**：

| 场景                      | 处理方式    | 来源                           | 说明                     |
| ------------------------- | ----------- | ------------------------------ | ------------------------ |
| 直接路由                  | 不变        | `emotion.intensity`            | 直接使用 Emotion 强度    |
| 讽刺确认 + 高置信度       | 采用新值    | `sarcasm.revised_intensity`    | 使用 Sarcasm 修正后的强度 |
| 讽刺确认 + 低置信度       | 不变        | `emotion.intensity`            | 回退到 Emotion 强度      |
| 未检测到讽刺              | 不变        | `emotion.intensity`            | 直接使用 Emotion 强度    |
| 混合确认 + 高置信度       | 采用新值    | `mix.adjusted_intensity`       | 使用 Mix 调整后的强度     |
| 混合确认 + 低置信度       | 不变        | `emotion.intensity`            | 回退到 Emotion 强度      |
| 未检测到混合              | 不变        | `emotion.intensity`            | 直接使用 Emotion 强度    |

**关键区别**：

- **置信度**：会根据场景**主动降低**，反映判断的不确定性（20% 或 10% 惩罚）
- **强度**：只在**高置信度确认**时才采用新的修正值，其他情况**保持不变**

**设计意图**：

- **加权合并（×0.3 + ×0.7）**：讽刺/混合 Agent 作为专项检测器，权重更高（70%），Emotion Agent 提供基础参考（30%）
- **20% 置信度惩罚**：虽然检测到了，但置信度不够高，系统选择"保守回退"到表层情绪，同时降低置信度表示不确定性
- **10% 置信度惩罚**：排除了讽刺/混合可能性，但"排除"本身也有不确定性，小幅降低置信度
- **强度保持不变**：低置信度时，强度仍使用 Emotion Agent 的原始值，确保输出稳定

##### ⑤.4 LLM 兜底触发条件

当规则判定不够可靠时，Judge Agent 会调用 LLM 进行辅助仲裁。触发条件：

**通用触发**（所有路由类型）：

- `emotion_confidence < 0.65`（Emotion Agent 自身就不确定）

**讽刺路由额外触发**（满足任一即触发）：

- Sarcasm Agent 结果缺失
- `sarcasm_confidence < 0.65`
- 两个 Agent 置信度差距 ≤ 0.15（无法判断谁更可靠）
- Sarcasm 检测到讽刺，但 `true_emotion ≠ emotion_label`（Agent 间情绪标签冲突）

**混合路由额外触发**（满足任一即触发）：

- Mix Agent 结果缺失
- `mix_confidence < 0.65`
- 两个 Agent 置信度差距 ≤ 0.15
- Mix 检测到混合，但 `primary_emotion ≠ emotion_label`（Agent 间情绪标签冲突）

##### ⑤.5 输出结果

```
JudgeResult = {
  final_emotion:      最终情绪标签（9 类之一）
  secondary_emotion:  次情绪（仅混合时有值）
  final_intensity:    最终强度 0-100
  final_confidence:   最终置信度 0-1
  is_sarcasm:         是否讽刺
  is_mixed:           是否混合情绪
  reason:             判定理由
}
```

##### ⑤.6 Judge Agent 在 Java 后端的调用位置

在 `EmotionAnalysisService.java` 中，Judge Agent 的调用位于情绪分析流水线的最后一步，接收所有上游 Agent 的结果，输出最终的仲裁结果，供 Chat Agent 生成回复使用。

---

#### ⑥ Chat Agent — 对话回复生成（最后调用）

**职责**：基于情绪分析结果和对话历史，生成温和、尊重、具体的聊天回复。

**输入**：
- 当前用户消息（`text`）
- Judge Agent 的情绪分析结果（`judge_result`）
- 对话历史（最近 20 条，仅包含 `user` 和 `assistant` 角色的消息）

**回复原则**：

- 优先理解最近 3 轮对话历史，处理"能给我一些建议吗""那怎么办"等依赖上下文的表达
- 用户请求建议时，先给出可执行帮助，不先追问；信息不足时先给通用建议，结尾只问 1 个必要问题
- 回复要抓住用户说过的具体事实，避免"听起来你很难受"这类空泛模板
- 建议轻量、具体、可执行（"先列清单"、"把任务拆成 3 块"、"休息 10 分钟再处理最小一项"）
- 普通场景回复 2-5 句话，像可靠的聊天助手而不是分析报告
- 不在回复中展示情绪标签、置信度或 JSON 字段名

**安全机制**：

- 检测到自伤/自杀/伤害他人倾向时，`risk_hint` 设为 `possible_crisis`，语气切换为 `crisis_support`
- 危机场景不提供危险方法，建议联系可信任的人或专业支持

**输出结构**：

```json
{
  "reply": "给用户的回复",
  "tone": "supportive | calm | encouraging | reflective | crisis_support",
  "risk_hint": "none | possible_crisis",
  "suggested_actions": ["行动建议"],
  "reason": "生成依据"
}
```

**历史消息查询逻辑**：

查询最近 20 条历史消息（排除当前用户消息），反转顺序使最早消息在前，只返回角色为 user 和 assistant 的消息。

#### ⑦ Fallback 降级机制

每个 Agent 都实现了 fallback 降级机制，确保在 LLM 调用失败或异常时系统仍能运行：

**降级触发条件**：
- LLM API 调用超时或网络异常
- API Key 缺失或无效
- LLM 返回格式异常（无法解析为预期结构）
- 大模型服务不可用

**各 Agent 降级策略**：

| Agent           | 降级行为                                     | 输出示例                                     |
| --------------- | -------------------------------------------- | -------------------------------------------- |
| Router Agent    | 默认返回 `direct` 类型，跳过后续条件 Agent    | `{ sample_type: "direct" }`                  |
| Emotion Agent   | 返回保守的中性情绪                           | `{ emotion: "中性", intensity: 50, confidence: 0.5 }` |
| Sarcasm Agent   | 默认不检测到讽刺                             | `{ is_sarcasm: false }`                      |
| Mix Agent       | 默认不检测到混合情绪                         | `{ is_mixed: false }`                        |
| Judge Agent     | 直接使用 Emotion Agent 结果（纯规则模式）     | 无 LLM 调用，仅规则判定                      |
| Chat Agent      | 返回通用安抚回复                             | `"我理解你的感受，有什么我可以帮助的吗？"`   |
| Profile Agent   | 返回空画像                                   | `{ profile: "" }`                           |

**降级模式标识**：

当系统进入降级模式时，健康检查接口返回特殊状态：

```
{
  "status": "degraded",
  "ready": false,
  "reason": "API_KEY missing or LLM unavailable"
}
```

**设计意图**：
- **可用性优先**：即使 AI 功能不可用，系统核心功能（登录、历史记录查询）仍正常
- **用户体验**：返回保守但合理的默认值，避免空白或错误页面
- **可观测性**：通过健康检查接口可监控降级状态，便于运维及时处理

---

### 2.3 Benchmark 对比分析

#### 2.3.1 实验设计

对比两种方案在三种文本类型上的表现：

- **GPT-5.5 直接模型**：直接调用 LLM 进行情绪分析
- **GPT-5.5 Agent 流水线**：通过多 Agent 协作进行情绪分析

测试样本：direct（直接表达）、mix（混合情绪）、sarcasm（讽刺）各 5 条，共 15 条。

#### 2.3.2 准确率

| 场景    | 直接模型 | Agent 流水线 |
| ------- | -------- | ------------ |
| 总体    | 100%     | 100%         |
| direct  | 100%     | 100%         |
| mix     | 100%     | 100%         |
| sarcasm | 100%     | 100%         |

两种方案在准确率上持平，均达到 100%。

#### 2.3.3 情绪强度对比

| 场景    | 直接模型 | Agent 流水线 | 差异 |
| ------- | -------- | ------------ | ---- |
| 总体    | 72.00    | 71.40        | -0.6 |
| direct  | 84.00    | 74.40        | -9.6 |
| mix     | 58.40    | 60.20        | +1.8 |
| sarcasm | 73.60    | 79.60        | +6.0 |

**分析**：

- Agent 流水线在讽刺场景下强度评分更高（79.6 vs 73.6），说明多 Agent 协作能更准确识别讽刺背后的真实情绪强度
- 混合情绪场景下 Agent 流水线也略高（60.2 vs 58.4）
- 直接模型在直接表达场景下强度偏高，可能存在过度判断

#### 2.3.4 置信度对比

| 场景    | 直接模型 | Agent 流水线 |
| ------- | -------- | ------------ |
| 总体    | 0.9100   | 0.8741       |
| direct  | 0.9660   | 0.9320       |
| mix     | 0.8600   | 0.8316       |
| sarcasm | 0.9040   | 0.8588       |

**分析**：

- 直接模型置信度普遍更高，因为单一 LLM 对自身判断更"自信"
- Agent 流水线置信度略低，但这是更合理的——多 Agent 仲裁机制会在不确定时降低置信度，反映了更真实的判断不确定性

#### 2.3.5 结论

Agent 流水线的核心优势不在于准确率（两者持平），而在于：

1. **更细致的情绪强度识别**，尤其在讽刺和混合情绪场景
2. **更合理的置信度校准**，不会过度自信
3. **可解释性**：每个 Agent 的中间结果可追溯，便于调试和优化
4. **可扩展性**：新增情绪类型只需添加对应 Agent，无需重新训练

---

## 三、前端设计

### 3.1 技术选型

采用**原生 HTML/CSS/JavaScript**，无任何框架依赖：

- 无构建工具、无打包器、无 npm 依赖
- 直接由 Nginx 作为静态文件服务
- 页面加载速度快，首屏无 JS 解析开销

### 3.2 页面结构

```
frontend/
├── index.html              # 入口文件
├── app.js                  # 全部业务逻辑（~1300 行）
├── styles.css              # 样式入口（@import 聚合）
├── scripts/
│   └── partials.js         # HTML 局部加载器
├── partials/
│   ├── auth.html           # 登录/注册页面
│   └── workspace.html      # 对话工作区
└── styles/
    ├── tokens.css          # CSS 变量与基础重置
    ├── auth.css            # 认证页面样式
    ├── layout.css          # 三栏布局
    ├── sidebar.css         # 侧边栏、会话历史
    ├── chat.css            # 聊天区域、消息气泡
    ├── insights.css        # 实时情绪洞察面板
    ├── history.css         # 历史情绪图表页
    └── responsive.css      # 响应式断点
```

### 3.3 自定义品牌 Logo

系统采用自主设计的品牌 Logo 作为应用标识：

- **视觉风格**：可爱卡通风格的拟人化角色，以柔和粉色为主色调
- **角色形象**：圆润的卡通人物，面带微笑、举起一只手打招呼，搭配粉色腮红
- **设计寓意**：传递温暖、友善、治愈的情感基调，与"情绪感知 AI 对话"的产品定位高度契合
- **应用场景**：登录页面品牌展示、对话界面顶部标识、系统加载动画
- **实现方式**：纯矢量 SVG 格式，支持无损缩放，文件体积小（~2KB），适配各类屏幕分辨率

### 3.4 核心交互设计

**实时情绪洞察面板**：

- 主情绪标签 + 次情绪显示
- 情绪强度条（0-100）
- 置信度百分比
- 讽刺/混合情绪标记
- 情绪强度趋势图（SVG 柱状图 + 折线图，纯 JS 绘制）

**表情强度交互组件（FaceRating）**：

- 拖动滑块实时驱动 SVG 表情变化
- 通过 CSS 自定义属性控制：眼睛倾斜、嘴巴弧度、颜色渐变
- 5 档表情：低落 → 有点低落 → 平静 → 有点开心 → 开心

**管理员面板**：

- 用户搜索 + 选择
- 用户画像展示（性格特征、沟通风格、MBTI）
- 情绪趋势可视化

### 3.5 状态管理

采用单全局 `state` 对象管理所有前端状态：

```javascript
const state = {
  auth: { token, userId, username, role },
  conversations: [],        // 会话列表
  messages: [],             // 当前对话消息
  conversationId: null,     // 当前会话 ID
  adminUsers: [],           // 管理员可见用户列表
  adminTargetUserId: "",    // 管理员选中的用户
  // ...
};
```

认证状态持久化到 `localStorage`，会话状态按用户隔离存储。

---

## 四、后端设计

### 4.1 数据库表设计

系统使用 SQLite 数据库，通过 JPA/Hibernate 自动建表，共 5 张表：

```
┌──────────────┐       ┌──────────────────┐
│    users      │       │  conversations   │
├──────────────┤       ├──────────────────┤
│ id (PK/UUID) │◄──┐   │ id (PK/UUID)     │
│ username (UQ) │   │   │ user_id (FK)     │──→ users.id
│ password_hash │   │   │ title            │
│ salt          │   │   │ created_at       │
│ role          │   │   │ updated_at       │
│ created_at    │   │   └────────┬─────────┘
└──────────────┘   │            │
                   │            │
         ┌─────────┤            │
         │         │            ▼
         │   ┌─────┴──────────────────┐
         │   │    chat_messages        │
         │   ├────────────────────────┤
         │   │ id (PK/UUID)           │
         │   │ conversation_id (FK)   │──→ conversations.id
         │   │ role (user/assistant)  │
         │   │ content (LOB)          │
         │   │ created_at             │
         │   └────────┬───────────────┘
         │            │
         ▼            ▼
┌─────────────────────────┐    ┌──────────────────┐
│   emotion_records        │    │  user_profiles    │
├─────────────────────────┤    ├──────────────────┤
│ id (PK/UUID)            │    │ id (PK/UUID)     │
│ conversation_id (FK)    │    │ user_id (UQ/FK)  │──→ users.id
│ message_id (FK)         │    │ profile_data(LOB) │
│ final_emotion           │    │ record_count      │
│ secondary_emotion       │    │ created_at        │
│ final_intensity         │    │ updated_at        │
│ final_confidence        │    └──────────────────┘
│ is_sarcasm              │
│ is_mixed                │
│ raw_analysis_json (LOB) │
│ created_at              │
└─────────────────────────┘
```

**表关系说明**：

- `users` → `conversations`：一对多（一个用户有多个对话）
- `conversations` → `chat_messages`：一对多（一个对话包含多条消息）
- `chat_messages` → `emotion_records`：一对一（每条用户消息对应一条情绪分析结果）
- `users` → `user_profiles`：一对一（缓存的用户画像）

**各表作用**：

| 表名              | 作用                           | 关键字段说明                                                                                              |
| ----------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `users`           | 存储用户账号信息               | `username` 唯一，`password_hash` + `salt` 存储加密密码，`role` 区分 admin/user                            |
| `conversations`   | 存储对话会话                   | 每次新建对话生成一条记录，`title` 取用户首条消息前 15 字                                                  |
| `chat_messages`   | 存储对话中的每条消息           | `role` 区分 user/assistant，`content` 存储消息文本（LOB 大文本）                                          |
| `emotion_records` | 存储每条用户消息的情绪分析结果 | `raw_analysis_json` 存储完整的 Agent 分析 JSON，`final_emotion/intensity/confidence` 存储仲裁后的最终结果 |
| `user_profiles`   | 缓存 AI 生成的用户画像         | `profile_data` 存储完整画像 JSON，`record_count` 记录生成时的情绪记录数（用于增量更新判断）               |

---

### 4.2 接口设计

#### 4.2.1 前端接口（Spring Boot 对外暴露）

**认证接口**（`/api/auth`，无需 JWT）：

| 方法   | 路径                 | 参数                                                | 说明                              |
| ------ | -------------------- | --------------------------------------------------- | --------------------------------- |
| `GET`  | `/api/auth/captcha`  | 无                                                  | 获取验证码 SVG 图片 + captcha_key |
| `POST` | `/api/auth/register` | `{ username, password, captcha_code, captcha_key }` | 注册新用户（需要验证码）          |
| `POST` | `/api/auth/login`    | `{ username, password, auto_login }`                | 登录，返回 JWT Token              |

**业务接口**（`/api/emotion`，需要 JWT）：

| 方法   | 路径                                       | 参数                                                                  | 说明                                                   |
| ------ | ------------------------------------------ | --------------------------------------------------------------------- | ------------------------------------------------------ |
| `GET`  | `/api/emotion/health`                      | 无                                                                    | 健康检查，透传 Python Agent 状态                       |
| `POST` | `/api/emotion/analyze`                     | `{ id, user_id, text, source, created_at, metadata }`                 | 完整情绪分析流水线（Router→Emotion→Sarcasm/Mix→Judge） |
| `POST` | `/api/emotion/chat`                        | `{ text, user_id, conversation_id, judge_result, history, metadata }` | 对话模式：情绪分析 + 生成回复                          |
| `GET`  | `/api/emotion/conversations`               | `?target_user_id=`（可选）                                            | 获取对话列表（admin 可看全部，user 只看自己）          |
| `GET`  | `/api/emotion/conversations/{id}/messages` | `?target_user_id=`（可选）                                            | 获取对话内所有消息                                     |
| `POST` | `/api/emotion/profile`                     | `?target_user_id=`（可选）                                            | 获取用户情绪统计数据（纯统计，无 LLM）                 |
| `POST` | `/api/emotion/profile/generate`            | `?target_user_id=&force=false`                                        | 生成完整用户画像（含 LLM 生成性格/MBTI 等）            |

#### 4.2.2 Spring Boot → Python Agent 通信

Spring Boot 通过 `RestClient`（HTTP）调用 Python FastAPI 服务，基础地址配置为 `http://127.0.0.1:8000`。

**AgentClient 方法映射**：

| Java 方法                  | HTTP   | Python 端点         | 说明                           |
| -------------------------- | ------ | ------------------- | ------------------------------ |
| `health()`                 | `GET`  | `/health`           | 健康检查                       |
| `router(request)`          | `POST` | `/router`           | 文本分流（direct/sarcasm/mix） |
| `emotion(request)`         | `POST` | `/emotion`          | 表层情绪检测                   |
| `sarcasm(request)`         | `POST` | `/sarcasm`          | 讽刺检测                       |
| `mix(request)`             | `POST` | `/mix`              | 混合情绪检测                   |
| `judge(request)`           | `POST` | `/judge`            | 最终仲裁（规则 + LLM 兜底）    |
| `chat(request)`            | `POST` | `/chat`             | 生成对话回复                   |
| `profile(payload)`         | `POST` | `/profile`          | 统计画像（无 LLM）             |
| `profileGenerate(payload)` | `POST` | `/profile/generate` | AI 生成完整画像                |

**数据流向示例（chat 接口）**：

```
前端 POST /api/emotion/chat
    │
    ▼
Spring Boot EmotionAnalysisService.chat()
    │
    ├─① 保存用户消息到 SQLite
    │
    ├─② 调用 Python Agent 流水线
    │   POST /router  → RouterAgent   → 分流结果
    │   POST /emotion → EmotionAgent  → 情绪标签 + 强度 + 置信度
    │   POST /sarcasm → SarcasmAgent  → 讽刺检测结果（条件调用）
    │   POST /mix     → MixAgent      → 混合情绪结果（条件调用）
    │   POST /judge   → JudgeAgent    → 最终仲裁结果
    │
    ├─③ 保存情绪记录到 SQLite
    │
    ├─④ 合并历史消息（DB + 前端，去重，最多 20 条）
    │
    ├─⑤ 调用 Python 生成回复
    │   POST /chat → ChatAgent → 对话回复
    │
    ├─⑥ 保存 AI 回复到 SQLite
    │
    ▼
返回 ChatResponse 给前端
```

#### 4.2.3 Python FastAPI Agent 端点

Python 服务运行在端口 8000，提供 9 个端点：

| 端点                     | 说明     | 输入            | 输出关键字段                                                                 |
| ------------------------ | -------- | --------------- | ---------------------------------------------------------------------------- |
| `GET /health`            | 健康检查 | 无              | `{ status, ready }`                                                          |
| `POST /router`           | 文本分流 | `BaseTextInput` | `{ sample_type, need_sarcasm_check, need_mix_check }`                        |
| `POST /emotion`          | 情绪检测 | `BaseTextInput` | `{ emotion, intensity, confidence, tokens, emotion_words }`                  |
| `POST /sarcasm`          | 讽刺检测 | `BaseTextInput` | `{ is_sarcasm, true_emotion, revised_intensity, confidence }`                |
| `POST /mix`              | 混合情绪 | `BaseTextInput` | `{ is_mixed, primary_emotion, secondary_emotion, mix_ratio }`                |
| `POST /judge`            | 最终仲裁 | `JudgeInput`    | `{ final_emotion, final_intensity, final_confidence, is_sarcasm, is_mixed }` |
| `POST /chat`             | 对话生成 | `ChatInput`     | `{ reply, content }`                                                         |
| `POST /profile`          | 统计画像 | `ProfileInput`  | `{ emotion_distribution, avg_intensity, timeline, radar_chart }`             |
| `POST /profile/generate` | AI 画像  | `ProfileInput`  | `{ personality_traits, mbti, communication_style, summary }`                 |

**每个 Agent 都有 fallback 降级机制**：当 LLM 调用失败或 API Key 缺失时，返回保守的默认结果，确保系统不崩溃。

### 4.3 登录注册与密码加密

#### 密码存储方案

采用 **SHA-256 + 随机盐值** 的哈希方案：

```
注册流程：
1. 生成随机盐值（UUID 去掉连字符）
2. 计算 hash = SHA-256(salt + password)
3. 将 hash（十六进制）和 salt 存入 users 表

登录流程：
1. 从 users 表取出 salt 和 stored_hash
2. 计算 hash = SHA-256(salt + 输入密码)
3. 比较 hash 与 stored_hash 是否一致
```

#### 验证码生成

验证码在服务端动态生成 SVG 图片，无需外部图片资源：

```
生成流程：
1. 从字符池中随机选取 4 个字符
   字符池：ABCDEFGHJKLMNPQRSTUVWXYZ23456789
   （排除 I/O/0/1 等易混淆字符）
2. 为每个字符生成随机位置、旋转角度、颜色
3. 添加干扰线和噪点
4. 组装为 SVG XML
5. Base64 编码后返回给前端
6. 验证码文本存入 ConcurrentHashMap（TTL = 5 分钟）
```

验证码仅在注册时要求，登录不需要。

#### 管理员账号机制

**管理员不允许通过注册创建**，采用配置文件预设 + 自动初始化方案：

```
application.yml 配置：
admin:
  username: admin
  password: admin123

启动时自动初始化（@PostConstruct）：
1. 检查 users 表是否存在 admin 用户
2. 不存在 → 自动创建（SHA-256 加盐哈希密码）
3. 已存在 → 跳过
```

**注册逻辑限制**：注册时硬编码角色为 "user"，无论前端传什么角色都只能创建普通用户。

**设计意图**：
- 管理员账号由运维在配置文件中预设，避免注册接口被滥用创建管理员
- 启动时自动创建，无需手动初始化数据库
- 配置文件中的密码在首次创建后修改不会影响已存在的管理员账号

#### JWT 认证

使用 `jjwt` 库实现无状态认证：

```
Token 结构：
{
  "sub": "用户ID",
  "username": "用户名",
  "role": "user/admin",
  "exp": 过期时间
}

双有效期策略：
- 未勾选"自动登录"：5 分钟过期
- 勾选"自动登录"：7 天过期
```

**拦截器配置**：

| 路径                  | 是否需要 JWT           |
| --------------------- | ---------------------- |
| `/api/auth/**`        | 不需要（登录注册公开） |
| `/api/emotion/health` | 不需要                 |
| `/api/emotion/**`     | 需要                   |

JWT 验证通过后，将用户信息存入 request attribute，Controller 直接获取使用。

### 4.3 角色权限管理

系统实现 **admin / user** 双角色权限控制：

#### 权限矩阵

| 接口                                       | 未登录 | 普通用户  | 管理员 |
| ------------------------------------------ | ------ | --------- | ------ |
| `/api/auth/register`                       | 可用   | 可用      | 可用   |
| `/api/auth/login`                          | 可用   | 可用      | 可用   |
| `/api/auth/captcha`                        | 可用   | 可用      | 可用   |
| `/api/emotion/chat`                        | 401    | 仅自己    | 仅自己 |
| `/api/emotion/conversations`               | 401    | 仅自己    | 所有人 |
| `/api/emotion/conversations/{id}/messages` | 401    | 仅自己    | 所有人 |
| `/api/emotion/profile`                     | 401    | 仅自己    | 所有人 |
| `/api/emotion/profile/generate`            | 401    | 禁止(403) | 所有人 |
| `/api/emotion/health`                      | 可用   | 可用      | 可用   |

#### 实现方式

- **JWT 层**：从 Token 中提取角色信息
- **Controller 层**：根据角色判断权限，普通用户只能看自己的数据，管理员可以指定查看某个用户
- **前端层**：检查角色，管理员看到完全不同的界面布局

---

### 4.4 用户画像系统

#### 4.4.1 画像生成流程

```
情绪记录（emotion_records 表）
    │
    ▼
特征提取（纯 Python 统计，无 LLM 调用）
    ├── 情绪分布统计
    ├── 平均强度/置信度
    ├── 讽刺/混合比例
    ├── 主导情绪识别
    ├── 强度趋势（线性回归）
    └── 活跃时段分析（凌晨/上午/下午/晚上）
    │
    ▼
LLM 画像生成
    ├── 性格特征（personality_traits）
    ├── 沟通风格（communication_style）
    ├── 情绪模式（emotional_patterns）
    ├── MBTI 推断
    └── 自然语言摘要（summary）
```

#### 4.4.2 画像缓存策略

```
用户请求画像
    │
    ├── 无缓存 → 直接调用 LLM 生成
    │
    └── 有缓存 → 计算新增记录数
                      │
                      ├── 新增 >= 10 条 → 重新生成，覆盖缓存
                      │
                      └── 新增 < 10 条  → 返回缓存
```

缓存存储在 `user_profiles` 表中，`record_count` 记录上次生成时的记录总数。

---

## 五、CI/CD 部署

### 5.1 CI 流程（持续集成）

触发条件：推送到 `main`/`master` 分支 或 Pull Request

```
┌─────────────────────────────────┐
│           CI Pipeline            │
├────────────────┬────────────────┤
│  Backend Build  │ Python Agent   │
│                 │ Check          │
│  Java 17        │ Python 3.10    │
│  Maven test     │ compileall     │
└────────────────┴────────────────┘
```

### 5.2 Deploy 流程（持续部署）

触发条件：CI 成功后自动触发 或 手动触发

```
┌──────────────────────────────────────────┐
│            Deploy Pipeline                │
├──────────────────────────────────────────┤
│ 1. Checkout 代码（Runner 上）             │
│ 2. rsync 上传代码到服务器（SSH）           │
│    排除：.git, .env, *.sqlite3,          │
│          node_modules, target, __pycache__ │
│ 3. SSH 重启服务                           │
│    ├── systemctl restart emotion-agent-python │
│    ├── systemctl restart nginx            │
│    └── systemctl restart emotion-agent    │
│ 4. 等待 Spring Boot 启动（最多 30 秒）     │
│ 5. 验证三个服务状态                        │
└──────────────────────────────────────────┘
```

### 5.3 服务器架构

```
华为云 ECS
├── /emotion_agent/
│   ├── frontend/          # 静态前端文件
│   ├── backend/           # Spring Boot 应用
│   │   └── data/
│   │       └── emoagent.sqlite3   # SQLite 数据库
│   ├── agents/            # Python FastAPI Agent
│   └── .env               # 环境变量（API Key 等）
│
├── systemd 服务
│   ├── emotion-agent-python.service  # FastAPI（端口 8000）
│   ├── emotion-agent.service         # Spring Boot（端口 8080）
│   └── nginx.service                 # Nginx（端口 80）
│
└── Nginx 配置
    ├── 监听 80 端口
    ├── 静态文件服务 → frontend/
    └── /api/* 反向代理 → 127.0.0.1:8080
```

### 5.4 Nginx 反向代理与端口转发

Nginx 是系统的入口网关，承担**静态文件服务**和**API 反向代理**两个职责：

```
用户浏览器
    │
    │  http://服务器IP:80
    ▼
┌─────────────────────────────────────────────┐
│  Nginx（监听 80 端口）                        │
│                                              │
│  location / {                                │
│      try_files $uri $uri/ /index.html;       │
│      → 静态文件：/emotion_agent/frontend/    │
│      → SPA 回退：所有不存在的路径返回 index.html │
│  }                                           │
│                                              │
│  location /api/ {                            │
│      proxy_pass http://127.0.0.1:8080;       │
│      → 转发到 Spring Boot 后端                │
│      → 注意：proxy_pass 尾部不能加 /           │
│         否则会丢失 /api 前缀                   │
│  }                                           │
└─────────────────────────────────────────────┘
```

**关键配置说明**：

```nginx
server {
    listen 80;
    server_name _;

    # 前端静态文件
    root /emotion_agent/frontend;
    index index.html;

    # SPA 路由回退 — 所有前端路由都返回 index.html
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API 反向代理 — 将 /api/* 请求转发到 Spring Boot
    location /api/ {
        proxy_pass http://127.0.0.1:8080;
        #  ↑ 尾部不加 / ，保留完整的 /api/auth/login 路径
        #  如果写成 proxy_pass http://127.0.0.1:8080/;（加 /）
        #  则 /api/auth/login 会被转发为 /auth/login（丢失 /api 前缀）
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

**请求流转示例**：

| 用户请求                 | Nginx 处理                                  | 最终到达                   |
| ------------------------ | ------------------------------------------- | -------------------------- |
| `GET /`                  | 静态文件 →`frontend/index.html`             | 浏览器渲染页面             |
| `GET /app.js`            | 静态文件 →`frontend/app.js`                 | 浏览器加载 JS              |
| `POST /api/auth/login`   | 反向代理 →`127.0.0.1:8080/api/auth/login`   | Spring Boot                |
| `POST /api/emotion/chat` | 反向代理 →`127.0.0.1:8080/api/emotion/chat` | Spring Boot → Python Agent |

**服务间通信**：

```
浏览器 ──HTTP:80──→ Nginx ──HTTP:8080──→ Spring Boot ──HTTP:8000──→ Python Agent ──HTTPS──→ DeepSeek/Qwen LLM
                  静态文件                 业务逻辑                    Agent 流水线              大模型 API
                  API 代理                JWT 鉴权                    情绪分析
                                         数据库读写                   对话生成
```

### 5.5 密钥管理

GitHub Secrets 中存储：

- `SSH_HOST`：服务器 IP
- `SSH_PORT`：SSH 端口
- `SSH_USER`：登录用户名
- `SSH_PRIVATE_KEY`：SSH 私钥

rsync 通过 SSH 隧道传输代码，服务器无需访问外网（解决国内服务器访问 GitHub 不稳定问题）。

---

## 六、测试

### 6.1 测试策略

| 测试类型         | 工具                     | 覆盖范围                                         |
| ---------------- | ------------------------ | ------------------------------------------------ |
| 后端单元测试     | JUnit + Spring Boot Test | Service 层逻辑、鉴权、缓存策略                   |
| Agent 单元测试   | unittest / pytest        | Router、Emotion、Sarcasm、Mix、Judge、Chat Agent |
| 接口联调测试     | curl / Postman           | 认证、情绪分析、聊天、历史、画像全流程           |
| 前端静态资源测试 | 浏览器 + HTTP 请求       | 页面加载、CSS/JS 资源、partial 片段              |
| 性能测试         | 并发请求工具             | 登录、分析、聊天、历史查询、画像统计             |
| 压力测试         | 多线程并发               | 10/20/50 并发用户场景                            |
| CI/CD 测试       | GitHub Actions           | 自动构建、部署、服务重启验证                     |

### 6.2 功能测试用例

#### 用户认证

| 测试项                  | 输入                   | 期望结果                 | 实际结果 |
| ----------------------- | ---------------------- | ------------------------ | -------- |
| 注册新用户              | 合法用户名+密码+验证码 | 返回成功，数据库新增记录 | 通过     |
| 重复注册                | 已存在用户名           | 返回 400 错误            | 通过     |
| 正确登录                | 正确用户名+密码        | 返回 JWT Token           | 通过     |
| 错误密码登录            | 错误密码               | 返回 400                 | 通过     |
| 获取验证码              | GET /api/auth/captcha  | 返回 SVG Base64 图片     | 通过     |
| 无 Token 访问受保护接口 | 不携带 Token           | 返回 401                 | 通过     |
| 伪造 Token              | 无效 Token             | 返回 401                 | 通过     |

#### 情绪分析

| 测试项   | 输入文本                                  | 期望结果                            | 实际结果 |
| -------- | ----------------------------------------- | ----------------------------------- | -------- |
| 直接情绪 | "我今天特别开心"                          | 返回开心情绪，含强度和置信度        | 通过     |
| 负面情绪 | "我现在特别焦虑，什么都做不好"            | 返回焦虑情绪                        | 通过     |
| 讽刺文本 | "太好了，周末又能继续改需求了"            | Router 标记讽刺，Judge 输出修正情绪 | 通过     |
| 混合情绪 | "拿到 offer 很开心，但也很担心自己做不好" | 包含主情绪和次情绪                  | 通过     |
| 中性文本 | "今天下午三点开会"                        | 返回中性或低强度                    | 通过     |
| 空文本   | ""                                        | 返回 400                            | 通过     |
| 超长文本 | 超长输入                                  | 正常降级处理，不崩溃                | 通过     |

#### 聊天对话

| 测试项       | 输入                | 期望结果                                 | 实际结果 |
| ------------ | ------------------- | ---------------------------------------- | -------- |
| 新建会话     | "我今天有点累"      | 返回 conversationId + 分析结果 + AI 回复 | 通过     |
| 连续对话     | 同一会话多条消息    | 沿用同一会话，历史上下文合并             | 通过     |
| 刷新后查询   | 查看会话列表        | 最近会话出现在列表中                     | 通过     |
| 打开历史会话 | 点击历史会话        | 按时间正序显示消息                       | 通过     |
| 越权访问     | 他人 conversationId | 返回 404 拒绝访问                        | 通过     |

#### 历史记录与权限

| 测试项           | 角色                    | 期望结果           | 实际结果 |
| ---------------- | ----------------------- | ------------------ | -------- |
| 查询会话列表     | 普通用户                | 仅返回自己的会话   | 通过     |
| 查看他人会话消息 | 普通用户                | 返回 404           | 通过     |
| 查询全部会话     | 管理员                  | 返回所有用户会话   | 通过     |
| 按用户筛选会话   | 管理员 + target_user_id | 仅返回指定用户会话 | 通过     |
| 退出后访问历史   | 未登录                  | 返回 401           | 通过     |

#### 用户画像

| 测试项       | 条件           | 期望结果                          | 实际结果 |
| ------------ | -------------- | --------------------------------- | -------- |
| 统计画像     | 普通用户       | 返回情绪分布、平均强度等          | 通过     |
| 首次生成画像 | 无缓存         | 调用 LLM 生成，写入 user_profiles | 通过     |
| 缓存命中     | 新增 < 10 条   | 返回 cached=true                  | 通过     |
| 强制刷新     | force=true     | 忽略缓存重新生成                  | 通过     |
| 无记录时生成 | record_count=0 | 返回空画像                        | 通过     |

### 6.3 Agent 单元测试

| Agent         | 测试内容                              | 测试文件              |
| ------------- | ------------------------------------- | --------------------- |
| Router Agent  | direct/sarcasm/mix 分类、非法类型处理 | test_router_agent.py  |
| Emotion Agent | 情绪标签、强度、置信度输出            | test_emotion_agent.py |
| Sarcasm Agent | 讽刺检测、真实情绪修正                | test_sarcasm_agent.py |
| Mix Agent     | 混合情绪、主次情绪、比例              | test_mix_agent.py     |
| Judge Agent   | 高置信直接结果、低置信回退、LLM 兜底  | test_judge_agent.py   |
| Chat Agent    | 带历史的聊天回复生成                  | test_chat_agent.py    |

### 6.4 性能测试结果

| 接口         | 期望均值  | 实际均值    | 结论 |
| ------------ | --------- | ----------- | ---- |
| 用户登录     | ≤ 100ms   | 10-12ms     | 通过 |
| 情绪分析     | ≤ 10000ms | 2078-5953ms | 通过 |
| 聊天回复     | ≤ 10000ms | 4039-7753ms | 通过 |
| 历史会话查询 | ≤ 100ms   | 8.46ms      | 通过 |
| 用户画像统计 | ≤ 100ms   | 13.52ms     | 通过 |

### 6.5 压力测试结果

| 场景         | 并发数 | 请求数 | 失败率          | 结论             |
| ------------ | ------ | ------ | --------------- | ---------------- |
| 并发登录     | 50     | 500    | 0%              | 通过             |
| 并发情绪分析 | 10     | 30     | 0%              | 通过             |
| 并发情绪分析 | 20     | 60     | 1.67%           | 通过             |
| 并发情绪分析 | 50     | 150    | 36%（API 限流） | 受限于大模型 API |
| 并发历史查询 | 50     | 500    | 0%              | 通过             |
| 并发画像查看 | 50     | 500    | 0%              | 通过             |

### 6.6 健壮性测试

| 异常场景             | 系统行为                               | 结论 |
| -------------------- | -------------------------------------- | ---- |
| 必填字段缺失         | 返回 400，提示补全信息                 | 通过 |
| 字段类型错误         | 返回参数错误，不调用 Agent             | 通过 |
| Token 缺失/伪造/过期 | 返回 401，拒绝访问                     | 通过 |
| 普通用户访问他人会话 | 返回 404/403                           | 通过 |
| Agent 服务未启动     | 健康检查异常，认证仍可用               | 通过 |
| 大模型 API Key 缺失  | Agent 进入 degraded 模式，返回保守结果 | 通过 |
| Agent 返回格式异常   | 记录异常，返回降级结果                 | 通过 |
| SQLite 文件不可写    | 事务失败，返回错误                     | 通过 |

### 6.7 测试准入与准出标准

**准入标准**：

- 前端、后端、Agent 主流程代码已完成
- 前端 5173、后端 8080、Agent 8000 可启动
- 数据库、JWT、Agent base-url 配置已准备
- 已准备普通用户、管理员和典型情绪文本样例

**准出标准**：

- 登录注册、情绪分析、聊天、历史、统计画像主流程均通过
- 后端测试和 Agent 单元测试全部通过
- 无阻塞级和严重级未关闭缺陷
- 普通用户不能访问他人会话，Token 校验正常
- Agent 异常、参数错误、服务不可用等场景有降级处理

> 详细测试计划参见 `docs/测试计划文档.docx`

---

## 七、项目亮点总结

1. **多 Agent 协作架构**：Router → Emotion → Sarcasm/Mix → Judge → Chat，流水线式处理，每个 Agent 职责单一
2. **规则优先 + LLM 兜底**：Judge Agent 对简单场景使用规则判定，减少 LLM 调用，降低成本和延迟
3. **优雅降级**：每个 Agent 都有 fallback 机制，LLM 不可用时系统仍能运行
4. **无框架前端**：纯原生实现，无构建依赖，加载速度快
5. **完整 CI/CD**：GitHub Actions 自动测试 + rsync 部署，一键发布
6. **双角色权限**：admin 可查看所有用户数据，user 只能操作自己的数据
7. **画像缓存**：增量更新策略，避免重复调用 LLM

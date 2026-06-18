# EmoAgent 多智能体流水线详解

## 系统概览

EmoAgent 是一个基于多智能体协作的情感分析系统，采用 Python + FastAPI 构建。每个 Agent 作为独立的微服务端点运行，由外部调用方（Java 后端）编排执行顺序。

**LLM 后端**：阿里云 DashScope（Qwen 系列模型）
- 分析类 Agent：`qwen-flash`（低温度 0.1，确定性输出）
- 画像生成：`qwen-plus`（温度 0.3，轻微创造性）
- 对话生成：`qwen-plus`（温度 0.4，自然回复）

---

## 9 种情绪标签

所有 Agent 统一使用以下情绪标签集合：

```python
EMOTION_LABELS = {"开心", "悲伤", "愤怒", "焦虑", "厌烦", "中性", "疲惫", "失落", "无奈"}
```

---

## 共享输入结构（BaseTextInput）

Router、Emotion、Sarcasm、Mix 四个 Agent 共享相同的输入结构：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | `string` | 是 | 消息唯一 ID |
| `user_id` | `string` | 是 | 用户 ID |
| `text` | `string` | 是 | 待分析的原始文本 |
| `source` | `string` | 是 | 消息来源标识 |
| `created_at` | `string` | 是 | 创建时间（ISO 格式） |
| `metadata` | `object` | 否 | 附加元数据，默认 `{}` |

---

## 1. Router Agent（路由分类器）

**目录**：`agents/router_agent/`
**作用**：判断文本类型，决定下游需要调用哪些 Agent。

### 输入

同 BaseTextInput。

### 输出（RouterResult）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `sample_type` | `string` | 是 | 路由类型：`"direct"` / `"sarcasm_suspected"` / `"mix"` |
| `need_sarcasm_check` | `bool` | 是 | 是否需要讽刺检测 |
| `need_mix_check` | `bool` | 是 | 是否需要混合情绪检测 |
| `routing_reason` | `string` | 是 | 分类理由 |
| `evidence` | `string[]` | 否 | 支持分类的证据文本，默认 `[]` |

### 路由逻辑

| 类型 | 触发条件 | need_sarcasm_check | need_mix_check |
|------|----------|-------------------|----------------|
| `direct` | 直接情感表达，无转折词、无讽刺模式 | false | false |
| `sarcasm_suspected` | 表面正面词汇 + 负面上下文（如"又""还真是""真棒"） | true | false |
| `mix` | 含转折词（"但""但是""不过""然而"）、矛盾情绪、模糊低能量表达 | false | true |

### Fallback

当 LLM 不可用时，返回 `sample_type: "direct"`，`fallback: true`。

---

## 2. Emotion Agent（表层情绪判断）

**目录**：`agents/emotion_agent/`
**作用**：对原始文本进行"表层情绪判断"，只输出文本字面表达的情绪，不处理讽刺或混合情绪。

### 输入

同 BaseTextInput。

### 输出（EmotionResult）

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `tokens` | `string[]` | 否 | `[]` | 分词结果 |
| `emotion_words` | `string[]` | 否 | `[]` | 情绪词 |
| `degree_words` | `string[]` | 否 | `[]` | 程度词（如"非常""有点"） |
| `negation_words` | `string[]` | 否 | `[]` | 否定词（如"不""没"） |
| `contrast_words` | `string[]` | 否 | `[]` | 转折词（如"但是""却"） |
| `emotion` | `string` | 否 | `"中性"` | 情绪标签，必须在 9 种标签内 |
| `intensity` | `int` | 否 | `0` | 情绪强度，0-100 |
| `confidence` | `float` | 否 | `0.0` | 置信度，0.0-1.0 |
| `reason` | `string` | 否 | `""` | 判断理由 |

### 强度指南

| 区间 | 含义 |
|------|------|
| 0-30 | 中性/微弱 |
| 40-65 | 中等强度 |
| 66-100 | 强烈 |

### Fallback

返回 `emotion: "中性"`, `intensity: 30`, `confidence: 0.2`。

---

## 3. Sarcasm Agent（讽刺检测器）

**目录**：`agents/sarcasm_agent/`
**作用**：检测讽刺/反语表达，纠正情绪标签。仅在 Router 返回 `need_sarcasm_check: true` 时调用。

### 输入

同 BaseTextInput。

### 输出（SarcasmResult）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `is_sarcasm` | `bool` | 是 | 是否为讽刺表达 |
| `surface_emotion` | `string` | 是 | 表面情绪（文本字面表达），必须在 9 种标签内 |
| `true_emotion` | `string` | 是 | 真实情绪（讽刺背后的真实情感），必须在 9 种标签内 |
| `revised_intensity` | `int` | 是 | 修正后的情绪强度，0-100 |
| `confidence` | `float` | 是 | 置信度，0.0-1.0 |
| `reason` | `string` | 是 | 判断理由 |

### 检测模式

- 正面词汇 + 负面事件（如"太好了，又要加班"）
- 夸张表扬 + 抱怨
- 重复标记（"又""还真是"）
- 负面场景（加班、改需求、深夜会议）

### Fallback

返回 `is_sarcasm: false`，情绪为中性。

---

## 4. Mix Agent（混合情绪检测器）

**目录**：`agents/mix_agent/`
**作用**：处理包含混合/复合情绪的复杂文本，无法用单一标签表达时调用。仅在 Router 返回 `need_mix_check: true` 时调用。

### 输入

同 BaseTextInput。

### 输出（MixResult）

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `is_mixed` | `bool` | 是 | -- | 是否为混合情绪 |
| `primary_emotion` | `string` | 是 | -- | 主情绪，必须在 9 种标签内 |
| `secondary_emotion` | `string` | 是 | -- | 次情绪，必须在 9 种标签内 |
| `mix_ratio` | `object` | 否 | `{}` | 情绪比例分配，键为情绪标签，值为 0-1 的浮点数 |
| `adjusted_intensity` | `int` | 否 | `0` | 调整后的情绪强度，0-100 |
| `confidence` | `float` | 否 | `0.0` | 置信度，0.0-1.0 |
| `reason` | `string` | 否 | `""` | 判断理由 |

### mix_ratio 验证规则

```json
{
  "开心": 0.3,
  "悲伤": 0.7
}
```

- 所有键必须是有效情绪标签
- 所有值必须在 [0, 1] 范围内
- 所有值之和必须在 [0.95, 1.05] 范围内（允许浮点误差）
- 必须包含 primary_emotion 和 secondary_emotion 对应的键

### 检测模式

- 转折结构："但""但是""不过""然而"
- 模糊低能量表达："提不起劲""说不上来"
- 单句内双向情绪

### Fallback

返回 `is_mixed: false`，情绪为中性。

---

## 5. Judge Agent（最终裁决者）

**目录**：`agents/judge_agent/`
**作用**：整合 Router、Emotion、Sarcasm、Mix 四个 Agent 的结果，输出最终情绪判定。采用"规则优先 + LLM 兜底"的混合策略。

### 输入（JudgeInput）

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `router_result` | `object` | 是 | -- | Router Agent 的输出 |
| `emotion_result` | `object` | 是 | -- | Emotion Agent 的输出 |
| `sarcasm_result` | `object \| null` | 否 | `null` | Sarcasm Agent 的输出（仅讽刺路径有） |
| `mix_result` | `object \| null` | 否 | `null` | Mix Agent 的输出（仅混合路径有） |
| `text` | `string \| null` | 否 | `null` | 原始文本（可选，供 LLM 兜底时参考） |

### 输出（JudgeResult）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `final_emotion` | `string` | 是 | 最终情绪标签 |
| `secondary_emotion` | `string \| null` | 是 | 次情绪（仅混合情绪时有值） |
| `final_intensity` | `int` | 是 | 最终情绪强度，0-100 |
| `final_confidence` | `float` | 是 | 最终置信度，0.0-1.0 |
| `is_sarcasm` | `bool` | 是 | 是否检测到讽刺 |
| `is_mixed` | `bool` | 是 | 是否为混合情绪 |
| `reason` | `string` | 是 | 裁决理由 |

### 阈值配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sarcasm_confidence_threshold` | 0.65 | 讽刺置信度阈值 |
| `mix_confidence_threshold` | 0.65 | 混合情绪置信度阈值 |
| `emotion_confidence_threshold` | 0.65 | 情绪置信度阈值 |
| `review_confidence_margin` | 0.15 | 置信度差异阈值 |

### 规则裁决逻辑

规则只处理**确定的情况**。不确定的情况（如检测到讽刺但置信度低）交给 LLM 仲裁。

**direct 路径**：
- 直接采用 Emotion Agent 结果，`is_sarcasm = false`，`is_mixed = false`

**sarcasm_suspected 路径**：

| 条件 | 处理 |
|------|------|
| `is_sarcasm = true` 且置信度 >= 0.65 | 规则裁决：采用讽刺 Agent 的 `true_emotion` 和 `revised_intensity`；加权置信度 = emotion*0.3 + sarcasm*0.7 |
| `is_sarcasm = true` 但置信度 < 0.65 | **升级 LLM**：讽刺检测不确定，规则无法裁决 |
| `is_sarcasm = false` | 规则裁决：采用 Emotion 结果，置信度 ×0.9 |

**mix 路径**：

| 条件 | 处理 |
|------|------|
| `is_mixed = true` 且置信度 >= 0.65 | 规则裁决：采用 Mix Agent 的 `primary_emotion`、`secondary_emotion`、`adjusted_intensity`；置信度按 `mix_ratio` 中主次情绪比例加权 |
| `is_mixed = true` 但置信度 < 0.65 | **升级 LLM**：混合情绪检测不确定，规则无法裁决 |
| `is_mixed = false` | 规则裁决：采用 Emotion 结果，置信度 ×0.9 |

### LLM 兜底触发条件

规则只处理确定的情况，不确定时升级到 LLM 仲裁：

1. Emotion Agent 置信度 < 0.65——情绪本身不可靠
2. Sarcasm/Mix 置信度 < 0.65——检测不确定，规则无法裁决
3. Sarcasm/Mix 与 Emotion 置信度差 <= 0.15——两个 Agent 打架，规则无法裁决

---

## 6. Chat Agent（对话生成器）

**目录**：`agents/chat_agent/`
**作用**：基于用户文本、对话历史和 Judge Agent 的情绪判定，生成共情式、可操作的对话回复。

### 输入（ChatInput）

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | `string` | 是 | -- | 用户最新消息 |
| `user_id` | `string \| null` | 否 | `null` | 用户 ID |
| `conversation_id` | `string \| null` | 否 | `null` | 对话 ID |
| `judge_result` | `object \| null` | 否 | `null` | Judge Agent 的输出 |
| `history` | `array` | 否 | `[]` | 对话历史，每项含 `role`（"user"/"assistant"）和 `content` |
| `metadata` | `object` | 否 | `{}` | 附加元数据 |

### 输出（ChatResult）

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `reply` | `string` | 是 | -- | 回复内容（不能为空） |
| `tone` | `string` | 是 | -- | 语气：`"supportive"` / `"calm"` / `"encouraging"` / `"reflective"` / `"crisis_support"` |
| `risk_hint` | `string` | 是 | -- | 风险提示：`"none"` / `"possible_crisis"` |
| `reason` | `string` | 否 | `""` | 生成理由 |

### 关键规则

- 历史消息最多取最近 20 条
- 危机检测：当 `risk_hint = "possible_crisis"` 时，语气必须为 `"crisis_support"`，不得提供危险方法，必须推荐专业帮助
- 回复 2-5 句话，引用对话中的具体事实，避免模板化
- 回复中不得包含情绪标签、置信度分数或 JSON 字段名

### 语气策略

| 语气 | 适用场景 |
|------|----------|
| `supportive` | 悲伤、失落、无奈 |
| `calm` | 愤怒、焦虑、厌烦 |
| `encouraging` | 疲惫、中性 |
| `reflective` | 开心（引导深层思考） |
| `crisis_support` | 检测到危机信号 |

### Fallback

无 fallback，LLM 错误时返回 HTTP 500。

---

## 7. Profile Agent（用户画像生成器）

**目录**：`agents/profile_agent/`
**作用**：基于历史情绪记录生成用户情绪画像。有两种模式：纯统计（不调用 LLM）和完整画像（调用 LLM）。

### 输入（ProfileInput）

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user_id` | `string` | 是 | -- | 用户 ID |
| `emotion_records` | `array` | 否 | `[]` | 历史情绪记录列表 |
| `chat_history` | `array` | 否 | `[]` | 对话历史 |
| `metadata` | `object` | 否 | `{}` | 附加元数据 |

**emotion_records 每项结构**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `string` | 记录 ID |
| `conversation_id` | `string` | 对话 ID |
| `message_id` | `string` | 消息 ID |
| `final_emotion` | `string` | 最终情绪标签 |
| `secondary_emotion` | `string \| null` | 次情绪 |
| `final_intensity` | `int` | 情绪强度，0-100 |
| `final_confidence` | `float` | 置信度，0-1 |
| `is_sarcasm` | `bool` | 是否讽刺 |
| `is_mixed` | `bool` | 是否混合情绪 |
| `raw_analysis_json` | `string` | 原始分析 JSON |
| `created_at` | `string` | ISO 时间戳 |

### 输出（ProfileResult）

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `total_records` | `int` | 特征提取 | 总记录数 |
| `emotion_distribution` | `object` | 特征提取 | 情绪分布 `{emotion: ratio}`，值为 0-1 |
| `avg_intensity` | `float` | 特征提取 | 平均情绪强度 |
| `avg_confidence` | `float` | 特征提取 | 平均置信度 |
| `sarcasm_rate` | `float` | 特征提取 | 讽刺比例 |
| `mixed_rate` | `float` | 特征提取 | 混合情绪比例 |
| `dominant_emotion` | `string` | 特征提取 | 主导情绪 |
| `intensity_trend` | `string` | 特征提取 | 强度趋势：`"上升"` / `"下降"` / `"平稳"` |
| `activity_pattern` | `object` | 特征提取 | 活跃时段 `{时段: 原始计数}`，4 个时段 |
| `personality_traits` | `string[]` | LLM | 性格特征 |
| `communication_style` | `string` | LLM | 沟通风格 |
| `emotional_patterns` | `string` | LLM | 情绪模式分析 |
| `mbti` | `string` | LLM | MBTI 人格类型（4 位字母或 "UNKNOWN"） |
| `summary` | `string` | LLM | 综合画像摘要 |
| `timeline` | `object` | 可视化 | 时间线数据（前端趋势图使用） |

### 活跃时段划分

| 时段 | 时间范围 |
|------|----------|
| 凌晨 | 0:00 - 5:59 |
| 上午 | 6:00 - 11:59 |
| 下午 | 12:00 - 17:59 |
| 晚上 | 18:00 - 23:59 |

### 趋势检测

使用简单线性回归分析强度值随时间的变化：
- 斜率 > 0.5 → "上升"
- 斜率 < -0.5 → "下降"
- 其余 → "平稳"

### MBTI 验证规则

- 必须为 4 个字符
- 第 1 位：`E` 或 `I`
- 第 2 位：`S` 或 `N`
- 第 3 位：`T` 或 `F`
- 第 4 位：`J` 或 `P`
- 无效时返回 `"UNKNOWN"`

---

## 完整流水线数据流

```
用户文本
    │
    ├──→ [Router Agent] ──→ sample_type, need_sarcasm_check, need_mix_check
    │
    ├──→ [Emotion Agent] ──→ emotion, intensity, confidence, 语言特征
    │         │
    │    (if need_sarcasm_check)
    │         │
    ├──→ [Sarcasm Agent] ──→ is_sarcasm, true_emotion, revised_intensity
    │         │
    │    (if need_mix_check)
    │         │
    ├──→ [Mix Agent] ─────→ is_mixed, primary/secondary_emotion, mix_ratio
    │         │
    └─────────┘
              │
              v
    [Judge Agent] ──→ final_emotion, secondary_emotion, final_intensity,
              │       final_confidence, is_sarcasm, is_mixed, reason
              v
    [Chat Agent] ──→ reply, tone, risk_hint, reason
              │
              v
         用户回复
```

**Profile Agent** 独立运行：接收历史 `emotion_records`，输出用户情绪画像。

---

## FastAPI 端点一览

| 端点 | 方法 | Agent | 入口方法 | 降级行为 |
|------|------|-------|----------|----------|
| `GET /health` | GET | -- | -- | 返回 `degraded` 状态 |
| `POST /router` | POST | RouterAgent | `route_dict` | 返回 `direct` + `fallback: true` |
| `POST /emotion` | POST | EmotionAgent | `emotionRe_dict` | 返回中性/低置信度 |
| `POST /sarcasm` | POST | SarcasmAgent | `detect_dict` | 返回非讽刺结果 |
| `POST /mix` | POST | MixAgent | `mixRe_dict` | 返回非混合结果 |
| `POST /judge` | POST | JudgeAgent | `judge_dict` | 先规则裁决，再回退到 emotion_result |
| `POST /chat` | POST | ChatAgent | `chat_dict` | 无 fallback（错误返回 500） |
| `POST /profile` | POST | -- | 纯函数 | 直接报错 |
| `POST /profile/generate` | POST | ProfileAgent | `generate_dict` | 返回仅统计数据 |

### 错误映射

| 异常类型 | HTTP 状态码 |
|----------|------------|
| `ValueError` | 400 |
| `HTTPError` / `URLError` | 502 |
| `TimeoutError` | 504 |
| 其他异常 | 500 |

---

## 关键架构特点

1. **Agent 自包含**：每个 Agent 包含独立的 `schemas.py`、`llm_agent.py`、`client.py`、`__init__.py`，通过 Protocol 接口解耦。

2. **两级执行**：Judge Agent 先用确定性规则裁决，仅在置信度模糊时升级到 LLM，兼顾效率和准确性。

3. **统一输入**：Router、Emotion、Sarcasm、Mix 四个 Agent 共享 `BaseTextInput` 结构，保证 API 一致性。

4. **温度策略**：分析类 0.1（确定性）→ 画像生成 0.3（轻微创造性）→ 对话生成 0.4（自然回复）。

5. **无内置编排**：FastAPI 将每个 Agent 暴露为独立端点，编排由 Java 后端负责。

6. **优雅降级**：每个 Agent 都有 fallback 函数，API_KEY 缺失时所有 Agent 置为 None，JudgeAgent 降级为纯规则模式。

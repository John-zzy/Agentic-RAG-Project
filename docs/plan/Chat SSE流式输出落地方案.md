# Chat SSE流式输出落地方案

## Summary

本方案用于把当前 `/chat` 的 `stream=true` 从“保留字段”落成“真实可用能力”，并且只交付今天范围内最适合闭环的一期能力：`SSE` 流式输出。

当前代码现状已经具备一半骨架：

- `ChatRequest` 已经有 `stream` 字段
- `ModelClient` 已经有 `stream()` 能力
- `/chat` 的同步主链、RAG 检索、citation 映射、session 落库都已经稳定可用

但真实缺口也很明确：

- `backend/application/runtime/service.py` 在 `payload.stream` 为 `true` 时仍直接抛 `501`
- API 层还没有 `text/event-stream` 返回分支
- 最终回答生成阶段使用的 RAG prompt 模板链还没有正式接通流式生成
- 文档与接口说明仍把 `stream` 标为“保留字段”

本次方案的固定目标如下：

- `POST /chat` 在 `stream=false` 时，继续维持当前 `ChatResponse` JSON 契约
- `POST /chat` 在 `stream=true` 时，返回 `text/event-stream`
- 流中至少能看到可消费的最终回答文本增量分片
- 流结束后，当前轮回答仍要按现有语义写入 `SQLiteSessionStore`


一句话概括本次方案：

“保持 `/chat` 现有同步契约不动，先同步完成 RAG 检索与证据准备，再在 API 层新增 SSE 分流，仅对最终回答阶段做流式输出，并保证流式结束后仍复用当前 session 持久化语义。”

## Key Changes

### 1. 固定 `/chat` 的双模式行为

`POST /chat` 本次正式固定为一个双模式接口：

- `stream=false`
  - 保持现状
  - 返回 `application/json`
  - 响应体继续是 `ChatResponse`
- `stream=true`
  - 返回 `text/event-stream`
  - 不再返回 `501`
  - 通过 SSE 持续输出最终回答文本分片与结束事件

本次不新增 `/chat/stream` 等新路由，避免把同一语义拆成两套入口。

这样做的原因是：

- 现有调用方已经围绕 `/chat` 建立心智
- `stream` 字段已经是现有契约的一部分
- 今天目标是把保留字段打通，而不是重新设计接口体系

### 2. 固定 SSE 事件契约

本次 SSE 事件统一使用：

- `event: start`
- `event: chunk`
- `event: done`
- `event: error`

并且所有事件都使用 JSON `data`，不混用纯文本与 JSON。

固定原因：

- 前后端后续扩展更稳定
- 可以自然携带 `session_id`、`request_id`、`citations` 等元数据
- 避免调用方在“文本流”和“结构化元数据”之间切换解析策略
- 避免把中间 retrieval 过程也暴露成另一套事件流

事件定义如下。

#### `start`

用途：

- 告知客户端本轮流式请求已经开始
- 返回服务端最终确定的 `session_id` 与 `request_id`
- 对“未传 `session_id` 自动建会话”的现有语义做流式场景适配

固定字段：

```json
{
  "session_id": "string",
  "request_id": "string",
  "scene": "generic_assistant",
  "agent": null,
  "knowledge_used": true
}
```

其中：

- `session_id`
  - 若请求未传 `session_id`，服务端先自动创建，再通过 `start` 返回
- `request_id`
  - 与当前同步接口保持一致，用于错误追踪与 session turn 关联
- `knowledge_used`
  - 在流开始前已经可以根据 retrieval + citations 判断

#### `chunk`

用途：

- 输出最终回答文本增量，不输出中间 retrieval 过程

固定字段：

```json
{
  "delta": "string"
}
```

约束：

- `delta` 是可直接拼接到当前回答缓冲区的文本片段
- 服务端不对每片 `delta` 做激进裁剪，避免吞掉空格、换行或标点
- 客户端若只关心展示，可以按顺序直接拼接 `delta`

#### `done`

用途：

- 标记本轮流式回答结束
- 一次性返回与同步接口同等级别的最终结构化结果

固定字段：

```json
{
  "session_id": "string",
  "request_id": "string",
  "answer": "string",
  "knowledge_used": true,
  "scene": "generic_assistant",
  "agent": null,
  "citations": []
}
```

其中：

- `answer`
  - 是最终权威答案
  - 允许比 chunk 拼接结果多出 `ensure_answer_citation_markers(...)` 补上的尾部引用编号
- `citations`
  - 与同步 `ChatResponse.citations` 结构保持一致

固定 `done` 携带完整元数据，而不是只返回“结束标记”，这样可以避免客户端在流结束后额外补查 session 才能拿到完整结果。

#### `error`

用途：

- 表示流已经进入流式阶段后发生失败

固定字段：

```json
{
  "code": "string",
  "message": "string",
  "request_id": "string"
}
```

错误边界固定为两类：

- 流启动前即可确定的错误
  - 例如 `SESSION_NOT_FOUND`、`SESSION_EXPIRED`、`SCENE_SESSION_MISMATCH`
  - 继续沿用当前 HTTP 错误响应，不进入 SSE
- 流启动后才发生的错误
  - 例如模型流式调用失败
  - 通过 `event: error` 返回

### 3. 固定无 `session_id` 时的流式行为

当 `stream=true` 且请求未传 `session_id` 时，本次行为固定为：

- 与当前同步 `/chat` 保持一致
- 服务端自动创建新会话
- 在 `start` 事件中返回新 `session_id`

本次不要求客户端必须先调用 `/sessions`，原因如下：

- 会破坏当前 `/chat` 的易用性
- 与已有同步路径不一致
- 不是今天目标，属于额外收紧调用约束

### 4. service 层拆出“共享准备阶段 + 同步/流式生成分支”

当前 `ChatService.chat()` 已经串起完整主链：

1. 生成 `request_id/session_id/timestamp`
2. 准备 session
3. 读取 mounted knowledge sources
4. 读取历史对话
5. 执行 retrieval
6. 构造 citations
7. 有知识命中时调用 LLM 生成答案
8. 无命中时返回 fallback
9. 将 answer 与 retrieval snippets 落库
10. 返回 `ChatResponse`

本次不应在流式路径重写一份近似逻辑，而是固定拆成两层：

#### 共享准备阶段

抽出一个内部准备结果，至少包含：

- `request_id`
- `session_id`
- `timestamp`
- `mounted_knowledge_sources`
- `history_text`
- `documents`
- `citations`
- `knowledge_used`
- `scene_metadata`

同步和流式都先走这一步，保证：

- retrieval 行为一致
- `knowledge_used` 判断一致
- citations 一致
- session 场景与 mounted source 语义一致

#### 同步生成分支

保持当前行为：

- 有知识命中时，按现有 RAG 模板链同步生成 answer
- 无知识命中时，使用 `fallback_policy.no_hit_message`
- 之后 `append_turn(...)`
- 返回 `ChatResponse`

#### 流式生成分支

新增 `chat_stream(...)`，语义固定为：

- 返回领域级流事件，而不是 FastAPI 的 `StreamingResponse`
- 这样 service 层仍然只负责业务语义，API 层负责协议封装

流式分支行为固定如下：

- 先完成共享准备阶段
- 共享准备阶段内部继续同步完成：
  - session 准备
  - history 读取
  - Agentic RAG / 文档召回
  - citations 计算
  - `knowledge_used` 判断
- 先产出 `start` 事件所需元数据
- 若 `knowledge_used=true`
  - 在 retrieval 已经完成、证据已经确定后，再进入最终回答生成阶段
  - 只对“最终回复用户的问题”这一段调用流式模板生成链，持续产出文本分片
  - 服务端在内存中累计完整 answer
- 若 `knowledge_used=false`
  - 不调用模型
  - 直接把 fallback 文本作为单个 `chunk`
- 结束前统一执行：
  - 如果需要，调用 `ensure_answer_citation_markers(...)`
  - `append_turn(...)` 写入最终 answer 与 retrieval snippets
  - 产出 `done`

### 5. 固定“检索不流式，只有最终回答流式”

本次需要明确一个边界：

- Agentic RAG 的检索、改写、切换工具、文档召回与证据聚合
  - 继续保持服务端内部同步完成
  - 不对外逐步流出中间检索过程
- 只有最终 answer generation 阶段
  - 才通过 SSE 向用户持续输出文本分片

这样做的原因是：

- 今天目标是补齐“用户可消费的最终回答流式输出”，不是暴露中间 RAG 执行轨迹
- 当前 retrieval、citation、session 持久化主链已经稳定，同步完成更容易保持行为一致
- 避免把“检索事件流”和“最终回答事件流”混成一套协议，增加客户端复杂度

因此本次 `stream=true` 的真实语义固定为：

- 先同步完成 retrieval 与证据准备
- 再对最终回答文本做 SSE 输出

### 6. 固定“无命中也走 SSE”

当 retrieval 没有得到足够证据时，当前同步行为是直接返回 fallback 文本。

本次流式模式固定为：

- 仍返回 `text/event-stream`
- 仍发送 `start`
- 发送一个 fallback `chunk`
- 最后发送 `done`

本次不允许“无命中时偷偷退回 JSON”，原因如下：

- `stream=true` 的协议应当稳定
- 客户端不应同时处理“SSE 或 JSON 回退”两种成功态
- fallback 也是回答，本质上同样属于可流式消费内容

### 7. 模型层补齐模板变量版流式能力

当前 `ModelClient` 已有：

- `invoke_template(...)`
- `invoke(prompt, ...)`
- `stream(prompt, ...)`

但现有 `stream(prompt, ...)` 只覆盖默认字符串模板，不足以直接服务“最终回答生成阶段”使用的 RAG prompt 模板链。

本次模型层新增的固定能力是：

- 一个模板变量版流式接口，例如 `stream_template(prompt_template, variables, complexity)`

其职责固定为：

- 复用当前 `get_runnable(...)` + `StrOutputParser()` 组合
- 在 retrieval 已完成、证据已准备好的前提下，按模板变量执行最终回答阶段的模型流式调用
- 持续返回文本片段

这样 `ChatService` 在流式生成时就不需要自己拼 LangChain 流式细节，只需要提供：

- 当前 RAG prompt
- `context`
- `input`
- `history`
- complexity

#### 为什么不直接复用 `create_stuff_documents_chain(...).stream(...)`

当前同步链使用 `create_stuff_documents_chain(...)` 是合理的，但本次方案固定倾向于把流式能力归口在 `ModelClient`，原因是：

- 现有模型调用已经集中在 `ModelClient`
- 后续不止 RAG，一般 prompt/template 流式也会复用
- service 层可以继续依赖“模型客户端协议”，而不是吸收更多 LangChain 细节

不过本次实现仍可保留当前同步 `create_stuff_documents_chain(...)` 路径，不要求顺手重构同步链路。

### 8. 固定 answer 累计与落库语义

当前 session 持久化发生在一次回答完成之后：

- `SQLiteSessionStore.append_turn(...)`
  - 写 `chat_turns`
  - 写 `chat_messages`
  - 更新 session 活跃时间

本次流式模式固定采用“结束后一次性落库”，不做分片级持久化。

固定原因：

- 与现有同步语义一致
- 不新增数据库结构
- 避免中途 chunk 写入导致脏半成品 session turn

因此：

- 服务端流式过程中只在内存里累计 answer buffer
- 只有准备发送 `done` 前，才进行最终 answer 固化和 `append_turn(...)`
- 若流中途异常且未完成最终 answer，则本轮不写入 `chat_turns`

### 9. API 层用 `StreamingResponse` 做协议分流

`backend/application/runtime/api/chat/routes.py` 本次固定新增按 `payload.stream` 分流：

- `stream=false`
  - 继续 `return service.chat(payload)`
- `stream=true`
  - 调用 `service.chat_stream(payload)`
  - 用 `StreamingResponse` 封装为 SSE 输出

实现边界固定如下：

- service 层产出结构化流事件
- route 层把结构化事件编码成：

```text
event: <name>
data: <json>

```

- response header 至少包含 `Content-Type: text/event-stream`

本次不引入额外第三方 SSE 框架，优先使用 FastAPI/Starlette 原生能力完成。

### 10. 对外契约变更范围固定为“新增 stream=true 成功态”

本次对外契约变化只允许发生在以下范围：

- `stream=true` 从“未支持”改为“支持 SSE”
- `docs/documents/reference/api-list.md` 更新说明
- `README.md` 增加最小验证命令

明确不变化的部分：

- `stream=false` 的 `ChatResponse` 字段
- `citations` 结构
- `/sessions` 接口
- session 存储结构
- mounted knowledge sources 行为
- `generic_assistant` 与 `ecommerce` 的路由/检索边界

## API Contract

### 1. `POST /chat` 请求体

请求体继续沿用：

```json
{
  "message": "string",
  "session_id": "string | null",
  "stream": true,
  "top_k": 4
}
```

本次不新增新字段，不改变默认值：

- `stream`
  - 默认仍为 `false`
  - 只有显式传 `true` 时才进入 SSE 模式

### 2. `stream=false` 响应

继续保持：

```json
{
  "session_id": "string",
  "request_id": "string",
  "answer": "string",
  "knowledge_used": true,
  "scene": "generic_assistant",
  "agent": null,
  "citations": []
}
```

### 3. `stream=true` 响应头

至少要求：

- `Content-Type: text/event-stream`

可选但推荐：

- `Cache-Control: no-cache`
- `Connection: keep-alive`

### 4. `stream=true` 事件流示例

```text
event: start
data: {"session_id":"s1","request_id":"r1","scene":"generic_assistant","agent":null,"knowledge_used":true}

event: chunk
data: {"delta":"根据产品手册，"}

event: chunk
data: {"delta":"AeroPhone X 售价 4599 元[1]。"}

event: done
data: {"session_id":"s1","request_id":"r1","answer":"根据产品手册，AeroPhone X 售价 4599 元[1]。","knowledge_used":true,"scene":"generic_assistant","agent":null,"citations":[...]}
```

### 5. 错误语义

固定错误行为如下：

- 进入 SSE 前失败
  - 返回当前 HTTP JSON 错误结构
- 已进入 SSE 后失败
  - 发送：

```text
event: error
data: {"code":"MODEL_INVOCATION_FAILED","message":"...","request_id":"r1"}
```

## Implementation Notes

### 1. 主要改动入口

本次主要落点固定为：

- `backend/application/runtime/api/chat/routes.py`
- `backend/application/runtime/service.py`
- `backend/platform/models/llm/client.py`

配套最小文档改动：

- `backend/application/runtime/api/chat/schemas.py`
- `docs/documents/reference/api-list.md`
- `README.md`

### 2. service 内部推荐拆分

为了避免 `chat()` 和 `chat_stream()` 复制大段逻辑，推荐固定拆出以下内部能力：

- `_prepare_chat_execution(...)`
  - 准备 session、history、documents、citations、scene metadata
- `_generate_answer_sync(...)`
  - 同步生成最终 answer
- `_generate_answer_stream(...)`
  - 流式生成 answer chunks，并累计完整 answer

本次实现不要求这些方法名必须完全一致，但要求职责边界一致。

### 3. 流式回答最终文本的权威来源

需要固定一个细节：

- `done.answer` 才是最终权威答案

原因：

- chunk 是增量输出
- 最终答案还可能经过：
  - 末尾 trim
  - citation markers 补齐

因此客户端若需要最终持久化、复制、再次展示，应优先使用 `done.answer`。

### 4. 对测试替身的最小要求

当前 `backend/tests/test_chat_api.py` 中的 `FakeModel` 只支持：

- `build_chat_model_for_complexity(...)`

本次需要为测试提供最小流式替身能力，例如：

- 新增模板流式方法
- 或新增专用假模型，按预设 chunk 列表输出

目标不是模拟真实 LLM，而是证明：

- API 层能返回 `text/event-stream`
- service 能按顺序发 `start/chunk/done`
- 结束后 session 真的写入了完整 answer

## Test Plan

### 1. `/chat` 非流式回归

保留并补强现有同步路径测试，至少断言：

- `stream=false` 或省略 `stream` 时状态码为 `200`
- 响应仍是 JSON
- 返回结构仍是 `ChatResponse`
- `session_id`、`request_id`、`answer`、`knowledge_used`、`scene`、`citations` 与当前语义一致
- session turn 仍然正常落库

### 2. `/chat` 流式成功路径

新增接口级测试，建议流程固定为：

1. 使用 `generic_assistant` 创建会话
2. 以 `stream=true` 调用 `POST /chat`
3. 使用 `TestClient.stream()` 消费原始事件流
4. 断言：
   - HTTP 状态码为 `200`
   - `content-type` 包含 `text/event-stream`
   - 事件顺序为 `start -> chunk... -> done`
   - 至少收到一个非空 `delta`
   - `done.answer` 等于最终完整回答
   - `done.session_id` 与本轮会话一致
   - `done.citations` 与当前同步语义兼容

### 3. `/chat` 流式无命中路径

新增接口级测试，固定断言：

- 仍返回 `text/event-stream`
- `start` 正常返回
- 只需一个 fallback `chunk`
- `done.knowledge_used` 为 `false`
- `done.citations` 为空
- 模型流式调用未触发

### 4. session 持久化验证

对流式成功与无命中两种路径，都需要额外验证：

- `append_turn(...)` 已执行
- `assistant_answer` 是最终完整文本，而不是半截 chunk
- `retrieval_snippets` 与同步模式保持同结构

可以通过以下任一方式验证：

- 直接查 `session_store.get_session_detail(...)`
- 或调用 `GET /sessions/{session_id}`

### 5. README 演示路径验证

README 至少补一条可人工执行的最小命令路径，建议固定为：

1. `POST /sessions`
2. 使用返回 `session_id`
3. `curl -N` 调 `POST /chat` 且 `stream=true`
4. `GET /sessions/{session_id}` 验证本轮已写入

## Acceptance

完成后，以下标准必须全部满足：

1. `POST /chat` 在 `stream=false` 时维持当前 `ChatResponse`
2. `POST /chat` 在 `stream=true` 时返回 `text/event-stream`
3. 事件流中至少能看到回答文本分片
4. 无命中场景下 `stream=true` 仍使用 SSE，而不是退回 JSON
5. 流结束后 session 中可以查到最终完整 answer 与 retrieval snippets
6. README 与 `docs/documents/reference/api-list.md` 已反映 `stream=true` 的真实能力

## Assumptions

- 本次只交付 SSE，不交付 WebSocket
- 本次不修改数据库 schema
- 本次不做分片级持久化，只在流式完成后一次性落库
- `done` 事件携带完整最终元数据，供客户端直接消费
- 对外只新增 `stream=true` 的成功态，不破坏已有同步 JSON 成功态

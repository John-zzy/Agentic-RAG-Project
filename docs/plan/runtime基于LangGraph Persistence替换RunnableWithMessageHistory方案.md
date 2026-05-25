# runtime基于LangGraph Persistence替换RunnableWithMessageHistory方案

## Summary

当前 `backend/application/runtime/service.py` 的 `/chat` 主链仍然使用 `RunnableWithMessageHistory` 管理多轮历史，因此会触发 LangChain 的弃用告警：

- `RunnableWithMessageHistory is deprecated`
- 官方推荐迁移到 `LangGraph` 的内建持久化机制

本方案的目标不是简单压掉 warning，而是把 `/chat` 的短期记忆主链正式迁移到 **LangGraph Persistence**：

- 用 `StateGraph + checkpointer` 替代 `RunnableWithMessageHistory`
- 用 `session_id` 作为 LangGraph 的 `thread_id`
- 保留现有 `/chat`、`/sessions`、`chat_messages`、`chat_turns` 的对外契约
- 在当前真实项目约束下，继续使用 **SQLite**，但新增 **LangGraph 专用 checkpoint 存储表**

一句话概括本次改造：

“让 runtime 的多轮历史从 LangChain 旧式 runnable history 包装器，迁移为 LangGraph 官方推荐的 thread-level persistence，同时保留当前项目的 session API 与消息视图不回归。”

## Key Changes

### 1. 将回答执行链从 RunnableWithMessageHistory 迁移到 LangGraph

当前问题：

- `ChatService._get_answer_runnable()` 仍返回 `RunnableWithMessageHistory`
- runtime 通过 `configurable.session_id` 驱动历史注入
- 历史写入依赖 `SQLiteChatMessageHistory`

目标改造：

- `ChatService` 不再缓存 `RunnableWithMessageHistory`
- 改为按 `TaskComplexity` 缓存编译后的 LangGraph `StateGraph`
- 每个 graph 只负责“最终回答生成”这一段，不改 retrieval 编排逻辑
- graph 使用 `MessagesState` 作为主 state
- graph 编译时挂载自定义 SQLite checkpointer
- graph 调用时通过：
  - `configurable.thread_id = session_id`
  - 固定 `checkpoint_ns = "chat_answer"` 或等价命名空间

这样改的结果：

- runtime 不再依赖已弃用的 history wrapper
- LangGraph 成为 `/chat` 多轮上下文的正式主机制
- 后续如果扩展更复杂的 runtime graph，不需要再推翻 memory 接口

### 2. 新增 LangGraph 专用 SQLite Checkpointer

由于项目当前真实落地环境是 SQLite，而不是 Postgres，本次不切换数据库，而是新增 **自定义 SQLite LangGraph Checkpointer**。

设计原则：

- 不强行复用 `chat_messages` 作为 LangGraph checkpoint 主存储
- 新增 LangGraph 专用表，尽量贴近官方 saver 模型
- 现有 `sessions / chat_messages / chat_turns` 继续保留为当前系统的读模型与兼容视图

建议新增三类表：

- `langgraph_checkpoints`
  - 保存 checkpoint 本体
  - 关键字段：`thread_id`、`checkpoint_ns`、`checkpoint_id`、`parent_checkpoint_id`、`checkpoint_payload`、`metadata_payload`、时间戳
- `langgraph_writes`
  - 保存 pending writes
  - 关键字段：`thread_id`、`checkpoint_ns`、`checkpoint_id`、`task_id`、`write_idx`、`channel`、`value_payload`、`task_path`、时间戳
- `langgraph_blobs`
  - 保存 channel version blob
  - 关键字段：`thread_id`、`checkpoint_ns`、`channel`、`version`、`blob_type`、`blob_payload`、时间戳

最小必须实现的 saver 方法：

- `get_tuple`
- `list`
- `put`
- `put_writes`
- `delete_thread`

这套实现要尽量遵循 LangGraph `BaseCheckpointSaver` 的行为，而不是做一个只够当前 happy path 的临时版本。

### 3. 会话迁移策略：不做一次性全量回填

当前线上或本地已有会话历史都在：

- `chat_messages`
- `chat_turns`

本次不建议一次性把所有历史会话全量转成 LangGraph checkpoints，因为：

- 迁移任务成本高
- 很多旧会话可能已经失效
- 出错后很难验证全量历史是否完全等价

建议采用 **懒迁移**：

- 当 `/chat` 收到请求时，先判断该 `session_id` 是否已有 LangGraph checkpoint
- 若已有 checkpoint：
  - 直接以当前用户消息作为新输入继续 graph
- 若没有 checkpoint：
  - 从现有 `chat_messages` 读取该 session 的历史消息
  - 以“历史消息 + 当前用户消息”作为 graph 首次输入
  - 让 graph 从这一轮开始建立自己的 checkpoint 链

这样做的好处：

- 无需全量数据迁移脚本
- 老会话在首次继续聊天时自然升级
- 新会话从第一轮开始就走新架构

### 4. 保持现有 API 与读模型不变

本次是 runtime memory 主链替换，不是 session API 重写。

必须保持不变的行为：

- `POST /chat` 非流式返回结构不变
- `POST /chat` SSE 事件结构不变
- `GET /sessions/{id}` 仍然返回当前的 message 视图
- `DELETE /sessions/{id}` 仍然返回 `deleted_messages`
- `chat_messages` 仍然是当前 session 详情接口的数据来源

因此 runtime 在 graph 成功完成后，仍然要继续同步现有持久化模型：

- 写入 `chat_turns`
- 写入 `chat_messages`
- 保留 assistant message 上的：
  - `request_id`
  - `timestamp`
  - `knowledge_used`
  - `citations`

也就是说：

- **LangGraph checkpoint** 是新的 runtime memory authority
- **chat_messages / chat_turns** 是当前对外 API 的兼容读模型

### 5. 流式输出继续保持现有业务事件语义

当前 `/chat` 的流式响应已经有稳定的业务事件协议：

- `start`
- `history`
- `tool`
- `chunk`
- `done`
- `error`

本次不能退化成只暴露 LangGraph 原始 stream。

推荐做法：

- graph 流式调用使用 LangGraph 的 `stream(..., stream_mode="messages", version="v2")`
- 仅筛选“最终回答节点”产出的 token
- 继续在 runtime 内部组装现有 SSE 事件协议

约束如下：

- `history` 事件仍然由 runtime 负责构造
- `tool` 事件仍然由 retrieval 阶段负责构造
- `chunk` 仍然只输出最终回答文本增量
- `done` 仍然返回权威 `ChatResponse`
- `error` 仍然由 API 层或 runtime 包装成当前标准错误事件

这样可以保证前端和测试不用因为 LangGraph 内部机制变化而整体重写。

## Implementation Strategy

建议按以下顺序实施：

### 阶段 1：先落 SQLite LangGraph Checkpointer

- 新增 LangGraph 专用 SQLite saver
- 补齐基础表结构、序列化和删除能力
- 先独立测试 saver 的 `put/get/list/delete`

这一阶段不要改 `/chat` 主链，只先把基础设施做稳。

### 阶段 2：把同步回答链切到 LangGraph graph

- `ChatService` 改为构建并缓存 answer graph
- 同步路径先切换到 `graph.invoke(...)`
- 用 `thread_id = session_id`
- 用懒迁移把旧 `chat_messages` 引入首轮 graph 输入

这一阶段先保证非流式稳定，再继续流式。

### 阶段 3：把流式回答链切到 LangGraph stream

- 流式路径改用 `graph.stream(..., stream_mode="messages")`
- 保持现有 `history/tool/chunk/done/error` 顺序和 payload
- 最终结果仍要从 graph 最终状态中提取权威 AI message

### 阶段 4：清理旧 runtime history 依赖

- 删除 `RunnableWithMessageHistory`
- 删除 `_active_request_id` / `_active_timestamp` 这种实例级活动上下文依赖
- runtime 不再直接依赖 `SQLiteChatMessageHistory` 作为主链 memory 接口

这一步还会顺带解决当前 `ChatService` 并发请求下实例字段可能串数据的问题。

## Test Plan

至少补充或确认以下测试场景：

### Checkpointer 层

- checkpoint 可成功写入并读取
- pending writes 可成功写入并恢复
- 相同 `thread_id` 可取回最新 checkpoint
- 删除 session 时可联动清理该 `thread_id` 的 LangGraph 数据

### Runtime 同步路径

- 同一 `session_id` 的后续问题能复用 graph 持久化历史
- 已存在旧 `chat_messages`、但无 graph checkpoint 的会话，可在首次请求时懒迁移成功
- `window_size` 仍能限制注入模型的历史窗口
- `ChatResponse` 契约不变

### Runtime 流式路径

- 仍输出 `start -> history -> tool -> chunk... -> done`
- 无知识命中时仍输出 `start -> history -> tool -> chunk -> done`
- 异常路径仍输出 `error`
- `done.answer` 仍是最终权威回答

### Session API 回归

- `/sessions/{id}` 仍返回现有 message 视图
- assistant message 的 metadata 不回归：
  - `request_id`
  - `timestamp`
  - `knowledge_used`
  - `citations`

## Assumptions

- 当前项目继续以 SQLite 作为真实运行环境，不切换到 Postgres
- 本次以“最新架构体系”为优先，因此选择 LangGraph 官方 persistence 心智，而不是继续包装 LangChain memory
- 现有 `chat_messages / chat_turns` 在本次改造中继续保留，不做立即删除
- 现有 `/chat`、`/sessions` 对外契约保持不变
- 本次目标是“彻底移除 `RunnableWithMessageHistory` 的主链依赖”，不是简单过滤 warning

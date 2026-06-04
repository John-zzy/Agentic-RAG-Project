# PRD-20260530-03 LangGraph Runtime 骨架设计方案

## Summary

本 PRD 的目标是先建立 LangGraph Runtime 最小骨架，而不是一次性迁移完整 Agentic Retrieval、Human-in-the-Loop 或 Workflow State Machine。

当前项目已有稳定的 `/chat`、SSE、session 读模型和 Agentic Retrieval trace，但最终回答链仍由 `ChatService` 通过 `RunnableWithMessageHistory` 连接 LangChain history。README P0 已将 LangGraph Runtime 骨架列为下一项，既有计划文档 `docs/plan/runtime基于LangGraph Persistence替换RunnableWithMessageHistory方案.md` 也明确了迁移方向。

本设计将原计划收敛为本 PRD 的最小边界：

- 先落 `Graph State`、`thread_id=session_id`、`request_id` metadata、checkpointer、run lifecycle、event mapping 的模块骨架。
- 保留现有 `/chat`、`/sessions`、`chat_messages`、`chat_turns` 对外契约。
- 不在本 PRD 中迁移 `AgenticRetriever` while 循环，不实现 interrupt/resume，不引入 Planner / Executor。

## Current Implementation

实施前现状：

- `backend/requirements.txt` 已包含 `langgraph==1.2.0`。
- 当前本地环境可导入 `langgraph` 和 `langgraph.checkpoint.memory`，但不能导入 `langgraph.checkpoint.sqlite`。
- `backend/application/runtime/service.py` 仍导入并使用 `RunnableWithMessageHistory`。
- `ChatService._get_answer_runnable()` 每次请求包装基础 answer runnable，并通过 `SQLiteChatMessageHistory` 注入历史。
- `ChatService._build_runnable_config()` 仍使用 `configurable.session_id`，不是 LangGraph 的 `configurable.thread_id`。
- SSE 已有稳定业务事件：`start / history / tool / chunk / done / error`。
- `SQLiteSessionStore.delete_session()` 只清理 `chat_messages`、`chat_turns` 和 `sessions`，没有 LangGraph checkpoint 清理路径。

## Implementation Gap Matrix

以下为实施前口径，不把本文档设计本身计入实现完成度。

| PRD 功能需求 | 当前状态 | 证据 | 缺口 |
| --- | --- | --- | --- |
| 复用既有 LangGraph Persistence 计划并明确最小边界 | 部分满足 | 已有 `docs/plan/runtime基于LangGraph Persistence替换RunnableWithMessageHistory方案.md` | 原计划目标是彻底替换 `RunnableWithMessageHistory`，本 PRD 需要先收敛为 runtime 骨架 |
| 定义最小 Graph State | 未满足 | 代码中无 LangGraph state 类型或 state 模块 | 缺少 `session_id / request_id / messages / answer / knowledge_used / citations / retrieval_trace` 等状态契约 |
| 约定 `thread_id=session_id`、`request_id` metadata，并保留外部契约 | 部分满足 | 现有 `/chat`、`/sessions`、`request_id`、session 读模型已稳定 | 还没有 graph config metadata，也没有将 `thread_id` 绑定到 `session_id` |
| 接入 SQLite checkpointer 或适配层，验证写读列删清理 | 未满足 | requirements 只有 `langgraph`，本地无 `langgraph.checkpoint.sqlite`；session store 无 checkpoint 表 | 缺少 SQLite saver、checkpoint 表、`put/get_tuple/list/put_writes/delete_thread` 测试 |
| 建立 graph run lifecycle：created/running/succeeded/failed | 未满足 | 只有业务 SSE start/done/error，没有 graph run 记录 | 缺少 run lifecycle 数据结构、状态转换和 trace/event 映射 |
| 设计 LangGraph stream event 到现有 SSE 的映射 | 部分满足 | 现有 SSE 协议和测试稳定 | 缺少 LangGraph 原始 event 到业务 `ChatStreamEvent` 的隔离适配层 |

统计：6 项功能需求中，已满足 0 项，部分满足 3 项，未满足 3 项。若按“必须有代码闭环”口径，6 项均未完整满足。

## Target Architecture

建议新增一个中立的 platform workflow runtime 小模块，并由 application runtime 负责装配：

```text
backend/platform/workflow/langgraph/
  state.py          # RuntimeGraphState / reducer / state 序列化边界
  checkpointer.py   # SQLiteLangGraphCheckpointer 或官方 saver 适配层
  lifecycle.py      # GraphRunRecord / GraphRunStatus / 状态转换
  events.py         # LangGraph event -> ChatStreamEvent 语义映射契约

backend/application/runtime/
  graph_runtime.py  # 面向 ChatService 的最小 graph runner facade
  service.py        # 保留 chat 协议，逐步把回答执行委托给 graph runtime
```

边界原则：

- `platform/workflow/langgraph` 只承载中立 runtime 能力，不感知 scene、RAG 业务或 FastAPI。
- `application/runtime/graph_runtime.py` 负责把 `PreparedChatTurn` 转为 graph input/config，并把 graph output 转回 `ChatResponse` 所需字段。
- `ChatService` 继续拥有 `/chat` 主链、retrieval 准备、外部响应契约和兼容读模型写入。
- `scenes` 暂不改动，Agentic Retrieval 仍由现有 `RetrievalExecutor` 和 scene definition 解析 candidate tools。

## Minimal Graph State

最小 state 建议使用 `TypedDict`，避免过早引入复杂模型对象序列化：

```python
class RuntimeGraphState(TypedDict, total=False):
    session_id: str
    request_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    user_message: str
    answer: str
    knowledge_used: bool
    citations: list[dict[str, Any]]
    retrieval_trace: dict[str, Any]
    metadata: dict[str, Any]
```

字段职责：

- `session_id`：业务会话 ID，也是 LangGraph `thread_id`。
- `request_id`：单次 graph run metadata，用于 API 响应、日志、checkpoint metadata 和 SSE 关联。
- `messages`：graph 内部的可持久化对话消息状态。初期可由 `chat_messages` 懒加载种子历史。
- `answer`：本次最终权威回答文本。
- `knowledge_used`、`citations`、`retrieval_trace`：保持现有 RAG 可解释契约，不让前端从 LangGraph 原始事件反推。
- `metadata`：存放 scene、agent、answer_mode、final_decision 等非稳定扩展字段。

## Thread ID And Metadata

Graph 调用配置统一约定：

```python
config = {
    "configurable": {
        "thread_id": prepared.session_id,
        "checkpoint_ns": "chat_runtime",
    },
    "metadata": {
        "request_id": prepared.request_id,
        "session_id": prepared.session_id,
        "scene": prepared.scene_metadata.scene,
        "agent": prepared.scene_metadata.agent,
    },
}
```

兼容策略：

- 外部仍暴露 `session_id` 和 `request_id`，不新增必须由前端理解的 graph ID。
- `thread_id` 只作为 LangGraph runtime persistence key，等价于现有 session。
- `request_id` 不作为 thread key，避免一轮请求创建一个不可恢复线程；它只描述本次 run。

## Checkpointer Design

由于当前环境没有 `langgraph.checkpoint.sqlite`，本 PRD 推荐先实现项目内 SQLite saver，后续若官方包可用再以 adapter 替换内部实现。

最小表结构：

- `langgraph_checkpoints`
  - `thread_id`
  - `checkpoint_ns`
  - `checkpoint_id`
  - `parent_checkpoint_id`
  - `checkpoint_payload`
  - `metadata_payload`
  - `created_at`
- `langgraph_writes`
  - `thread_id`
  - `checkpoint_ns`
  - `checkpoint_id`
  - `task_id`
  - `write_idx`
  - `channel`
  - `value_payload`
  - `task_path`
  - `created_at`
- `langgraph_blobs`
  - 如当前 LangGraph 版本需要 channel blob/version 存储，再补齐；若 saver 序列化模式暂不需要，可作为 Phase 2。

最小方法闭环：

- `put(config, checkpoint, metadata, new_versions)`：写 checkpoint，返回包含最新 `checkpoint_id` 的 config。
- `get_tuple(config)`：按 `thread_id/checkpoint_ns/checkpoint_id` 读取；未指定 checkpoint 时取最新。
- `list(config, before=None, limit=None, filter=None)`：按 thread 或全局列出 checkpoint。
- `put_writes(config, writes, task_id, task_path="")`：记录 pending writes。
- `delete_thread(thread_id)`：删除该 thread 下 checkpoints/writes/blobs。

清理路径：

- `SQLiteSessionStore.delete_session()` 不能直接依赖 LangGraph saver，避免 memory 层反向依赖 workflow runtime。
- 推荐在 `ActiveSceneChatService.delete_session()` 或 API route service facade 中编排：先调用 `checkpointer.delete_thread(session_id)`，再调用 `session_store.delete_session(session_id)`。
- 若短期不改 route，可在文档和测试中标记为明确待办，但本 PRD 验收要求至少应有可调用清理路径。

## Graph Run Lifecycle

最小 lifecycle 不等于完整 Workflow State Machine。它只记录一次 graph run 的执行状态：

```text
created -> running -> succeeded
created -> running -> failed
```

建议新增：

```python
GraphRunStatus = Literal["created", "running", "succeeded", "failed"]

class GraphRunRecord(TypedDict):
    run_id: str
    thread_id: str
    request_id: str
    status: GraphRunStatus
    created_at: str
    updated_at: str
    error: str | None
```

初期可以只用内存/日志记录加测试断言；若需要跨进程可观察，再持久化到 `langgraph_runs`。不要在本 PRD 扩展到 `waiting_user / retrying / cancelled`，这些属于后续 HITL 和 Workflow State Machine。

## Stream Event Mapping

外部 SSE 协议保持不变，新增内部 mapper 隔离 LangGraph 原始事件：

| LangGraph 内部事件 | Runtime 业务事件 | 说明 |
| --- | --- | --- |
| graph run created | `start` | 由 runtime 构造，携带 `session_id/request_id/knowledge_used/scene/agent` |
| history snapshot | `history` | 继续从兼容读模型或 graph seed messages 构造 |
| retrieval prepared / tool result | `tool` | 当前仍由 `RetrievalExecutor` 产出，不暴露 LangGraph 原始 tool event |
| answer node token/message delta | `chunk` | 只输出最终回答节点的文本增量 |
| graph succeeded | `done` | 输出权威 `ChatResponse.model_dump()` |
| graph failed / model failed | `error` | 沿用现有错误 payload：`code/message/request_id` |

约束：

- 前端和 Eval Harness 不消费 LangGraph 原始 event name。
- `done.retrieval_trace` 仍必须与 `tool.retrieval_trace` 保持一致。
- 非证据分支 `ask_user / no_evidence / max_rounds_reached / retrieval_failed` 可以先不进入 LLM answer node，但仍应写入 graph state/checkpoint，保证 run lifecycle 完整。

## Delivery Phases

### Phase 1：Checkpointer 独立闭环

- 新增 SQLite saver 和表结构。
- 补 saver 定向测试：`put/get_tuple/list/put_writes/delete_thread`。
- 不改 `/chat` 主链。

### Phase 2：Runtime graph skeleton

- 新增 `RuntimeGraphState`。
- 新增最小 answer graph：输入 prepared state，执行一个 answer/finalize node，产出 `answer/citations/retrieval_trace`。
- graph config 使用 `thread_id=session_id`，metadata 写入 `request_id`。
- 添加最小 graph run 测试，确认 checkpoint metadata 与 state 一致。

### Phase 3：ChatService 接入 facade

- `ChatService` 保留 retrieval 和响应组装。
- 仅把最终回答生成委托给 `application/runtime/graph_runtime.py`。
- 成功后继续写入 `chat_turns/chat_messages`，保证 `/sessions` 不回归。

### Phase 4：SSE stream mapper

- 将 graph stream 事件收敛为现有 `ChatStreamEvent`。
- 保持 `start -> history -> tool -> chunk... -> done` 和错误路径测试。

## Why Skeleton First

本次先做 Runtime 骨架，而不是直接做完整 Workflow State Machine 或 HITL，原因是：

- 当前 `/chat` 协议、Eval Harness 和 session 读模型已经稳定，直接大迁移会扩大回归面。
- HITL 需要 interrupt/resume、等待态、用户动作协议和前端交互，不应和 persistence 基础设施混在同一个 PRD。
- Agentic Retrieval Subgraph 迁移会触及 query rewrite、tool routing、rerank、充分性判断和 fallback，属于更高风险的行为迁移。
- 先把 state、thread、checkpoint、lifecycle、event mapping 打通，可以为后续子图迁移提供可验证的落点。

## Verification Plan

定向测试：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_checkpointer.py backend\tests\test_langgraph_runtime.py -q -c backend\tests\pytest.ini
```

回归测试：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py backend\tests\test_session_store.py backend\tests\test_eval_assets.py -q -c backend\tests\pytest.ini
```

验收检查：

- 同一 `thread_id=session_id` 可写入并读取最新 checkpoint。
- `request_id` 出现在 graph run metadata 或 checkpoint metadata 中。
- 删除 session 时有 checkpoint 清理路径。
- `/chat` JSON、SSE、`/sessions` response schema 不变。
- SSE mapper 不暴露 LangGraph 原始事件名。



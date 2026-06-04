# 请求上下文隔离：ChatService 实例级 mutable state 去除方案

## 背景

`ActiveSceneChatService` 会按 scene 缓存 `ChatService` 实例，实例级对象可以复用，但单次 `/chat` 请求的 `request_id`、`timestamp`、历史消息写入上下文不能复用。若这些字段挂在 `ChatService` 实例字段上，并由 `_get_session_history()` 或 `_persist_turn()` 隐式读取，并发请求和流式请求可能把前一轮元数据写入后一轮消息，破坏 session history、SSE trace 和后续 checkpoint 的可信度。

本方案的边界是请求上下文隔离，不改变现有 `/chat` 响应模型、SSE 事件结构、session schema、retrieval trace 语义或 LangGraph Runtime 骨架。

## 当前路径

当前工作树已经按请求级上下文实现了主链路：

1. `ChatService.chat()` 和 `chat_stream()` 每次调用都先进入 `_prepare_chat_turn()`。
2. `_prepare_chat_turn()` 生成本次请求的 `request_id`、`session_id`、`timestamp`，并返回不可变的 `PreparedChatTurn`。
3. JSON 和 SSE 共用同一个 `PreparedChatTurn`，后续回答生成、history event、tool event、done event 与持久化都从该对象取元数据。
4. `_get_answer_runnable()` 每次请求创建 `RunnableWithMessageHistory` 包装器，闭包捕获本次 `PreparedChatTurn`，并把 `request_id`、`timestamp` 显式传入 `_get_session_history()`。
5. `SQLiteChatMessageHistory` 只保存构造时传入的请求元数据，并在 `add_messages()` 中传给 `SQLiteSessionStore.append_messages()`。
6. `_persist_turn()` 显式把 `PreparedChatTurn.request_id` 与 `timestamp` 传给 `append_turn()`，最终由 `chat_turns` 与 `chat_messages` 共用同一请求元数据。

## 目标架构

请求上下文应作为 application runtime 的值对象在调用栈中传递：

```text
ChatRequest
  -> _prepare_chat_turn()
  -> PreparedChatTurn(request_id, timestamp, session_id, retrieval result, citations)
  -> JSON path: _generate_answer() -> _persist_turn() -> _build_chat_response()
  -> SSE path: start/history/tool/chunk -> _persist_turn() -> done
  -> SQLiteChatMessageHistory / append_messages / append_turn
```

允许缓存的对象：

- `ChatService` 实例
- scene definition、retriever、citation mapper
- 不带请求上下文的 base runnable 缓存

禁止共享的对象或字段：

- 当前请求的 `request_id`
- 当前请求的 `timestamp`
- 当前请求的 history 写入 metadata
- 当前请求的 citations / `knowledge_used` / final answer metadata

## 方案设计

### 1. 请求上下文对象

以 `PreparedChatTurn` 作为请求级上下文载体，字段保持不可变。所有后续函数只接收 `prepared` 或显式的 `request_id`、`timestamp` 参数，不读取 `ChatService` 临时字段。

关键约束：

- `_prepare_chat_turn()` 是唯一生成 `request_id` 与 `timestamp` 的入口。
- `_build_history_event()`、`_build_tool_event()`、`_persist_turn()`、`_build_chat_response()` 均从同一个 `PreparedChatTurn` 取值。
- 任何新增 helper 如果需要请求元数据，必须通过参数传入。

### 2. 历史消息适配

`RunnableWithMessageHistory` 的 history factory 不应读取实例字段，而是在本次请求内构造闭包：

```text
_get_answer_runnable(prepared)
  -> history_factory(session_id)
  -> _get_session_history(session_id, request_id=prepared.request_id, timestamp=prepared.timestamp)
```

这样即使同一 `ChatService` 被两个线程同时调用，两个 history factory 也分别绑定各自的请求上下文。

### 3. 持久化一致性

`append_turn()` 继续保持现有 schema，不新增字段。它应接收显式 `request_id`、`timestamp`，并负责：

- 写入 `chat_turns.request_id / created_at`
- 写入或同步 `chat_messages.request_id / created_at`
- assistant message `additional_kwargs` 中写入同一 `request_id`、`timestamp`、`knowledge_used`、`citations`

### 4. SSE 一致性

SSE 不扩展协议字段。当前协议下：

- `start`、`history`、`tool`、`done` 均带有同一 `request_id`
- `done` 复用 `_build_chat_response()` 的结果
- `history` event 展示的是模型调用前的窗口快照，当前消息最终由 `done` 后的 session detail 验证
- `timestamp` 不在 ChatResponse / SSE 顶层协议中暴露，仍通过 `/sessions/{session_id}` 的 message view 验证

## 需求匹配

| 编号 | 功能需求 | 当前状态 | 证据 |
| --- | --- | --- | --- |
| FR-1 | 梳理 `chat`、`chat_stream`、`_prepare_chat_turn`、`_get_session_history`、`_persist_turn` 上下文流转 | 已满足 | `backend/application/runtime/service.py` 已形成 `PreparedChatTurn` 流转 |
| FR-2 | `request_id`、`timestamp` 改为请求级显式参数，去除实例字段依赖 | 已满足 | 未发现 `_history_request_id`、`_history_timestamp`、`_active_request_id`、`_active_timestamp` 残留 |
| FR-3 | `SQLiteChatMessageHistory`、`append_turn`、`_build_history_event`、SSE 从同一请求上下文读取元数据 | 已满足 | history factory、history event、tool event、persist turn 均从 `PreparedChatTurn` 取值 |
| FR-4 | 保持单请求行为、session 持久化、scene 绑定与 `/chat` 返回格式不变 | 已满足 | 定向测试 `test_chat_api.py`、`test_session_store.py` 通过 |
| FR-5 | 补 README / Demo Path / tracker 输出说明并发验证方式 | 已满足 | README 已包含“请求上下文隔离 Demo Path” |

当前 5 条功能需求中，未满足数量：0。

## 剩余验证缺口

以下不是功能缺失，但可作为后续测试加固项：

- 当前已有并发 cached service 非流式测试，尚未单独增加并发 `chat_stream` 测试。
- 当前 SSE success 测试校验事件结构和 trace 一致性，但没有显式断言 `start/history/tool/done` 的 `request_id` 全部相等。
- `timestamp` 未出现在 ChatResponse / SSE 顶层协议中，按“保持返回格式不变”原则不建议为本 PRD 新增响应字段；验证应落在 `/sessions/{session_id}` message view。

## 验收检查

- 定向命令：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py backend\tests\test_session_store.py -q -c backend\tests\pytest.ini
```

- 当前结果：`55 passed, 19 warnings`。
- warnings 均为 `RunnableWithMessageHistory` 弃用提示，属于后续 LangGraph Persistence 替换范围，不影响本 PRD。



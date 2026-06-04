# PRD-20260601-02 Workflow State Machine 基础状态接入与状态转移边界设计方案

## Summary

本 PRD 的目标是把当前 LangGraph Runtime 和 HITL 最小闭环之上的 run 状态语义收敛成一套最小 Workflow State Machine。它不是长期任务队列，也不是完整调度系统，而是先定义一次 graph run 在 runtime 内部、checkpoint、SSE/API 和测试中共用的状态口径。

目标状态集合：

```text
created / planning / running / waiting_user / retrying / succeeded / failed / cancelled
```

核心边界：

- `waiting_user` 是可恢复等待态，不是失败。
- `succeeded / failed / cancelled` 是终态，不能继续 resume、retry 或重新写成 running。
- `resume_approve` 和 `resume_respond` 从 `waiting_user` 回到 `running`。
- `resume_reject` 默认从 `waiting_user` 进入 `cancelled`，表示人工拒绝终止本次等待任务；它不是模型错误。
- `tool_error` 只有在 retry policy 允许时进入 `retrying`；不可重试或超过次数进入 `failed`。
- LangGraph checkpoint 保存最新可恢复状态；lifecycle/event log 保存状态转移审计。

## Current Implementation Audit

| 能力 | 当前位置 | 当前状态 | 结论 |
| --- | --- | --- | --- |
| README P0 路线 | `README.md` | `Workflow State Machine` 明确是 P0 未完成项 | 本 PRD 顺延路线正确 |
| graph run lifecycle | `backend/platform/workflow/langgraph/lifecycle.py` | `GraphRunStatus = created/running/succeeded/failed`，内存事件记录 | 只有最小生命周期，不是完整状态机 |
| graph checkpoint state | `backend/platform/workflow/langgraph/state.py` | `RuntimeGraphStatus = running/waiting_user/succeeded/failed`，包含 `hitl/hitl_resume` | 能表达 HITL 等待，但缺 `created/planning/retrying/cancelled` |
| LangGraph runtime facade | `backend/application/runtime/assembly/runtime_factory.py` | 普通 invoke 创建 run 后直接 `running -> succeeded/failed`；HITL wait 写 checkpoint `status=waiting_user` | 已有状态写入点，但缺统一转移校验 |
| HITL resume | `backend/application/runtime/assembly/runtime_factory.py` | resume 只允许当前 checkpoint 为 `waiting_user`，重复 resume 会被拒绝 | 已有局部幂等保护，但不是通用终态保护 |
| `/chat` / `/chat/resume` | `backend/application/runtime/service.py`、`backend/application/runtime/api/chat/routes.py` | JSON/SSE 已暴露 `waiting_user`、`resume`、`done/error` | 外部协议已有接入点，但状态字段不完整 |
| SSE mapper | `backend/application/runtime/stream_events.py` | 映射 `start/history/tool/chunk/waiting_user/resume/done/error` | 事件序列可兼容扩展状态字段 |
| session persistence | `backend/platform/memory/base/session_store.py` | `sessions.status = active/expired`，保存聊天会话生命周期 | 会话状态与 workflow run 状态应继续分离 |
| LangGraph checkpointer | `backend/platform/workflow/langgraph/checkpointer.py` | `thread_id=session_id`，checkpoint/blobs/writes 持久化 | 适合保存最新 workflow state，不适合作为审计日志唯一来源 |
| tests | `backend/tests/test_langgraph_runtime.py`、`backend/tests/test_generic_assistant_hitl.py` | 覆盖 success/failed lifecycle、waiting_user、approve/reject/respond、重复 resume | 缺 retrying/cancelled/终态通用拒绝测试 |

## Gap Count

按 PRD 的 6 个需求目标计数，当前代码层面“未完全满足”为 **5/6**：

| PRD 目标 | 当前满足度 | 证据与缺口 |
| --- | --- | --- |
| 走查当前 LangGraph Runtime、SSE、session persistence、chat run 生命周期 | 已通过本设计走查完成 | 这是设计交付物，不是代码能力 |
| 定义 `WorkflowRunState` 八状态集合 | 部分满足 | lifecycle 有 `created/running/succeeded/failed`；checkpoint/API 有 `running/waiting_user/succeeded/failed`；缺统一 enum，缺 `planning/retrying/cancelled` |
| 设计状态转移表 | 未满足 | 当前没有集中 transition table 或 validator |
| 终态保护规则 | 部分满足 | HITL 重复 resume 因 `status != waiting_user` 被拒绝；但没有对 `succeeded/failed/cancelled` 的统一 resume/retry/running 写入保护 |
| `waiting_user` 与 `failed` 边界 | 部分满足 | checkpoint 能保存 `waiting_user`，reject 不是异常；但 lifecycle 在 `create_hitl_wait()` 后记为 `succeeded`，不能表达“仍在等用户” |
| 最小接入点：状态写入、SSE 字段、测试断言、checkpoint 关系 | 部分满足 | 写入点和 SSE 事件已有，但缺 `state_event/final_state` 字段约定、状态机测试和 checkpoint/lifecycle 分工 |

更细分到功能点，当前项目已满足 2 项、部分满足 5 项、未满足 6 项：

| 功能点 | 状态 |
| --- | --- |
| LangGraph checkpoint 可持久化 graph state | 已满足 |
| SSE 已有 `waiting_user/resume/done/error` 业务事件 | 已满足 |
| RuntimeGraphState 有 `status/hitl/hitl_resume` | 部分满足 |
| Chat API 有可选 `status/hitl` | 部分满足 |
| HITL resume 有等待点校验和重复 resume 局部保护 | 部分满足 |
| reject/等待态不作为模型异常处理 | 部分满足 |
| 测试覆盖 HITL approve/reject/respond | 部分满足 |
| 统一 `WorkflowRunState` 八状态 enum | 未满足 |
| `planning` 状态和 planner 边界 | 未满足 |
| `retrying` 状态、retry count 和 retry policy | 未满足 |
| `cancelled` 状态和 cancel/reject 语义 | 未满足 |
| 集中状态转移表与非法转移拒绝 | 未满足 |
| 终态 resume/retry/running 写入统一拒绝测试 | 未满足 |

## Target State Semantics

| 状态 | 类型 | 语义 | 可恢复性 |
| --- | --- | --- | --- |
| `created` | 初始态 | run 已分配 `run_id/request_id/thread_id`，还未进入规划或执行 | 可进入 `planning/running/cancelled` |
| `planning` | 运行中 | runtime 正在做 scene、policy、候选工具、任务计划或 graph 路由准备 | 可进入 `running/waiting_user/failed/cancelled` |
| `running` | 运行中 | graph 节点、模型、工具或检索正在执行 | 可进入 `waiting_user/retrying/succeeded/failed/cancelled` |
| `waiting_user` | 暂停态 | graph 已 interrupt，checkpoint 持有 `hitl`，等待人工输入 | 只能被 resume 或 cancel 推进 |
| `retrying` | 运行中 | 上一次模型/工具失败可重试，runtime 正在等待或准备下一次尝试 | 可进入 `running/failed/cancelled` |
| `succeeded` | 终态 | run 已正常产出最终结果 | 不可再推进 |
| `failed` | 终态 | run 因系统、模型、工具或不可恢复异常失败 | 不可再推进 |
| `cancelled` | 终态 | run 被用户、系统策略或人工 reject/cancel 终止 | 不可再推进 |

状态字段建议统一命名为 `WorkflowRunState`，由 `platform.workflow` 提供中立定义，application runtime 复用。

```python
WorkflowRunState = Literal[
    "created",
    "planning",
    "running",
    "waiting_user",
    "retrying",
    "succeeded",
    "failed",
    "cancelled",
]
```

## Transition Table

| 当前状态 | 事件 | 下一状态 | 合法性 | 说明 |
| --- | --- | --- | --- | --- |
| none | `create` | `created` | 合法 | 分配 run id，写 lifecycle event；可暂不写 checkpoint |
| `created` | `plan_start` | `planning` | 合法 | 开始 scene/policy/plan 准备 |
| `created` | `run_start` | `running` | 合法 | 简单路径可跳过 planning |
| `created` | `cancel` | `cancelled` | 合法 | 请求尚未执行时取消 |
| `planning` | `run_start` | `running` | 合法 | 规划完成，进入执行 |
| `planning` | `interrupt` | `waiting_user` | 合法 | 规划阶段需要用户补充信息 |
| `planning` | `tool_error` | `failed` | 合法 | 规划阶段失败通常不可重试，除非后续定义 planner retry |
| `planning` | `cancel` | `cancelled` | 合法 | 人工或系统取消 |
| `running` | `interrupt` | `waiting_user` | 合法 | HITL checkpoint 必须包含 `hitl.interrupt_id` |
| `running` | `tool_error_retryable` | `retrying` | 合法 | 记录 `last_error/retry_count/next_retry_at` |
| `running` | `tool_error_final` | `failed` | 合法 | 不可重试或达到最大次数 |
| `running` | `success` | `succeeded` | 合法 | 产出最终 answer/result |
| `running` | `cancel` | `cancelled` | 合法 | 执行期间取消 |
| `waiting_user` | `resume_approve` | `running` | 合法 | 清空当前 `hitl`，记录 `hitl_resume.action=approve` |
| `waiting_user` | `resume_respond` | `running` | 合法 | 清空当前 `hitl`，把用户补充输入并入后续执行 |
| `waiting_user` | `resume_reject` | `cancelled` | 合法 | 人工拒绝后终止本 run；不是 failed |
| `waiting_user` | `cancel` | `cancelled` | 合法 | 超时、用户关闭或系统取消 |
| `waiting_user` | `success/tool_error_retryable/tool_error_final` | - | 非法 | 等待态没有活跃执行，不能直接成功或失败 |
| `retrying` | `retry` | `running` | 合法 | 按 retry policy 进入下一次尝试 |
| `retrying` | `tool_error_retryable` | `retrying` | 合法 | 更新 retry count，但不能超过 max |
| `retrying` | `tool_error_final` | `failed` | 合法 | 超过上限或错误不可恢复 |
| `retrying` | `cancel` | `cancelled` | 合法 | 重试等待中取消 |
| `succeeded` | `resume/retry/run_start/cancel/success/tool_error` | - | 非法 | 终态保护 |
| `failed` | `resume/retry/run_start/cancel/success/tool_error` | - | 非法 | 终态保护；如需恢复，应创建新 run 或显式 fork |
| `cancelled` | `resume/retry/run_start/cancel/success/tool_error` | - | 非法 | 终态保护 |

说明：

- 本 PRD 中 `resume_reject -> cancelled` 是目标语义；当前 HITL 最小闭环里 reject 返回 `succeeded`，后续实现需要迁移或兼容说明。
- 如果某些业务图未来允许“拒绝可选工具后走替代路径”，应建模为 `resume_reject_continue -> running`，不要把所有 reject 混成成功。
- `failed` 只用于系统、模型、工具或不可恢复执行错误，不用于用户拒绝、取消或等待用户。

## Terminal Protection Rules

终态集合：

```text
succeeded / failed / cancelled
```

保护规则：

1. 任意状态写入前，先读取当前 thread 最新 checkpoint 或 run record。
2. 如果当前状态在终态集合内，拒绝以下事件：`resume_approve/resume_respond/resume_reject/retry/run_start/interrupt/success/tool_error_*`。
3. 终态 run 只能执行只读查询、审计导出、删除会话或创建新 run；不能原地复活。
4. `/chat/resume` 对终态应返回 `409 WORKFLOW_TERMINAL_STATE`，错误体包含 `current_state`、`requested_event`、`request_id`。
5. retry 必须只从 `retrying` 出发；`failed -> retrying` 不允许隐式发生。后续若做 Failure Recovery，应通过显式 `recover` 创建新 run 或 fork checkpoint。

终态必须不可变的原因：

- 防止已成功工具再次被 approve 执行，造成副作用重复。
- 防止 failed 后被误写成 running，掩盖真实失败原因。
- 防止 cancelled/reject 后又继续执行，违背人工决策。
- 保证审计日志可按单调状态转移复盘。

## HITL Boundary

HITL 状态映射：

| HITL 动作 | 状态转移 | 语义 |
| --- | --- | --- |
| `interrupt` | `running/planning -> waiting_user` | runtime 暂停，checkpoint 保存 `hitl` |
| `resume_approve` | `waiting_user -> running` | 用户批准当前等待点，继续执行 |
| `resume_respond` | `waiting_user -> running` | 用户补充信息，继续执行 |
| `resume_reject` | `waiting_user -> cancelled` | 用户拒绝本次等待任务，终止 run |
| `cancel` | `waiting_user -> cancelled` | 用户或系统取消等待任务 |

`waiting_user` 与 `failed` 的边界：

- `waiting_user` 是业务上“需要人类输入”的暂停，不代表工具或模型失败。
- `reject/cancel` 是人工或系统决策，不是模型错误，应落到 `cancelled`。
- `failed` 只承载异常：模型调用失败、工具不可恢复失败、checkpoint 写入失败、非法状态无法处理等。
- SSE 等待路径不应发 `error`，应发 `waiting_user`。
- SSE reject/cancel 路径不应发 `error`，应发 `done`，payload 的 `final_state` 为 `cancelled`。

## Integration Design

### 1. State Machine Module

建议新增中立模块：

```text
backend/platform/workflow/state_machine.py
```

职责：

- 定义 `WorkflowRunState`、`WorkflowRunEvent`、终态集合。
- 提供 `validate_transition(current_state, event) -> next_state`。
- 提供 `is_terminal(state)`。
- 不依赖 FastAPI、scene、RAG 或 session store。

### 2. Lifecycle Recorder

`GraphRunLifecycleRecorder` 应从“无约束事件追加器”升级为状态机消费者：

- `create_run()` 记录 `created`。
- 新增 `mark_planning()`、`mark_waiting_user()`、`mark_retrying()`、`mark_cancelled()`。
- 现有 `mark_running/mark_succeeded/mark_failed` 内部都走 transition validator。
- 当前内存 recorder 可继续存在；如果后续需要跨进程审计，再持久化到 `workflow_run_events` 或 `langgraph_runs`。

### 3. RuntimeGraphState

`RuntimeGraphStatus` 与 `GraphRunStatus` 收敛为同一组 `WorkflowRunState`。

checkpoint 中建议保存：

```json
{
  "status": "waiting_user",
  "run_id": "run-...",
  "request_id": "req-...",
  "hitl": {},
  "hitl_resume": null,
  "retry": {
    "attempt": 0,
    "max_attempts": 0,
    "last_error": null
  }
}
```

最小实现可以先只新增 `run_id` 和 `retry` 可选字段，不迁移历史 checkpoint。读取旧 checkpoint 时，缺失状态按当前兼容逻辑处理。

### 4. ChatGraphRuntime Write Points

建议写入位置：

- `invoke()`：`created -> planning -> running -> succeeded/failed`。如果当前普通图没有 planner 节点，可先 `created -> running`，但 lifecycle 仍保留 `planning` API。
- `create_hitl_wait()`：`created -> running -> waiting_user`，不要把等待写成 lifecycle `succeeded`。
- `resume_hitl()`：
  - 读取 checkpoint，验证当前状态必须是 `waiting_user`。
  - `approve/respond` 写 `running`，完成后写 `succeeded` 或 `failed`。
  - `reject` 写 `cancelled`，不执行工具。
- 工具调用边界：捕获 retryable error 时写 `retrying`，再由 `retry` 转回 `running`。

### 5. SSE/API Compatibility

不重写 `/chat` 对外协议，只新增可选字段：

- `state`: 当前 runtime state。
- `final_state`: `done/error` 中的最终状态。
- `run_id`: 可选，用于审计关联。
- `state_event`: 可选，描述本次状态转移，例如 `interrupt`、`success`、`resume_reject`。

事件兼容策略：

| SSE 事件 | 新增字段 |
| --- | --- |
| `start` | `state="created"` 或 `state="running"`，`run_id` |
| `waiting_user` | `state="waiting_user"`，`state_event="interrupt"` |
| `resume` | `state_event="resume_approve/resume_respond/resume_reject"` |
| `done` | `final_state="succeeded/cancelled"` |
| `error` | `final_state="failed"` |

### 6. Session Persistence Boundary

`sessions.status` 继续只表示聊天会话是否 `active/expired`，不扩展为 workflow 状态。

原因：

- 一个 session 可以有多次 run。
- session 过期不等于 run failed。
- workflow run 的真实恢复点在 LangGraph checkpoint；审计事件在 lifecycle/run event log。

### 7. LangGraph Checkpoint Relationship

分工建议：

| 存储 | 保存内容 | 用途 |
| --- | --- | --- |
| LangGraph checkpoint | 最新可恢复 graph state：`status/hitl/retry/messages/answer/retrieval_trace` | resume、恢复执行 |
| lifecycle event log | 每次状态转移：`run_id/thread_id/request_id/from/to/event/error/timestamp` | 审计、测试、可观测 |
| session store | 会话、消息、turn 和 citations 读模型 | 聊天历史与前端查询 |

checkpoint 是“当前事实”，event log 是“历史事实”。不能只依赖 checkpoint 复盘状态变化。

## Minimal Test Matrix

| 测试样本 | 目标路径 | 关键断言 |
| --- | --- | --- |
| normal success | `created -> planning -> running -> succeeded` | final_state 为 `succeeded`；done 不含 error；checkpoint status 为 `succeeded` |
| HITL approve | `running -> waiting_user -> running -> succeeded` | waiting_user 不是 failed；approve 后只执行一次 proposed tool；重复 resume 返回 409 |
| HITL reject | `running -> waiting_user -> cancelled` | 工具不执行；SSE `done.final_state=cancelled`；不是 error |
| tool failure retry | `running -> retrying -> running -> succeeded` 或 `running -> retrying -> failed` | retry count 增加；超过上限进入 failed |
| terminal resume denied | `succeeded/failed/cancelled + resume` | 返回 409；状态不改变；不会写入新的 running |

候选测试文件：

- `backend/tests/test_langgraph_runtime.py`：状态机 unit test、checkpoint 状态断言、终态保护。
- `backend/tests/test_generic_assistant_hitl.py`：HITL approve/reject/resume API 与 SSE 回归。
- `backend/tests/test_chat_api.py`：兼容性回归，确保普通 `/chat` 响应不被破坏。

候选命令：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_generic_assistant_hitl.py -q -c backend\tests\pytest.ini
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

## Acceptance Checklist

- [x] 指出当前 runtime 中已有 run 生命周期语义：lifecycle recorder、checkpoint status、HITL wait/resume、SSE mapper。
- [x] 指出当前缺口：无统一八状态 enum、无转移表、无 `retrying/cancelled/planning`、无通用终态保护。
- [x] 给出状态转移表，覆盖 `created/running/waiting_user/retrying/succeeded/failed/cancelled`，并补充 `planning`。
- [x] 说明 HITL interrupt/resume 映射：`interrupt -> waiting_user`，`approve/respond -> running`，`reject/cancel -> cancelled`。
- [x] 说明终态保护原因和防重复 resume 策略。
- [x] 列出最小测试样本：正常成功、HITL approve、HITL reject、工具失败重试、终态 resume 被拒绝。
- [x] 形成 JD 证明点。

## JD Proof Points

可用于 Agent Runtime 岗位讲解的证明点：

- 将一次 Agent graph run 从普通函数调用结果升级为可审计的 Workflow State Machine，明确区分 `created/planning/running/waiting_user/retrying/succeeded/failed/cancelled`。
- 基于 LangGraph checkpoint 保存可恢复状态，基于 lifecycle event log 保存状态转移历史，避免只靠 SSE `done/error` 推断任务状态。
- 为 HITL 定义 `waiting_user -> running/cancelled` 边界，证明等待人工输入、拒绝和取消不是模型失败。
- 通过终态保护防止已完成、已失败或已取消的 run 被重复 resume、retry 或重新写成 running，降低工具副作用重复执行风险。
- 为后续 Failure Recovery、长任务队列、Workflow Evaluation 和可观测性预留统一状态口径。

## Follow-up Implementation Plan

建议后续实现分四步：

1. 新增 `platform.workflow.state_machine`，先写纯 unit test 覆盖转移表和终态保护。
2. 扩展 `GraphRunLifecycleRecorder` 和 `RuntimeGraphState`，让 lifecycle 与 checkpoint 共享 `WorkflowRunState`。
3. 改造 `ChatGraphRuntime.create_hitl_wait()` 和 `resume_hitl()`，让 wait 写 `waiting_user`、reject 写 `cancelled`、终态 resume 返回 409。
4. 补 SSE/API 可选字段 `state/final_state/run_id/state_event`，跑 HITL 与 chat API 回归。

## Non-goals

- 不实现长期任务队列和分布式调度。
- 不扩展复杂审批 UI。
- 不迁移 Planner / Executor、Failure Recovery 或 Agentic RAG Subgraph。
- 不把历史所有会话迁移成 workflow run。
- 不把 `sessions.status` 扩展成 workflow 状态字段。



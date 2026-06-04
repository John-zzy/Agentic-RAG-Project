# PRD-20260602-01 Planner / Executor 最小任务编排设计与接入审计

## 0. 一页结论

本 PRD 的目标是把当前 `/chat` 主链升级为最小 Agent Runtime，而不是把 `AgenticRetriever` 的检索循环包装成顶层 ReAct。

核心调整：

- `/chat` 顶层由 LangGraph 组装为 Agent Runtime。
- 简单问题走顶层 ReAct 模式。
- 复杂问题走顶层 Plan 模式。
- `backend/platform/rag/` 保持 Modular RAG 能力底座。
- Native RAG、Agentic RAG 都应封装为 Agent 可调用工具。
- `Agentic RAG Subgraph` 是 README P0 后续补充项，不阻塞本次 Planner / Executor。
- Workflow State Machine 只治理 run 级状态，turn / step / tool call 状态保存在各自结构内。

推荐目标链路：

```text
POST /chat
  -> ActiveSceneChatService 解析 scene / session
  -> ChatGraphRuntime 启动 LangGraph run
  -> AgentRuntimeGraph
       -> load_context
       -> mode_selector
       -> react_loop 或 plan_workflow
       -> tool_executor
       -> final_synthesizer
  -> checkpoint / SSE / HITL / final response
```

当前满足度：

| 需求目标 | 满足度 |
| --- | --- |
| 走查当前 planning / execution 职责 | 部分满足 |
| 定义最小 PlanStep | 未满足 |
| 定义最小 PlanRun | 未满足 |
| 设计 Executor 执行规则 | 部分满足 |
| 明确 Planner / Executor 与 HITL、Workflow、SSE、checkpoint 边界 | 部分满足 |
| 写出最小验证路径 | 未满足 |

数量结论：

- 完全满足：0/6
- 部分满足：3/6
- 未满足：3/6
- 尚未完全满足：6/6

## 1. 需求边界

### 1.1 背景

README 当前已完成 `Human-in-the-Loop`、`Workflow State Machine` 和 LangGraph Runtime 骨架，并把 P0 下一项写为 `Planner / Executor`。下一项才是 `Agentic RAG Subgraph`。

这意味着今天的主任务不是迁移 `AgenticRetriever` while loop，也不是继续扩展 RAG 内部 trace，而是给 `/chat` 组装顶层 Agent Runtime：

- 能判断简单问题还是复杂任务。
- 能让 LLM 在顶层决定调用哪个工具。
- 能将 RAG 作为工具能力调用。
- 能将复杂任务拆成步骤并沉淀步骤结果。
- 能把失败、重试、等待用户、取消和成功都落到统一状态边界。

### 1.2 本次目标

- 走查当前 `AgenticRetriever`、LangGraph Runtime、scene tools、`/chat`、Workflow State Machine 的职责边界。
- 明确 RAG / Agentic RAG 是可插拔工具能力，不是顶层 Agent Runtime。
- 设计 `/chat` 顶层 ReAct / Plan 双模式。
- 定义最小 `ReActRun` / `ReActTurn`。
- 定义最小 `PlanRun` / `PlanStep`。
- 设计统一 `ToolExecutor` 和 `PlanExecutor`。
- 梳理 Workflow State Machine 全部状态的边界与用途。
- 明确 HITL、SSE、checkpoint 和 LangGraph 接入方式。
- 产出最小验证路径、测试样本、面试卡、八股口径和简历素材。

### 1.3 非目标

- 今天不实现完整多 Agent 协作。
- 今天不做复杂 Planner 提示词优化和自动反思链路。
- 今天不迁移完整 `AgenticRetriever` while loop 到 LangGraph Subgraph。
- 今天不把 `Agentic RAG Subgraph` 作为本 PRD 的前置依赖。
- 今天不新增复杂 UI。
- 今天不做生产级任务队列、分布式调度和权限系统。

## 2. 当前系统审计

### 2.1 当前职责分布

| 位置 | 当前职责 | 本 PRD 判断 |
| --- | --- | --- |
| `backend/platform/rag/` | Modular RAG 底座，包含 retrieval contracts、query rewrite、document retrieval、rerank、agentic orchestration | 应下沉为工具能力来源 |
| `backend/platform/rag/orchestration/agentic.py` | `AgenticRetriever.retrieve_with_trace()` 执行 RAG 内部多轮检索、rewrite、switch retrieval tool、ask_user | 是 Agentic RAG 工具内部流程，不是顶层 ReAct |
| `backend/platform/rag/contracts.py` | 定义 `RetrievalPlan`、`RetrievalResult`、`RetrievalContext`、`RetrievalTool` | 可作为 RAG tool output / trace 的底层结构 |
| `backend/application/runtime/service.py` | `ChatService` 串联检索、回答、HITL、SSE；`RetrievalExecutor` 执行 retriever | 当前 `/chat` 主入口，后续应把顶层编排交给 LangGraph Agent Runtime |
| `backend/application/runtime/graph_runtime.py` | 创建 graph run、写 checkpoint、处理 HITL wait/resume、保护终态 | 应升级为 ReAct / Plan graph 的承载层 |
| `backend/platform/workflow/state_machine.py` | 定义 run 级状态和合法转移 | 可复用，但必须清晰区分 run / step / turn / tool call |
| `backend/platform/workflow/langgraph/state.py` | checkpoint 保存 `messages/answer/retrieval_trace/status/hitl/retry_metadata` | 需要扩展 Agent Runtime 上下文 |
| `backend/scenes/base.py` | `SceneDefinition` 暴露 `build_retriever()`、`build_tools()`、候选 retrieval tools、复杂度策略 | 说明项目已经具备工具与 retriever 的双装配入口 |
| `backend/scenes/generic_assistant/definition.py` | generic scene 组装 docs-first `AgenticRetriever` 和结构化工具 | 后续应同时提供 RAG tool 与 scene tools 给 ToolExecutor |
| `backend/scenes/generic_assistant/hitl.py` | 生成 clarification / approval 等 HITL 等待计划 | 可复用为 Agent Runtime 的 HITL interrupt policy |
| `backend/scenes/ecommerce/definition.py` | 电商意图路由、retrieval tools、action tools | 可作为复杂 Plan / ReAct 调用多工具的演示场景 |

### 2.2 关键结论

当前项目里有三类能力，但它们还没有组成顶层 Agent Runtime：

1. RAG 能力已有：Hybrid Search、Agentic Retrieval、trace、引用、rerank 接入位。
2. Runtime 基础已有：LangGraph checkpoint、SSE、HITL、Workflow State Machine。
3. 工具边界已有：scene 可以提供 retriever、retrieval tools、structured tools。

缺口在顶层编排：

- `/chat` 没有先进入 `mode_selector`。
- 没有顶层 ReAct graph。
- 没有 Plan graph。
- 没有统一 `ToolExecutor`。
- RAG 还被当作主链，而不是 Agent 可调用工具。
- `AgenticRetriever` 的 rounds 不能代表顶层 ReAct turns。

## 3. 架构边界：RAG 是工具，Agent Runtime 是顶层

### 3.1 RAG 分层定位

`backend/platform/rag/` 继续保持 Modular RAG 架构：

```text
platform.rag
  -> pre_retrieval       query rewrite
  -> retrieval.documents document retrieval / hybrid search
  -> post_retrieval      rerank
  -> orchestration       AgenticRetriever
  -> contracts           RetrievalTool / RetrievalResult / RetrievalContext
```

上层可按需封装为工具：

| 工具 | 来源 | 用途 |
| --- | --- | --- |
| `native_rag_search` | `DocumentRetrievalService` | 单次文档召回 / Hybrid Search |
| `agentic_rag_search` | `AgenticRetriever.retrieve_with_trace()` | RAG 内部多轮 rewrite / switch retrieval tool / sufficiency judge |
| `agentic_rag_subgraph` | 后续 LangGraph Subgraph | 迁移 while loop 后的可复用 RAG 子图 |

工具输出建议统一为 `RagObservation`：

```python
class RagObservation(BaseModel):
    tool_name: str
    query: str
    success: bool
    documents: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    retrieval_trace: dict[str, Any] = {}
    final_decision: str | None = None
    follow_up_question: str | None = None
    result_summary: str | None = None
    error: str | None = None
```

原则：

- RAG 工具可以返回证据、引用、trace 和检索决策。
- 顶层 Agent 决定是否继续调用别的工具、等待用户或生成最终回答。
- RAG 内部 trace 可嵌套在 ReAct observation 或 PlanStep output 中。
- 不把 RAG 内部 round 直接升级为顶层 ReActTurn。

### 3.2 `/chat` 目标顶层结构

```text
ChatService
  -> ChatGraphRuntime.invoke_agent()
       -> AgentRuntimeGraph
            -> load_context_node
            -> mode_selector_node
            -> react_loop_node
            -> planner_node
            -> plan_executor_node
            -> tool_executor_node
            -> hitl_interrupt_node
            -> final_synthesizer_node
```

建议新增或调整模块：

| 模块 | 职责 |
| --- | --- |
| `backend/platform/agent_runtime/contracts.py` | `AgentRun`、`ReActRun`、`PlanRun`、`ToolCall` 等中立合同 |
| `backend/platform/agent_runtime/mode_selector.py` | 选择 `react` 或 `plan` |
| `backend/platform/agent_runtime/tool_executor.py` | `ToolExecutor`，统一调用 RAG tools 与 scene structured tools |
| `backend/platform/agent_runtime/react/runtime.py` | 顶层 ReAct loop，不等同于 `AgenticRetriever` |
| `backend/platform/agent_runtime/plan/planner.py` | 最小 Planner，生成 `PlanRun` |
| `backend/platform/agent_runtime/plan/executor.py` | `PlanExecutor`，按依赖执行 `PlanStep` |
| `backend/application/runtime/graph_runtime.py` | 组装 LangGraph nodes、checkpoint、HITL resume |

命名也可以继续使用 `backend/platform/planning/`，但语义上应是 Agent Runtime 编排层，而不是 RAG adapter 层。

## 4. ReAct 模式：简单问题的顶层行动循环

### 4.1 适用场景

ReAct 用于简单或中等复杂度问题：

- 单一目标。
- 不需要提前展示完整计划。
- 下一步依赖上一轮工具观察。
- 最多有限轮工具调用。
- 可以调用 RAG 工具，也可以调用 scene structured tool。

典型例子：

- “根据知识库解释 Planner / Executor 是什么。”
- “查一下这个商品有没有库存。”
- “如果当前证据不够，再换一个工具补充。”

### 4.2 ReActRun / ReActTurn

```python
ReActTurnStatus = Literal[
    "running",
    "succeeded",
    "retrying",
    "failed",
    "waiting_user",
    "cancelled",
]

ReActAction = Literal[
    "tool_call",
    "ask_user",
    "final_answer",
]

class ReActTurn(BaseModel):
    turn_id: str
    round_index: int
    goal: str
    action: ReActAction
    tool_name: str | None = None
    input: dict[str, Any] = {}
    status: ReActTurnStatus = "running"
    observation_summary: str | None = None
    result_summary: str | None = None
    output: dict[str, Any] = {}
    error: str | None = None
    retry_count: int = 0
    interrupt_id: str | None = None

class ReActRun(BaseModel):
    react_run_id: str
    session_id: str
    request_id: str
    mode: Literal["react"] = "react"
    user_goal: str
    workflow_status: WorkflowRunState = "created"
    turns: list[ReActTurn] = []
    current_turn_id: str | None = None
    max_turns: int = 3
    final_answer: str | None = None
    result_summary: str | None = None
    error: str | None = None
```

说明：

- `rewrite`、`switch_tool` 不建议作为顶层 ReAct action 的必选枚举；它们可以作为 `agentic_rag_search` 工具内部 `retrieval_trace`。
- 顶层 ReAct 只关心调用哪个工具、观察到什么、是否继续、是否等待用户、是否最终回答。
- 不保存隐藏思维链，只保存可审计摘要。

### 4.3 ReAct 执行规则

1. workflow run 从 `created` 进入 `running`。
2. LLM 根据上下文选择一个 action。
3. `tool_call` 必须经过 `ToolExecutor` 和 scene 工具白名单校验。
4. 工具可以是 `native_rag_search`、`agentic_rag_search`、订单查询、商品查询、审批前置工具等。
5. 工具结果写入 `observation_summary/output/result_summary`。
6. 需要用户澄清时 run 与 turn 进入 `waiting_user`。
7. 可重试错误进入 `retrying`，retry 后回到 `running`。
8. 达到 `max_turns` 后必须 `final_answer`、`ask_user` 或 `failed`，不能无限自旋。
9. 用户 reject / cancel 进入 `cancelled`。
10. 最终回答由顶层 `final_synthesizer` 生成。

## 5. Plan 模式：复杂问题的显式步骤编排

### 5.1 适用场景

Plan 用于复杂任务：

- 多目标。
- 多工具组合。
- 多知识源或业务系统协作。
- 需要步骤结果沉淀。
- 中间步骤可能失败、重试、等待人工澄清或审批。
- 用户显式要求“先规划再执行”。

### 5.2 PlanStep / PlanRun

```python
PlanStepStatus = Literal[
    "pending",
    "running",
    "succeeded",
    "retrying",
    "failed",
    "waiting_user",
    "skipped",
    "cancelled",
]

class PlanStep(BaseModel):
    step_id: str
    goal: str
    tool_name: str
    input: dict[str, Any]
    depends_on: list[str] = []
    status: PlanStepStatus = "pending"
    result_summary: str | None = None
    error: str | None = None
    output: dict[str, Any] = {}
    retry_count: int = 0
    max_retries: int = 1
    requires_approval: bool = False
    interrupt_id: str | None = None

class PlanRun(BaseModel):
    plan_run_id: str
    session_id: str
    request_id: str
    mode: Literal["plan"] = "plan"
    user_goal: str
    workflow_status: WorkflowRunState = "created"
    steps: list[PlanStep] = []
    current_step_id: str | None = None
    final_answer: str | None = None
    result_summary: str | None = None
    error: str | None = None
```

### 5.3 Planner 职责

Planner 只生成计划，不执行工具：

- 输入：用户目标、scene、mounted knowledge sources、可用工具、HITL policy。
- 输出：1 到 3 个 `PlanStep`。
- `tool_name` 必须来自 ToolExecutor 暴露的 allowlist。
- `depends_on` 必须是无环依赖。
- Planner 不直接调用工具。
- Planner 不绕过 scene policy。

### 5.4 PlanExecutor 执行规则

1. workflow run 从 `created` 进入 `planning`。
2. Planner 生成并校验 `PlanRun.steps`。
3. workflow run 从 `planning` 进入 `running`。
4. Executor 只执行依赖全部成功的 `pending` step。
5. 每个 step 调用都经 `ToolExecutor`。
6. 工具成功后 step 进入 `succeeded`，写入 `output/result_summary`。
7. 可重试错误使 step/run 进入 `retrying`，retry 后回到 `running`。
8. 重试耗尽或不可恢复错误使 step/run 进入 `failed`。
9. 需要人工澄清或审批时 step/run 进入 `waiting_user`。
10. 用户 `respond` 后更新 step input 或追加 follow-up step。
11. 用户 `approve` 后执行 proposed tool call。
12. 用户 `reject` 后 step/run 进入 `cancelled`，不得执行副作用工具。
13. 必要 step 全部成功后，由 `final_synthesizer` 汇总最终回答，run 进入 `succeeded`。

## 6. ToolExecutor 设计

`ToolExecutor` 是 ReAct 和 Plan 的共同工具调用边界。

```text
ToolExecutor
  -> validate tool allowlist
  -> validate input schema
  -> call selected tool
  -> normalize ToolObservation
  -> apply HITL policy when needed
  -> return structured output
```

工具来源：

| 来源 | 示例 |
| --- | --- |
| RAG tools | `native_rag_search`、`agentic_rag_search` |
| scene structured tools | 订单查询、商品查询、库存查询、写操作测试工具 |
| internal tools | `final_synthesizer`、`ask_user` |
| future subgraph tools | `agentic_rag_subgraph`、business handoff subgraph |

`ToolObservation` 建议结构：

```python
class ToolObservation(BaseModel):
    tool_name: str
    success: bool
    output: dict[str, Any] = {}
    result_summary: str | None = None
    citations: list[dict[str, Any]] = []
    trace: dict[str, Any] = {}
    requires_user: bool = False
    interrupt: dict[str, Any] | None = None
    retryable: bool = False
    error: str | None = None
```

## 7. LangGraph 接入设计

### 7.1 ReAct Graph

```text
START
  -> load_context
  -> select_react_action
  -> execute_tool
  -> observe
  -> should_continue
       -> select_react_action
       -> hitl_wait
       -> final_synthesizer
       -> fail
  -> END
```

### 7.2 Plan Graph

```text
START
  -> load_context
  -> planner
  -> validate_plan
  -> execute_next_step
  -> should_continue
       -> execute_next_step
       -> hitl_wait
       -> retry
       -> final_synthesizer
       -> fail
  -> END
```

### 7.3 RuntimeGraphState 扩展

建议增加：

```python
agent_mode: NotRequired[Literal["react", "plan"] | None]
react_run: NotRequired[dict[str, Any] | None]
plan_run: NotRequired[dict[str, Any] | None]
current_turn_id: NotRequired[str | None]
current_step_id: NotRequired[str | None]
current_tool_call: NotRequired[dict[str, Any] | None]
```

为了兼容旧命名，OpenSpec 中如果继续使用 `plan_mode`，需要明确它表示 orchestration mode，而不是只有 Plan 模式。

## 8. Workflow State Machine 状态清理

当前 run 级状态保留 8 个即可：

| 状态 | 用途 | 可用于 ReAct | 可用于 Plan |
| --- | --- | --- | --- |
| `created` | run 已创建，尚未执行 | 是 | 是 |
| `planning` | 正在生成或校验显式计划 | 否 | 是 |
| `running` | 正在执行 ReAct turn、Plan step、tool call 或 final synthesis | 是 | 是 |
| `waiting_user` | HITL interrupt，等待澄清或审批 | 是 | 是 |
| `retrying` | 可重试错误后的重试准备态 | 是 | 是 |
| `succeeded` | 终态，最终回答完成 | 是 | 是 |
| `failed` | 终态，不可恢复错误或重试耗尽 | 是 | 是 |
| `cancelled` | 终态，用户 reject/cancel | 是 | 是 |

边界规则：

- Workflow 状态只管一次 `/chat` run。
- `sessions.status` 只管聊天会话生命周期，不能替代 workflow 状态。
- `PlanStep.status` 只管 step。
- `ReActTurn.status` 只管 turn。
- `ToolObservation.success/retryable` 只管单次工具调用结果。
- HITL `pending_action` 只管等待用户做什么。
- 不新增 `agent_status`、`planner_status`、`executor_status`、`rag_status` 等重复状态。

状态转移建议：

| 场景 | Run 转移 |
| --- | --- |
| ReAct 开始 | `created -> running` |
| Plan 开始规划 | `created -> planning` |
| Plan 规划完成 | `planning -> running` |
| 工具成功但还需继续 | 保持 `running` |
| 最终汇总成功 | `running -> succeeded` |
| HITL 等待 | `running/planning -> waiting_user` |
| 用户 respond / approve | `waiting_user -> running` |
| 用户 reject | `waiting_user -> cancelled` |
| 可重试错误 | `running -> retrying` |
| 开始重试 | `retrying -> running` |
| 重试耗尽 / 不可恢复错误 | `running/retrying/planning -> failed` |
| 用户取消 | 非终态 -> `cancelled` |

## 9. HITL、SSE、Checkpoint

### 9.1 HITL

ReAct：

- `ask_user` 进入 `waiting_user`。
- HITL metadata 包含 `mode=react`、`react_run_id`、`current_turn_id`。
- resume respond 后恢复当前 turn。
- reject 后 run 进入 `cancelled`。

Plan：

- clarification / approval 都绑定 `current_step_id`。
- HITL metadata 包含 `mode=plan`、`plan_run_id`、`current_step_id`。
- approve 后执行 proposed tool call。
- respond 后更新 step input 或追加 follow-up step。
- reject 后 run 进入 `cancelled`。

### 9.2 SSE

保持现有事件名兼容：

```text
start / history / tool / chunk / waiting_user / resume / done / error
```

用 `tool` payload 标识阶段：

```json
{
  "stage": "react_turn",
  "react_run_id": "react_xxx",
  "turn_id": "turn_1",
  "turn_status": "succeeded",
  "tool_name": "agentic_rag_search"
}
```

```json
{
  "stage": "plan_step",
  "plan_run_id": "plan_xxx",
  "step_id": "step_1",
  "step_status": "succeeded",
  "tool_name": "native_rag_search"
}
```

### 9.3 Checkpoint

写入时机：

- mode selection 后写入 `agent_mode`。
- ReAct 每个 turn 完成后写入 `react_run`。
- Plan 生成后写入完整 `plan_run.steps`。
- step/turn 进入 `waiting_user` 前必须写入当前 id 和 HITL payload。
- retry 前必须写入 retry metadata 和当前执行点。
- resume 时先消费 HITL，再执行副作用工具，避免重复执行。

## 10. 最小验证路径

### 10.1 ReAct 简单成功

用户：

> 根据知识库解释 Planner / Executor 是什么。

预期：

- ModeSelector 选择 `react`。
- 顶层 ReAct 调用 `agentic_rag_search` 或 `native_rag_search`。
- RAG 内部 trace 写入 turn observation。
- final_synthesizer 生成回答。
- run `created -> running -> succeeded`。

### 10.2 ReAct 多工具

用户：

> 查一下这个商品有没有库存。

预期：

- ReAct 第一轮可调用商品搜索工具。
- 第二轮根据 observation 调用库存工具。
- turns 保存两个顶层 tool call。
- run 成功。

### 10.3 Plan 多步成功

用户：

> 对比当前知识库里的 Planner/Executor 设计和电商售后场景，说明哪些步骤要查文档、哪些步骤要查订单或商品工具，最后给出落地建议。

Plan：

| step_id | goal | tool_name | depends_on |
| --- | --- | --- | --- |
| `step_docs` | 检索 Planner / Executor 设计资料 | `agentic_rag_search` | `[]` |
| `step_ecommerce` | 检索售后相关订单或商品信息 | `order_semantic_search` 或 `product_semantic_search` | `[]` |
| `step_synthesis` | 汇总前两步结果 | `final_synthesizer` | `["step_docs", "step_ecommerce"]` |

预期：

- run `created -> planning -> running -> succeeded`。
- step 1、2 保存结构化 output 和 result_summary。
- step 3 汇总最终回答。

### 10.4 Plan 工具失败重试

预期：

- step 工具第一次失败且 retryable。
- step/run 进入 `retrying`。
- retry 后回到 `running`。
- 成功后继续后续 step。

### 10.5 Plan HITL 中断

预期：

- step 缺少订单号或需要审批。
- step/run 进入 `waiting_user`。
- checkpoint 保存 `plan_run/current_step_id/hitl`。
- respond / approve 后恢复。
- reject 后 `cancelled`。

## 11. 测试样本建议

| 样本 | 覆盖点 |
| --- | --- |
| ReAct 单工具成功 | 顶层 ReActTurn 调用 RAG tool，run succeeded |
| ReAct 多工具成功 | 两个顶层 tool calls，不把 RAG 内部 rounds 当 turn |
| ReAct ask_user | turn/run waiting_user，resume respond 后继续 |
| Plan 单步成功 | 生成一个 step 并执行 |
| Plan 多步依赖 | 按 `depends_on` 执行 |
| Plan 工具失败重试 | retrying -> running -> succeeded |
| Plan 工具失败耗尽 | failed，记录 error |
| Plan HITL 中断 | waiting_user -> running |
| 人工拒绝 | waiting_user -> cancelled，副作用工具不执行 |
| SSE 兼容 | `tool(stage=react_turn/plan_step)` |
| legacy checkpoint | 缺少新字段仍可读取 |

候选命令：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_react_plan_runtime.py -q -c backend\tests\pytest.ini
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agentic_retrieval.py backend\tests\test_langgraph_runtime.py backend\tests\test_generic_assistant_hitl.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

本次仅调整方案与 OpenSpec，不运行 pytest。

## 12. 面试表达素材

### 12.1 一句话

> 我把 `/chat` 从一次增强 RAG 调用升级为 LangGraph Agent Runtime：简单问题走顶层 ReAct，复杂任务走 Plan；RAG 和 Agentic RAG 被封装成工具，由外层 Agent 自主调用，Workflow State Machine 统一治理运行状态，HITL 和 checkpoint 负责中断恢复。

### 12.2 八股口径

- RAG 是能力底座，不是 Agent Runtime 本身。
- Agentic RAG 的 while loop 是 RAG 工具内部策略，不等于顶层 ReAct。
- 顶层 ReAct 记录 Agent 对工具的行动和观察。
- Plan 记录复杂任务的步骤、依赖、结果和错误。
- ToolExecutor 统一执行 RAG tools、scene tools 和 internal tools。
- Workflow State Machine 只管 run 级状态。
- HITL 是 interrupt/resume，不是失败。
- `Agentic RAG Subgraph` 是后续把 RAG 内部 while loop LangGraph 化的补充项。

### 12.3 简历素材

> 设计 `/chat` 顶层 Agent Runtime：基于 LangGraph 组装 ReAct / Plan 双模式，简单问题由 ReAct 自主调用工具，复杂任务由 Planner 生成 PlanStep 并由 Executor 按依赖执行；将 Modular RAG 与 Agentic RAG 封装为 Agent 可调用工具，统一接入 ToolExecutor、Workflow State Machine、HITL interrupt/resume、checkpoint 和 SSE，实现复杂任务的步骤审计、失败重试、中断恢复和最终汇总。

## 13. 实施前验收清单

- [x] 明确 `/chat` 顶层是 Agent Runtime，不是 RAG 主链。
- [x] 明确 RAG / Agentic RAG 应封装为工具。
- [x] 明确 `Agentic RAG Subgraph` 是后续补充项。
- [x] 定义 ReAct / Plan 双模式边界。
- [x] 定义 `ReActRun` / `ReActTurn`。
- [x] 定义 `PlanRun` / `PlanStep`。
- [x] 定义 `ToolExecutor` 与 `PlanExecutor`。
- [x] 梳理 Workflow State Machine 8 个状态用途。
- [x] 说明 HITL、SSE、checkpoint 接入边界。
- [x] 列出最小验证路径和测试样本。
- [x] 产出面试卡、八股口径和简历素材。

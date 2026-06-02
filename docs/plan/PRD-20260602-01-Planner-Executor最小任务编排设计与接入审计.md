# PRD-20260602-01 Planner / Executor 最小任务编排设计与接入审计

## 0. 一页结论

本 PRD 的目标不是立刻实现完整多 Agent 平台，而是给现有 `ai-rag-project` 补齐最小 Agent Runtime 编排语义：简单问题走 ReAct，复杂问题走 Plan。

推荐双模式：

| 模式 | 适用问题 | 核心对象 | 运行方式 | 当前项目状态 |
| --- | --- | --- | --- | --- |
| ReAct | 简单或中等复杂度问题，下一步依赖上一轮观察结果 | `ReActRun` / `ReActTurn` | 边观察边行动，最多执行有限轮工具调用或检索 | 行为已有雏形，结构未显式沉淀 |
| Plan | 复杂问题、多工具组合、需要步骤结果沉淀或中途恢复 | `PlanRun` / `PlanStep` | 先生成计划，再按依赖顺序执行步骤并汇总 | 结构、执行器、checkpoint、SSE 均未满足 |

当前代码层面结论：

- 完全满足 PRD 需求目标：0/6
- 部分满足：3/6
- 未满足：3/6
- 尚未完全满足：6/6

本阶段建议先做设计和审计，不直接迁移 `AgenticRetriever` while loop 到 LangGraph Subgraph，也不新增复杂 UI、生产级任务队列、分布式调度或权限系统。

## 1. 需求边界

### 1.1 背景

README 当前已将 `Human-in-the-Loop` 与 `Workflow State Machine` 标为完成，并将 `Planner / Executor` 放在 P0 队列。现有系统已经具备 RAG 主链、LangGraph Runtime、HITL 和工作流状态机，但 Agent 仍主要围绕一次检索问答链路执行。

面试或架构追问通常会集中在：

- 复杂需求如何拆成可执行步骤。
- 每一步如何决定调用哪个工具。
- 每一步结果如何沉淀并参与最终回答。
- 工具失败、重试、人工确认、中断恢复如何表达。
- Agent Runtime 如何区别于增强版 RAG。

### 1.2 本次目标

- 走查当前 `AgenticRetriever`、LangGraph Runtime、scene tool routing 和 `/chat` 入口，确认哪些逻辑已经承担 planning / execution 职责。
- 定义 ReAct 模式的最小运行结构。
- 定义 Plan 模式的最小 `PlanStep` 与 `PlanRun` 结构。
- 明确简单问题使用 ReAct，复杂问题使用 Plan。
- 设计 Plan Executor 的最小执行规则。
- 明确 Planner / Executor 与 HITL、Workflow State Machine、SSE event、checkpoint 的接入边界。
- 写出最小验证路径、测试样本、面试卡、八股口径和简历素材。

### 1.3 非目标

- 今天不实现完整多 Agent 协作。
- 今天不做复杂 Planner 提示词优化和自动反思链路。
- 今天不迁移完整 `AgenticRetriever` while 循环到 LangGraph Subgraph。
- 今天不新增复杂 UI，只关注后端任务编排语义和验证路径。
- 今天不做生产级任务队列、分布式调度和权限系统。

## 2. 当前系统审计

### 2.1 当前职责分布

| 位置 | 当前职责 | 与 ReAct / Plan 的关系 |
| --- | --- | --- |
| `backend/platform/rag/orchestration/agentic.py` | `AgenticRetriever.retrieve_with_trace()` 执行多轮检索，根据 `finish/rewrite/switch_tool/ask_user` 决策继续或退出 | 已具备 ReAct 风格行为，但没有显式 `ReActRun/ReActTurn` |
| `backend/platform/rag/contracts.py` | 定义 `RetrievalPlan`、`RetrievalResult`、`RetrievalContext` | 可作为 ReAct 轮次或 Plan 检索 step 的底层结构，不能替代跨工具计划 |
| `backend/application/runtime/service.py` | `RetrievalExecutor` 执行 retriever，`ChatService` 串联检索、回答、HITL、SSE | 是当前执行入口；新任务执行器建议命名为 `PlanExecutor`，避免与 `RetrievalExecutor` 混淆 |
| `backend/application/runtime/graph_runtime.py` | 创建 graph run、写 checkpoint、处理 HITL wait/resume、保护终态 | 是 `ReActRun/PlanRun` 可恢复状态的主要接入口 |
| `backend/platform/workflow/state_machine.py` | 定义 `created/planning/running/waiting_user/retrying/succeeded/failed/cancelled` 状态与转移 | 状态机已满足 run 级基础，但缺 step/turn 映射 |
| `backend/platform/workflow/langgraph/state.py` | checkpoint 保存 `messages/answer/retrieval_trace/status/hitl/retry_metadata` | 需要扩展 `react_run/plan_run/current_turn_id/current_step_id/plan_mode` |
| `backend/platform/workflow/langgraph/lifecycle.py` | 提供 lifecycle recorder，包含 `mark_planning()` | Plan 模式可直接接入；普通 `/chat` 当前多数跳过 `planning` |
| `backend/application/runtime/stream_events.py` | 映射 `start/history/tool/chunk/waiting_user/resume/done/error` | 可先复用 `tool` 事件承载 turn/step payload |
| `backend/scenes/generic_assistant/definition.py` | 定义默认工具、候选 retrieval tools、复杂度估算 | 模式选择和 Planner 应复用 scene 工具白名单 |
| `backend/scenes/ecommerce/definition.py` | 根据意图和上一轮结果切换商品、库存、详情、评价、订单工具 | 已接近 ReAct follow-up routing，可作为 Plan 工具选择素材 |
| `backend/scenes/generic_assistant/hitl.py` | `GenericAssistantHitlPlanner` 判断 clarification 等待点 | 是 HITL 等待规划器，不是复杂任务 Planner；新 Planner 应复用等待协议 |

### 2.2 当前结论

现有项目已经具备三类基础：

- ReAct 雏形：`AgenticRetriever` 的多轮 while loop 能根据观察结果继续 rewrite、switch tool、ask user 或 finish。
- 工具与场景边界：scene definition 已控制候选工具和知识源，不需要让 runtime 绕过 scene 自由选工具。
- 状态与恢复基础：Workflow State Machine、HITL、checkpoint 和 SSE 都已有接入口。

核心缺口：

- 缺少显式 `ReActRun/ReActTurn`，当前 ReAct 行为只能通过 retrieval trace 间接解释。
- 缺少显式 `PlanStep/PlanRun`，无法表达复杂任务拆解、步骤依赖和步骤级结果沉淀。
- 缺少 `PlanExecutor`，无法按依赖顺序执行 step，也无法做 step 级 retry / failed / waiting_user。
- checkpoint 与 SSE 还没有 turn/step 级 payload。

## 3. 目标运行时总览

### 3.1 模式选择原则

`/chat` 仍保持统一入口，但运行时先选择执行模式：

1. 简单或中等复杂度问题进入 ReAct。
2. 复杂、多目标、多工具、需要步骤沉淀的问题进入 Plan。
3. 用户显式要求“先列计划、分步骤执行”时进入 Plan。
4. scene 的 `infer_complexity()` 返回 `complex` 时优先进入 Plan。
5. 命中 action tool、审批工具或多个知识源协作时优先进入 Plan。

### 3.2 推荐运行流程

```text
POST /chat
  -> ChatService 读取 session、scene、mounted knowledge sources
  -> ModeSelector 选择 react 或 plan
  -> ReAct 模式：AgenticRetriever / ReAct adapter 执行有限轮 observe-act
  -> Plan 模式：Planner 生成 PlanRun，PlanExecutor 按依赖执行 PlanStep
  -> GraphRuntime 写 lifecycle、checkpoint、HITL 状态
  -> SSE 输出 tool/chunk/waiting_user/done/error
  -> final answer 汇总返回
```

### 3.3 模块边界

| 模块 | 建议职责 |
| --- | --- |
| `backend/platform/planning/contracts.py` | 定义 `ReActTurn`、`ReActRun`、`PlanStep`、`PlanRun`、状态类型和协议 |
| `backend/platform/planning/react.py` | 将现有 `AgenticRetriever` 轮次映射为 `ReActRun/ReActTurn`，后续可承载 `ReActExecutor` |
| `backend/platform/planning/planner.py` | 最小 Planner：根据用户目标、scene policy 和候选工具生成 1 到 3 个 step |
| `backend/platform/planning/executor.py` | `PlanExecutor`：校验依赖、调用工具、沉淀结果、处理 retry/HITL |
| `backend/application/runtime/service.py` | 选择 ReAct 或 Plan，组装 `/chat` 响应，保持 API 兼容 |
| `backend/application/runtime/graph_runtime.py` | 写 checkpoint、lifecycle、HITL wait/resume，不生成业务计划 |
| `backend/scenes/*/definition.py` | 暴露可规划工具、工具说明、复杂度策略和审批策略 |

边界原则：

- `platform.planning` 只依赖中立工具协议，不引入 scene 业务代码。
- scene 决定工具白名单、工具描述和业务 policy。
- Planner 不直接调用工具。
- Executor 不重新发明业务 policy。
- Tool 只返回结构化结果，不决定跨步骤调度。
- Workflow State Machine 管 run 级状态，不替代 turn/step 状态。
- HITL 是中断恢复机制，不是失败状态。

## 4. ReAct 模式：简单问题的边观察边行动

### 4.1 适用场景

ReAct 用于简单或中等复杂度问题：

- 用户只问一个目标明确的问题。
- 最多需要少量检索或一次工具切换。
- 下一步依赖上一轮工具观察结果，不需要提前展示完整计划。
- 任务不需要跨多个步骤沉淀中间产物。

典型例子：

- “根据知识库解释 Planner / Executor 是什么。”
- “查一下这个商品有没有库存。”
- “如果当前结果不够，再换一个检索工具补充。”

### 4.2 当前可复用基础

- `AgenticRetriever.retrieve_with_trace()` 已经有 while loop。
- `SufficiencyDecision.next_action` 已表达 `finish/rewrite/switch_tool/ask_user`。
- `RetrievalTraceRound` 已能对外展示检索轮次。
- `GenericAssistantHitlPlanner` 可承接 `ask_user` clarification。
- `/chat` SSE 已有 `tool` 和 `waiting_user` 事件。

### 4.3 ReActTurn 最小结构

ReAct 不提前生成完整步骤列表，而是保存有上限的观察和行动轨迹。注意：只保存可审计摘要，不保存模型隐藏思维链。

```python
ReActTurnStatus = Literal[
    "running",
    "observed",
    "succeeded",
    "retrying",
    "failed",
    "waiting_user",
    "cancelled",
]

ReActAction = Literal[
    "tool_call",
    "rewrite",
    "switch_tool",
    "ask_user",
    "finish",
]

class ReActTurn(BaseModel):
    turn_id: str
    round_index: int
    goal: str
    action: ReActAction
    tool_name: str | None = None
    input: dict[str, Any] = {}
    status: ReActTurnStatus = "running"
    decision_summary: str | None = None
    observation_summary: str | None = None
    result_summary: str | None = None
    error: str | None = None

    rewritten_query: str | None = None
    next_action: str | None = None
    is_sufficient: bool | None = None
    confidence: float | None = None
    output: dict[str, Any] = {}
    retry_count: int = 0
    interrupt_id: str | None = None
```

字段说明：

- `decision_summary`：记录为什么选择某个动作或工具，只保存摘要。
- `observation_summary`：记录工具结果、检索命中数量、关键证据摘要。
- `next_action/is_sufficient/confidence`：可映射当前 `SufficiencyDecision`。
- `rewritten_query`：可映射 query rewrite 结果。
- `interrupt_id`：用于 `ask_user` 进入 HITL 后恢复当前轮。

### 4.4 ReActRun 最小结构

```python
class ReActRun(BaseModel):
    react_run_id: str
    session_id: str
    request_id: str
    mode: Literal["react"] = "react"
    user_goal: str
    status: WorkflowRunState = "created"
    turns: list[ReActTurn] = []
    current_turn_id: str | None = None
    max_turns: int = 3
    final_decision: str | None = None
    final_answer: str | None = None
    result_summary: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = {}
```

### 4.5 ReAct 执行规则

1. run 创建后进入 `running`，不需要进入 `planning`。
2. 每轮只能选择 `tool_call/rewrite/switch_tool/ask_user/finish` 之一。
3. `tool_call` 和 `switch_tool` 必须经过 scene 候选工具白名单校验。
4. `rewrite` 只产生下一轮输入，不直接生成最终回答。
5. 工具成功后写入 `observation_summary/output/result_summary`。
6. `finish` 且 `is_sufficient=True` 时 run 进入 `succeeded`。
7. `ask_user` 时当前 turn 和 run 进入 `waiting_user`，checkpoint 保存 `react_run/current_turn_id/hitl`。
8. 工具可重试失败时 turn/run 进入 `retrying`，retry 后回到 `running`。
9. 达到 `max_turns` 后不能继续自旋；需要转 `ask_user` 或 `failed`。
10. 用户 reject 时 run 进入 `cancelled`，不是 `failed`。

### 4.6 ReAct 与现有结构映射

| 现有结构 | ReAct 字段 | 说明 |
| --- | --- | --- |
| `RetrievalRound.plan.round_index` | `round_index` | 检索轮次序号 |
| `RetrievalRound.plan.active_query` | `input.query` | 当前轮查询 |
| `RetrievalRound.result.tool_name` | `tool_name` | 当前轮调用工具 |
| `RetrievalRound.result.success/error` | `status/error` | 工具结果状态 |
| `RetrievalRound.decision.next_action` | `action/next_action` | 下一步动作 |
| `RetrievalRound.decision.reason` | `decision_summary` | 决策摘要 |
| `RetrievalRound.decision.is_sufficient` | `is_sufficient` | 证据是否足够 |
| `RetrievalRound.rewrite.query` | `rewritten_query` | 改写后的查询 |
| `RetrievalTraceRound.document_count/result_count` | `observation_summary` | 观察摘要素材 |

## 5. Plan 模式：复杂问题的显式步骤编排

### 5.1 适用场景

Plan 用于复杂任务：

- 用户目标包含多个子任务。
- 需要多个工具或多个知识源组合。
- 需要步骤级结果沉淀和最终综合。
- 中间步骤可能失败、重试、等待人工补充或审批。
- 用户显式要求“先规划再执行”。

典型例子：

- “对比当前知识库里的 Planner/Executor 设计和电商售后场景，说明哪些步骤要查文档、哪些步骤要查订单或商品工具，最后给落地建议。”
- “帮我排查一个售后问题，先查订单，再查商品库存，再生成处理建议。”
- “先列一个迁移方案，再按步骤验证每个模块是否满足。”

### 5.2 Planner 职责

Planner 只负责生成计划，不负责执行工具：

- 输入：用户目标、session、scene policy、候选工具、知识源、HITL policy。
- 输出：`PlanRun` 和 1 到 3 个 `PlanStep`。
- 每个 step 必须有明确 goal、tool_name、input、depends_on。
- Planner 必须保证 `tool_name` 来自 scene 允许范围。
- Planner 不直接调用工具，也不绕过 Executor。

### 5.3 PlanStep 最小结构

PRD 要求字段包括 `step_id`、`goal`、`tool_name`、`input`、`depends_on`、`status`、`result_summary`、`error`。为支持 retry、HITL 和恢复，建议补充少量审计字段。

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
    started_at: str | None = None
    finished_at: str | None = None
```

### 5.4 PlanRun 最小结构

```python
class PlanRun(BaseModel):
    plan_run_id: str
    session_id: str
    request_id: str
    mode: Literal["plan"] = "plan"
    user_goal: str
    status: WorkflowRunState = "created"
    steps: list[PlanStep] = []
    current_step_id: str | None = None
    final_answer: str | None = None
    result_summary: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = {}
```

状态关系：

- `created`：已创建 run，还未生成计划。
- `planning`：Planner 正在生成或校验步骤。
- `running`：PlanExecutor 正在执行步骤。
- `waiting_user`：某个 step 需要澄清或审批。
- `retrying`：当前 step 工具错误可重试。
- `succeeded`：必要 step 全部成功并生成最终回答。
- `failed`：不可恢复错误或重试耗尽。
- `cancelled`：用户 reject/cancel 或依赖步骤取消导致后续不可执行。

### 5.5 PlanExecutor 执行规则

1. `PlanRun` 从 `created` 进入 `planning`。
2. Planner 生成 steps 后，run 从 `planning` 进入 `running`。
3. Executor 只执行 `depends_on` 全部为 `succeeded` 的 step。
4. `tool_name` 必须存在于 scene 允许的工具集合或 retrieval tool 集合。
5. `input` 必须通过工具 `args_schema` 或 retrieval contract 校验。
6. 工具成功时 step 进入 `succeeded`，写入 `output/result_summary`。
7. 工具可重试失败时 step 进入 `retrying`，run 进入 `retrying`；retry 后回到 `running`。
8. 工具不可恢复失败或重试耗尽时 step 进入 `failed`，run 进入 `failed`。
9. 需要人工补充或审批时 step 进入 `waiting_user`，run 进入 `waiting_user`。
10. 用户 respond 后更新 step input 或新增 follow-up step，再回到 `running`。
11. 用户 approve 后执行 proposed tool call。
12. 用户 reject 后 step 和 run 进入 `cancelled`。
13. 所有必要 step 成功后进入 final answer 汇总，run 进入 `succeeded`。

### 5.6 Plan 状态映射

| Step 事件 | Step 状态 | Run 事件 | Run 状态 |
| --- | --- | --- | --- |
| planner_start | pending | plan_start | planning |
| planner_done | pending | run_start | running |
| step_start | running | run_start 或保持 running | running |
| step_success | succeeded | 若还有 step 则保持 running，否则 success | running / succeeded |
| step_retryable_error | retrying | tool_error_retryable | retrying |
| retry_step | running | retry | running |
| step_final_error | failed | tool_error_final 或 fail | failed |
| step_wait_user | waiting_user | interrupt | waiting_user |
| step_approve | running | resume_approve | running |
| step_respond | running | resume_respond | running |
| step_reject | cancelled | resume_reject 或 cancel | cancelled |

## 6. ReAct 与 Plan 的模式选择

### 6.1 决策规则

建议新增 `ModeSelector`，也可以先放在 `ChatService` 中做最小实现。

| 条件 | 推荐模式 | 理由 |
| --- | --- | --- |
| 单一问题、只需一次知识库检索 | ReAct | 计划成本高于收益 |
| 单一问题，但可能需要 rewrite 或 switch tool | ReAct | 当前 `AgenticRetriever` 已覆盖 |
| 用户要求解释概念、总结单个主题 | ReAct | 不需要显式步骤依赖 |
| 用户要求对比多个对象、排查流程、落地方案 | Plan | 需要拆分步骤和汇总 |
| 用户请求包含多个系统动作 | Plan | 需要步骤级审计和中断恢复 |
| 需要订单号、审批、外部 action tool | Plan | HITL 与副作用工具需要明确恢复点 |
| 用户显式要求“先列计划” | Plan | 符合用户交互预期 |
| `scene.infer_complexity()` 返回 `complex` | Plan | 复用现有复杂度判断 |

### 6.2 设计取舍

- ReAct 是默认轻量路径，不要求用户看到完整计划。
- Plan 是复杂路径，必须显式记录每个 step。
- Plan 不替代 ReAct；否则简单问答会被过度工程化。
- ReAct 可以复用现有 retrieval trace，Plan 需要新建任务结构。
- 两种模式都共享 Workflow State Machine、HITL、checkpoint、SSE 和 scene 工具边界。

## 7. 与现有能力的接入设计

### 7.1 Workflow State Machine

可直接复用 `backend/platform/workflow/state_machine.py` 的 run 级状态。接入差异：

- ReAct：通常 `created -> running -> succeeded/failed/waiting_user/retrying/cancelled`。
- Plan：必须经过 `created -> planning -> running`。
- Workflow 只管理 run 状态，turn/step 状态由 `ReActRun/PlanRun` 内部保存。
- `waiting_user` 表示等待用户输入，不是失败。
- `cancelled` 表示用户拒绝或系统取消，不是工具失败。

### 7.2 HITL

复用现有 `HitlWaitInput`、`HitlResumeInput`、`RuntimeHitlState`。

ReAct 模式：

- `ask_user` 时写入 `hitl.metadata.mode=react`。
- metadata 包含 `react_run_id/current_turn_id`。
- resume respond 后恢复当前 turn。
- resume reject 后 run 进入 `cancelled`。

Plan 模式：

- clarification step：`pending_action=clarification`。
- approval step：`pending_action=tool_approval` 或 `external_api_approval`。
- `proposed_tool_call` 写入 `tool_name/input/step_id`。
- metadata 包含 `mode=plan`、`plan_run_id/current_step_id`。
- resume respond 后更新 step input 或新增 follow-up step。
- resume approve 后继续执行当前 step。
- resume reject 后 run 进入 `cancelled`。

### 7.3 SSE event

现有业务事件为 `start/history/tool/chunk/waiting_user/resume/done/error`。为降低 API 破坏面，最小方案先不新增事件名，而是扩展 `tool` payload。

ReAct payload：

```json
{
  "stage": "react_turn",
  "react_run_id": "react_xxx",
  "turn_id": "turn_1",
  "turn_status": "succeeded",
  "action": "tool_call",
  "tool_name": "generic_knowledge_document_search"
}
```

Plan payload：

```json
{
  "stage": "plan_step",
  "plan_run_id": "plan_xxx",
  "step_id": "step_doc_policy",
  "step_status": "succeeded",
  "tool_name": "generic_knowledge_document_search"
}
```

后续如果前端需要更清晰进度，再新增 `plan` 和 `step` 事件。

### 7.4 Checkpoint

`RuntimeGraphState` 建议增加：

```python
react_run: NotRequired[dict[str, Any] | None]
plan_run: NotRequired[dict[str, Any] | None]
current_turn_id: NotRequired[str | None]
current_step_id: NotRequired[str | None]
plan_mode: NotRequired[Literal["react", "plan"]]
```

写入时机：

- ReAct 每轮 action/observation 完成后写入 `react_run.turns`。
- ReAct 进入 `waiting_user` 前必须写入 `react_run/current_turn_id/hitl`。
- Plan 完成 planning 后写入完整 `plan_run.steps`。
- Plan 每个 step 状态变化后写入 checkpoint。
- Plan 进入 `waiting_user` 前必须写入 `plan_run/current_step_id/hitl`。
- resume 后先消费 hitl，再恢复对应 turn 或 step，避免重复执行副作用。

## 8. 最小验证路径

### 8.1 ReAct 简单成功路径

用户问题：

> 根据知识库解释 Planner / Executor 是什么。

预期：

- ModeSelector 选择 ReAct。
- run `created -> running -> succeeded`。
- 生成 1 个 `ReActTurn`。
- turn 调用 `generic_knowledge_document_search`。
- 工具成功后写入 `observation_summary`。
- `finish` 后生成最终回答。
- SSE 输出 `tool(stage=react_turn)`、`chunk`、`done`。

### 8.2 ReAct 工具切换路径

用户问题：

> 查一下这个商品有没有库存，如果商品检索结果不够就继续查库存。

预期：

- 第一轮调用商品工具。
- decision 选择 `switch_tool`。
- 第二轮调用库存工具。
- 第二轮 `finish`。
- `ReActRun.turns` 保留两轮 action 和 observation。

### 8.3 Plan 多步成功路径

用户问题：

> 帮我对比当前知识库里的 Planner/Executor 设计和电商售后场景，说明哪些步骤要查文档、哪些步骤要查订单或商品工具，最后给出落地建议。

Plan 生成 3 个步骤：

| step_id | goal | tool_name | depends_on |
| --- | --- | --- | --- |
| `step_doc_policy` | 检索 Planner / Executor 设计资料 | `generic_knowledge_document_search` | `[]` |
| `step_ecommerce_context` | 检索电商售后相关订单或商品证据 | `order_semantic_search` 或 `product_semantic_search` | `[]` 或 `["step_doc_policy"]` |
| `step_synthesis` | 汇总文档和业务工具结果 | `final_answer_builder` 或模型汇总节点 | `["step_doc_policy", "step_ecommerce_context"]` |

预期：

- run `created -> planning -> running -> succeeded`。
- step 1、2 保存各自 `result_summary` 和引用摘要。
- step 3 使用前两步结果生成最终回答。
- SSE 输出 `tool(stage=plan_step)`、`tool(stage=plan_step)`、`chunk`、`done`。

### 8.4 Plan 工具失败重试路径

预期：

- step 2 第一次工具失败。
- step `running -> retrying`。
- run `running -> retrying`。
- retry 后 step 回到 `running` 并成功。
- 后续汇总成功，run `running -> succeeded`。

### 8.5 Plan HITL 中断路径

预期：

- Planner 或 Executor 发现订单号缺失，step 2 进入 `waiting_user`。
- run 进入 `waiting_user`。
- checkpoint 保存 `plan_run/current_step_id/hitl`。
- 用户 respond 补充订单号后 step 2 回到 `running`。
- 用户 reject 时 run 进入 `cancelled`。

## 9. 测试样本建议

后续实现建议新增 `backend/tests/test_react_plan_runtime.py`，或拆分为 `test_react_executor.py` 与 `test_planner_executor.py`。

| 样本 | 覆盖点 |
| --- | --- |
| ReAct 单轮成功 | 生成 1 个 `ReActTurn`，工具成功，run `succeeded` |
| ReAct 多轮工具切换 | 第一轮 `switch_tool`，第二轮 `finish` |
| ReAct ask_user | 当前 turn 与 run `waiting_user`，resume respond 后继续 |
| Plan 单步计划 | 生成 1 个 retrieval step，执行成功，run `succeeded` |
| Plan 多步计划 | 生成 2 到 3 个 step，按 `depends_on` 顺序执行 |
| Plan 工具失败重试 | step/run 进入 `retrying`，retry 成功后继续 |
| Plan 工具失败耗尽 | step/run 进入 `failed`，checkpoint 记录 error |
| Plan HITL 中断 | step/run `waiting_user`，resume respond 后继续执行 |
| 人工拒绝 | resume reject 后 run `cancelled`，工具不再执行 |
| SSE 兼容 | ReAct 输出 `tool(stage=react_turn)`，Plan 输出 `tool(stage=plan_step)` |

候选回归命令：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_react_plan_runtime.py -q -c backend\tests\pytest.ini
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_planner_executor.py -q -c backend\tests\pytest.ini
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agentic_retrieval.py backend\tests\test_langgraph_runtime.py backend\tests\test_generic_assistant_hitl.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

本次仅重写设计文档，不运行测试。

## 10. 当前功能满足度

按 PRD 的 6 条需求目标统计，当前项目代码层面还没有完全满足 Planner / Executor 闭环。

| 需求目标 | 当前满足度 | 证据 | 缺口 |
| --- | --- | --- | --- |
| 走查当前 planning / execution 职责 | 部分满足 | `AgenticRetriever`、`RetrievalExecutor`、HITL Planner、GraphRuntime 已分散承担部分职责 | 缺统一设计审计和显式 Planner/Executor 所属边界 |
| 定义最小 `PlanStep` | 未满足 | 当前只有 `RetrievalPlan`、`RetrievalResult`、`ToolResult` | 缺 `PlanStep` 模型、step status、步骤结果沉淀 |
| 定义最小 `PlanRun` | 未满足 | 当前 `RuntimeGraphState` 只有 chat/retrieval/hitl/retry 字段 | 缺 `PlanRun`、`current_step_id`、步骤列表 checkpoint |
| 设计 Executor 执行规则 | 部分满足 | 工具结果、workflow retry/wait/fail 状态已存在 | 缺按依赖执行、步骤级 retry、步骤级失败传播 |
| 明确 Planner/Executor 与 HITL、Workflow、SSE、checkpoint 边界 | 部分满足 | HITL、状态机、SSE mapper、checkpoint 均已有基础 | 缺 turn/step event 和 plan checkpoint 字段 |
| 写出最小验证路径 | 未满足 | 现有测试覆盖 Agentic Retrieval、HITL、workflow state | 缺 planner/executor smoke test、step status 和 final summary 断言 |

数量结论：

- 完全满足：0/6
- 部分满足：3/6
- 未满足：3/6
- 尚未完全满足：6/6

按模式拆分：

| 模式 | 当前覆盖 | 结论 |
| --- | --- | --- |
| ReAct | `AgenticRetriever` 和电商 follow-up tool routing 已部分覆盖行为 | 缺 `ReActRun/ReActTurn` 显式结构、checkpoint 字段和 SSE turn payload |
| Plan | 无显式 `PlanStep/PlanRun/PlanExecutor` | 是本 PRD 的主要实现缺口 |

## 11. 面试表达素材

### 11.1 一句话

> 这个项目不是只做一次 RAG 检索，而是设计了 ReAct / Plan 双模式 Agent Runtime：简单问题用 ReAct 做有限轮观察和行动，复杂问题用 Plan 显式拆分步骤；Workflow State Machine 管状态，HITL 负责中断恢复，checkpoint 保存可恢复上下文。

### 11.2 问题表达

- 企业里的复杂需求往往不是“一问一答”，而是多轮检索、多个系统工具、人工审批和失败恢复的组合。
- 如果只有一次 RAG 调用，无法解释每一步为什么执行、失败后从哪里恢复、人工确认后如何继续。
- ReAct / Plan 的价值是把隐式推理变成显式运行状态，使 trace、SSE、测试和面试讲解都有统一对象。

### 11.3 八股口径

- ReAct 适合轻量问题：边观察边决定下一步工具，用 `ReActRun/ReActTurn` 保存动作和观察摘要，当前 `AgenticRetriever` 已具备行为雏形。
- Plan 适合复杂任务：先形成步骤，再按依赖执行，步骤结果进入 checkpoint，可中断可恢复。
- Workflow State Machine 管 run 级状态，不直接替代 step 状态。
- HITL 是步骤执行中的 interrupt 机制，不是失败状态。
- Tool 只返回结构化结果，不能自己决定跨步骤调度。
- Planner 不直接调用工具，Executor 不重新发明业务 policy，scene 负责工具白名单和业务边界。

### 11.4 简历素材

> 设计 Agent Runtime 的 ReAct / Plan 双模式编排闭环：用 `ReActRun/ReActTurn` 表达边观察边行动的轻量任务，用 `PlanRun/PlanStep` 表达复杂任务拆解，支持检索与业务工具调用、结果沉淀、工具失败重试、HITL 中断恢复和 Workflow State Machine 状态治理，推动项目从增强 RAG 升级为可审计、可恢复的 Agent Runtime。

## 12. 实施前验收清单

- [x] 指出当前 planning / execution 逻辑分散位置。
- [x] 明确简单问题使用 ReAct，复杂问题使用 Plan。
- [x] 给出 `ReActTurn` 最小字段设计。
- [x] 给出 `ReActRun` 最小字段设计。
- [x] 给出 `PlanStep` 最小字段设计。
- [x] 给出 `PlanRun` 最小字段设计。
- [x] 说明 Planner、Executor、Tool、Workflow State Machine、HITL 关系。
- [x] 说明 step 成功、失败、重试、等待人工确认如何改变 run state。
- [x] 列出 ReAct / Plan 最小测试样本。
- [x] 产出面试卡、八股口径和简历素材。

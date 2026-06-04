# PRD-20260603-03 Agentic RAG Subgraph 迁移审计与节点拆分方案

## 1. 结论

这版方案不再把问题收窄成“只把 `AgenticRetriever` 改成 LangGraph subgraph”。当前项目真正需要补的是 **顶层 `/chat` ChatGraph + ReAct / Plan 子图 + Agentic RAG 工具内子图** 的分层迁移。

当前判断：

- `LangGraph Runtime` 已有 checkpoint、thread、lifecycle、HITL 状态写回和极小 graph 骨架。
- `/chat` 的真实 Plan / ReAct / Agentic RAG 执行拓扑仍主要由手写 loop 和 runtime glue 承担。
- `Agentic RAG` 应该迁移为工具内部嵌套 subgraph，但它不是唯一需要 graph 化的对象。
- 最小迁移顺序应该先定义顶层 `ChatGraph`，再逐步拆 `Agentic RAG`、`ReAct`、`Plan` 内部循环。

对应 OpenSpec change 已创建：

- `openspec/changes/graphify-chat-agent-runtime-and-rag-subgraph/`

## 2. 当前 LangGraph 是如何定义的

当前项目里的 LangGraph 定义仍然偏骨架层：

- 普通回答图：`START -> answer -> END`
  - 位置：`backend/application/runtime/graph_runtime_parts/answer_graph.py`
- HITL 状态写入图：`START -> hitl_state_update -> END`
  - 位置：`backend/application/runtime/graph_runtime_parts/state_store.py`
- `ChatGraphRuntime` 负责 thread/checkpoint/lifecycle/run state 的装配和调用
  - 位置：`backend/application/runtime/graph_runtime.py`

也就是说，当前 LangGraph 已经承担了运行时状态能力，但还没有显式表达：

- `/chat` turn preparation
- mode selection
- ReAct / Plan route
- tool execution branch
- Agentic RAG retrieval loop
- final synthesis
- persist turn

这就是 README 里需要区分“Runtime 骨架已完成”和“真实 Agent Runtime Graph 接入未完成”的原因。

## 3. 当前手写循环审计

### 3.1 ReAct

位置：`backend/platform/agent_runtime/react/runtime.py`

当前 ReAct 仍然由 runtime 内部循环控制：

- action selection
- tool call
- observation 归一化
- HITL waiting / resume 边界
- final synthesis
- max turn / failure / fallback

迁移判断：适合下沉为 ReAct subgraph，但第一步可以先作为 `ChatGraph` 的粗粒度 `react_branch` 节点运行。

### 3.2 Plan

位置：`backend/platform/agent_runtime/plan/planner.py`、`backend/platform/agent_runtime/plan/executor.py`

当前 Plan 仍然由 planner/executor 类控制：

- plan create
- step dependency selection
- step execution
- retry / failed / waiting_user
- final plan result synthesis

迁移判断：适合下沉为 Plan subgraph，但第一步可以先作为 `ChatGraph` 的粗粒度 `plan_branch` 节点运行。

### 3.3 Agentic RAG

位置：`backend/platform/rag/orchestration/agentic.py`

当前 `AgenticRetriever` 内部仍然串行控制：

- query rewrite
- tool decision / switch tool
- retrieval
- rerank
- sufficiency check
- no-hit fallback
- final evidence synthesis
- retrieval trace 聚合

迁移判断：适合拆成 `AgenticRagSubgraph`，但应继续作为 RAG tool 内部子图，而不是顶层 `/chat` graph 的直接主分支。

## 4. 功能需求满足情况

### 4.1 按原 PRD 的 7 个候选节点统计

这 7 个业务阶段在当前代码中“能跑”，但都还没有以 LangGraph subgraph 节点形式落地：

| 候选节点 | 当前业务能力 | 当前 graph 化状态 |
| --- | --- | --- |
| query rewrite | 已有 | 未满足 |
| tool decision | 已有 | 未满足 |
| retrieval | 已有 | 未满足 |
| rerank | 已有 | 未满足 |
| sufficiency check | 已有 | 未满足 |
| no-hit fallback | 已有 | 未满足 |
| final synthesis | 已有 | 未满足 |

统计口径：

- 业务执行能力：`7/7` 已具备。
- LangGraph subgraph 迁移能力：`0/7` 已满足，`7/7` 未满足。

### 4.2 按扩展后的 ChatGraph 迁移需求统计

| 需求项 | 当前状态 | 说明 |
| --- | --- | --- |
| LangGraph checkpoint / thread / lifecycle 骨架 | 已满足 | 已有 runtime skeleton |
| Workflow State Machine / HITL 状态语义 | 已满足 | 已有统一状态和等待/恢复语义 |
| `/chat` response / SSE / citation / trace 兼容基础 | 已满足 | 现有链路可运行 |
| 顶层 `ChatGraph` 显式拓扑 | 未满足 | 目前没有完整 `prepare_turn -> route -> synthesis -> persist` 图 |
| ReAct subgraph | 未满足 | 仍是手写 loop |
| Plan subgraph | 未满足 | 仍是 planner/executor loop |
| Agentic RAG subgraph | 未满足 | 仍是 `AgenticRetriever` 内部 loop |
| graph state 统一字段回写 | 部分满足 | 字段已有一部分，但还不是统一图状态合同 |
| SSE 事件由真实 graph 节点进度驱动 | 部分满足 | 已有事件，但大量映射仍来自 runtime glue |
| 子图单测和 trace 一致性验证 | 未满足 | 需后续实现后补齐 |

统计口径：

- 已满足：`3/10`
- 部分满足：`2/10`
- 未满足：`5/10`

## 5. 目标架构

### 5.1 顶层 ChatGraph

目标拓扑：

```text
prepare_turn
  -> select_mode
  -> route_mode
     -> react_branch
     -> plan_branch
  -> resolve_answer_mode
  -> final_synthesis
  -> persist_turn
```

顶层 `ChatGraph` 负责：

- 解析 session / scene / mounted knowledge sources
- 准备 messages 和 request metadata
- 选择 `agent_mode=react|plan`
- 路由到 ReAct 或 Plan 分支
- 保留 `direct response / fallback` 这类不需要进入子图的普通回答出口
- 汇总 answer / citations / retrieval_trace / final_state
- 统一 checkpoint、lifecycle 和 turn persistence 边界

第一阶段不要求立刻拆开 ReAct / Plan 内部循环，可以先将现有 `ReActRuntime.run()` 和 `PlanExecutor.run()` 包为粗粒度 branch node。

### 5.2 ReAct Subgraph

ReAct 这边需要保留一个普通回答节点，不然会把“不需要工具，也不需要追问”的最常见路径漏掉。

候选节点：

- `select_action`
- `validate_action`
- `route_action`
- `respond`
- `execute_tool`
- `ask_user`
- `final_answer`
- `loop_or_finish`

核心回写点：

- `react_run`
- `current_turn_id`
- `current_tool_call`
- `tool_observation`
- `knowledge_used`
- `citations`
- `retrieval_trace`
- `final_state`

HITL 判断：需要保留。`ask_user`、工具审批、工具 reject/cancel 都应继续映射到 `waiting_user` / `cancelled`。
`respond` 用于普通直接回答，不代表工具调用，也不代表人工补充。

### 5.3 Plan Subgraph

Plan 不需要和 ReAct 一样单独补一个 `respond` 节点，它的正常语义是先生成计划，再按 step 执行，再汇总。

候选节点：

- `create_plan`
- `select_next_step`
- `execute_step`
- `handle_retry`
- `handle_waiting_user`
- `synthesize_plan_result`

核心回写点：

- `plan_run`
- `current_step_id`
- `current_tool_call`
- `step.status`
- `step.result_summary`
- `step.error`
- `final_state`

HITL 判断：需要保留。Plan step 可能等待用户补充、审批或取消，必须继续走统一 workflow state。

### 5.4 Agentic RAG Subgraph

候选节点：

- `initialize_plan`
- `tool_decision`
- `retrieval`
- `rerank`
- `sufficiency_check`
- `route_next_action`
- `query_rewrite`
- `no_hit_fallback`
- `final_evidence_synthesis`

RAG subgraph 输入：

- `query`
- `candidate_tools`
- `scene`
- `mounted_knowledge_sources`
- retrieval policy
- rerank config

RAG subgraph 输出：

- `tool_observation`
- `retrieval_trace`
- `candidate_docs`
- `knowledge_used`
- `citations`
- `final_decision`
- `follow_up_question`

HITL 判断：RAG 内部不应新增顶层人工审批语义，但 `ask_user` / clarification 结果需要继续作为工具观察结果向上冒泡，由顶层 ReAct / Plan 进入 `waiting_user`。

## 6. Graph State 字段

### 6.1 顶层 ChatGraph state

建议保留：

- `session_id`
- `request_id`
- `scene`
- `agent_mode`
- `messages`
- `answer`
- `knowledge_used`
- `citations`
- `retrieval_trace`
- `status`
- `run_id`
- `state_event`
- `final_state`
- `react_run`
- `plan_run`
- `current_turn_id`
- `current_step_id`
- `current_tool_call`

### 6.2 Agentic RAG subgraph state

建议保留：

- `query`
- `active_query`
- `rewritten_query`
- `selected_tool`
- `candidate_tools`
- `attempted_tools`
- `candidate_docs`
- `retrieval_trace`
- `tool_observation`
- `knowledge_used`
- `citations`
- `final_decision`
- `follow_up_question`

## 7. 哪些逻辑下沉到 LangGraph

适合下沉：

- mode route
- ReAct loop route
- Plan step dependency route
- Agentic RAG next action route
- retry / waiting / terminal decision
- final synthesis 前的状态归一化
- checkpoint 和 lifecycle 边界

暂时保留在外围：

- FastAPI route 和 request/response schema
- session store / chat message read model
- scene definition 和工具装配
- Tool Registry / ToolExecutor 的访问控制
- 具体 retrieval 算法、rerank provider、query rewrite prompt 策略
- SSE business protocol 的对外事件名

## 8. 最小迁移顺序

1. 顶层 `ChatGraph` 先接入。
   - 先把 `ReActRuntime.run()` 和 `PlanExecutor.run()` 包成粗粒度 branch node。
   - 目标是先让 `/chat` 的拓扑、checkpoint、lifecycle 和 final synthesis 边界显式化。

2. 拆 `AgenticRetriever.retrieve_with_trace()` 为 `AgenticRagSubgraph`。
   - 优先拆 retrieval、rerank、sufficiency、rewrite、fallback、final decision。
   - 保持 retrieval trace 嵌套在 tool observation 下。

3. 拆 ReAct 内部 loop。
   - 从 `respond`、`execute_tool`、`ask_user`、`final_answer` 这些边界清楚的节点开始。

4. 拆 Plan 内部 loop。
   - 从 `create_plan`、`select_next_step`、`execute_step` 开始，再补 retry/HITL。

5. 统一验证。
   - ChatGraph state shape
   - Agentic RAG subgraph terminal decision
   - `/chat` JSON 兼容
   - SSE `tool / waiting_user / done`
   - retrieval_trace / citations / knowledge_used 一致性

## 9. 后续验证候选

今天只做审计和计划，不做完整执行验证。后续实现时至少补：

- `ChatGraph` state shape 单测
- ReAct coarse branch / Plan coarse branch 路由测试
- `AgenticRagSubgraph` terminal decision 单测
- `/chat` non-streaming 回归测试
- `/chat` SSE `tool` / `waiting_user` / `done` 回归测试
- retrieval trace 字段一致性测试
- citations / `knowledge_used` 不误采纳候选证据测试

## 10. 面试追问口径

如果被问“为什么不只做 RAG subgraph”，回答应该是：

> RAG subgraph 是必要的，但它只是工具内部的 evidence orchestration。当前项目更大的问题是 `/chat` 的顶层 ReAct / Plan 编排还没有显式 graph 化。如果只拆 RAG，顶层 mode route、HITL resume、final synthesis、checkpoint 和 SSE 映射仍然散在 runtime glue 里。所以正确顺序是先定义 ChatGraph，再把 ReAct / Plan / RAG 分层下沉。

如果被问“为什么不一次性全改”，回答应该是：

> 因为当前链路已经有可运行的 API、SSE、HITL、citation 和 trace 行为。一次性改所有 loop 风险太大。第一步把 ReAct / Plan 包成粗粒度 graph node，可以先稳定 graph state 和生命周期边界，再逐步拆内部节点。

## 11. 1.x 审计核对结果

这次审计已经按当前代码核对完毕，结论和本方案保持一致：

- `backend/application/runtime/graph_runtime_parts/answer_graph.py` 仍然是 `START -> answer -> END` 的最小图。
- `backend/application/runtime/graph_runtime_parts/state_store.py` 仍然只负责 checkpoint 写回与状态装配，没有顶层 ChatGraph 拓扑。
- `backend/application/runtime/graph_runtime.py` 负责 checkpointer、lifecycle、input state、stream state 的统一装配。
- ReAct 的实现已迁移到 `backend/platform/agent_runtime/react/runtime.py`。
- Plan 的实现已迁移到 `backend/platform/agent_runtime/plan/executor.py`。
- Agentic RAG 的多轮编排仍在 `backend/platform/rag/orchestration/agentic.py`。
- `RuntimeGraphState` 目前保留的核心字段包括 `session_id`、`request_id`、`messages`、`answer`、`knowledge_used`、`citations`、`retrieval_trace`、`metadata`，以及 `agent_mode`、`react_run`、`plan_run`、`current_turn_id`、`current_step_id`、`current_tool_call`。
- `/chat` 的 SSE 和 HITL 兼容仍依赖 `chat_service_parts/events.py`、`chat_service_parts/hitl.py` 和 `graph_runtime_parts/agent_state.py` 这些投影边界。

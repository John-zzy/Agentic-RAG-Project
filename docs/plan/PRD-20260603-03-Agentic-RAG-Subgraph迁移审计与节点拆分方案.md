# PRD-20260603-03 Agentic RAG Subgraph 迁移审计与节点拆分方案

## 1. 范围

这版方案按新的分析重写：不再只讨论 `AgenticRetriever` 的局部子图，而是把 **整个 `/chat` 的 Plan / ReAct / RAG 过程** 放进统一的 LangGraph 视角里审计。

结论先说：

- 顶层应该有 `ChatGraph`
- `ReAct` 和 `Plan` 应该逐步变成子图
- `AgenticRAG` 继续作为工具内的子图
- 当前代码里的 LangGraph 还只是 checkpoint / lifecycle 壳层，不是完整编排图

这份文档只做架构审计、节点拆分和迁移顺序，不在本阶段做完整重构。

## 2. 当前实现审计

### 2.1 现在的 LangGraph 只负责状态壳

当前真正编译成 LangGraph 的图很小：

- 普通回答图只有 `START -> answer -> END`，见 [answer_graph.py](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/graph_runtime_parts/answer_graph.py:12>)。
- 状态写入图只有 `START -> hitl_state_update -> END`，见 [state_store.py](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/graph_runtime_parts/state_store.py:81>)。

也就是说，LangGraph 目前承担的是：

- checkpoint 保存
- lifecycle 记录
- HITL 状态写回

它还没有接管 `/chat` 的实际策略编排。

### 2.2 当前 ReAct / Plan / RAG 仍是手写循环

现状如下：

- ReAct 仍在 [runtime.py](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/react_parts/runtime.py:42>) 里用 `while` 循环调 action、tool、HITL、final synthesis。
- Plan 仍在 [plan_executor.py](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/plan_executor.py:78>) 里用 `while` 循环执行 step。
- Agentic RAG 仍在 [agentic.py](</d:/Programs/interview-projects/ai-rag-project/backend/platform/rag/orchestration/agentic.py:54>) 里用 `while True` 管理 rewrite、switch_tool、rerank、sufficiency 和 fallback。

所以现在的真相是：

`/chat` 的“Agent Runtime”已经跑起来了，但还没有被真正 graph 化。

## 3. 需求差距

### 3.1 以这次 PRD 为准的差距

如果只看 Agentic RAG 子图这件事，当前仍然是：

- 业务能力可运行
- 图化能力未落地

也就是：

- `query rewrite` 已有实现
- `tool decision` 已有实现
- `retrieval` 已有实现
- `rerank` 已有实现
- `sufficiency check` 已有实现
- `no-hit fallback` 已有实现
- 但这些都还是 `AgenticRetriever` 内部的手写循环，不是 subgraph 节点

### 3.2 以整个 chat graph 为准的差距

更关键的是，顶层 `ChatGraph` 也还没有成型。

目前缺少：

- `prepare_turn`
- `select_mode`
- `route_to_react_or_plan`
- `react_subgraph`
- `plan_subgraph`
- `final_synthesis`
- `persist_turn`

因此这次方案要修正的重点不是“只做 RAG 子图”，而是“先把顶层 chat graph 定出来，再把 ReAct / Plan / RAG 逐层下沉”。

## 4. 目标架构

### 4.1 顶层 ChatGraph

建议顶层图这样划分：

```text
prepare_turn
  -> select_mode
  -> route_mode
     -> react_subgraph
     -> plan_subgraph
  -> resolve_answer_mode
  -> final_synthesis
  -> persist_turn
```

顶层图负责：

- 会话 / scene / mounted knowledge source 解析
- mode selection
- ReAct / Plan 路由
- 最终 answer / citation / trace 汇总
- checkpoint 和 lifecycle 写回

### 4.2 ReAct 子图

ReAct 不应该继续停留在一个大 `while` 函数里。建议拆成：

- `select_action`
- `validate_action`
- `route_action`
- `execute_tool`
- `ask_user`
- `final_answer`
- `loop_or_finish`

### 4.3 Plan 子图

Plan 建议拆成：

- `create_plan`
- `select_next_step`
- `execute_step`
- `handle_retry`
- `handle_waiting_user`
- `synthesize_plan_result`

### 4.4 Agentic RAG 子图

RAG 仍然是工具内子图，不是顶层入口。建议节点保持：

- `initialize_plan`
- `retrieval`
- `rerank`
- `sufficiency_check`
- `route_next_action`
- `query_rewrite`
- `tool_decision`
- `no_hit_fallback`
- `final_evidence_synthesis`

这部分和之前方案一致，但它现在是挂在顶层 chat graph 下面的一个局部子图，而不是整个方案的唯一重点。

## 5. Graph State 设计

### 5.1 顶层 ChatGraph state

建议顶层状态至少保留：

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

### 5.2 RAG 子图 state

RAG 子图继续保留：

- `query`
- `active_query`
- `rewritten_query`
- `selected_tool`
- `candidate_tools`
- `attempted_tools`
- `results`
- `candidate_docs`
- `retrieval_trace`
- `tool_observation`
- `knowledge_used`
- `citations`
- `final_decision`
- `follow_up_question`

## 6. 为什么要这么拆

### 6.1 不是为了“全图化而全图化”

现在不把整个 chat 图化，问题不在于“不能跑”，而在于：

- 逻辑边界不显式
- 恢复点不清晰
- 事件映射靠手工 glue
- ReAct / Plan / RAG 的循环语义不统一

### 6.2 为什么不能只做 RAG 子图

只做 RAG 子图，会留下一个结构性问题：

顶层 `/chat` 还是手写 orchestration，RAG 只是一个更大的工具。

这对当前项目的定位不够完整，因为：

- ReAct / Plan 已经是主链路，不应该长期停留在 loop class 里
- `ChatGraphRuntime` 已经在做 checkpoint 和 lifecycle，下一步应该承接真正的图编排
- 面试里问“整个 chat 怎么 graph 化”时，答案不能只停在 RAG

## 7. 建议的迁移顺序

### 第 1 步

先把当前 `ReActRuntime.run()` 和 `PlanExecutor.run()` 包成顶层 `ChatGraph` 的粗粒度节点。

目标不是改内部算法，而是先把顶层路由、状态和 checkpoint 边界 graph 化。

### 第 2 步

把 `AgenticRetriever.retrieve_with_trace()` 拆成 `AgenticRagSubgraph`。

这里优先做：

- retrieval
- rerank
- sufficiency
- rewrite
- fallback

### 第 3 步

再把 ReAct / Plan 内部的 `while` 循环进一步拆细成子节点。

### 第 4 步

统一：

- SSE 事件
- retrieval trace
- run lifecycle
- HITL resume
- checkpoint payload

## 8. 现在的结论

如果只问“要不要定义整个 chat 的 Plan / ReAct 过程为 LangGraph”，答案是：**要**。

但正确做法不是一口气把所有细节重写，而是：

1. 先有顶层 `ChatGraph`
2. 再把 ReAct / Plan 变成子图或粗粒度节点
3. 再把 RAG 做成局部子图

这样既能保留当前可运行状态，又能逐步把手写循环收敛成可恢复、可观测、可组合的 graph。

## 9. 验收口径

- 顶层 chat graph 已显式定义
- ReAct / Plan 不再只是普通 loop class
- Agentic RAG 是嵌套子图，不是唯一 graph 化对象
- checkpoint / HITL / SSE 事件与 graph state 对齐
- 迁移顺序是分层推进，不是一次性重构


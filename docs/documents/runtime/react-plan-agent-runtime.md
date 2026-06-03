# ReAct / Plan Agent Runtime

这份文档只讲一件事：`/chat` 是怎么跑起来的。

先记住三个最重要的判断：

1. `/chat` 不是“先检索再回答”，而是先组织成一次可执行的 Agent 运行。
2. 简单任务走 ReAct，复杂任务走 Plan。
3. RAG 和 Agentic RAG 只是可调用工具，不是顶层入口。

主链路图在这里：

- [SVG](./main-chat-agent-runtime-flow.svg)
- [Mermaid](./main-chat-agent-runtime-flow.mmd)

## 先统一理解

你可以把整个链路想成四层：

1. 入口层：`POST /chat` 收到用户问题。
2. 运行层：`ChatService` 把这次请求整理成一个运行上下文。
3. 编排层：`ModeSelector` 决定走 ReAct 还是 Plan。
4. 执行层：`ToolExecutor` 真正调用工具，RAG 只是其中一种工具。

最后，`ChatGraphRuntime` 把这次运行写入 checkpoint，SSE 或普通 JSON 只是不同的输出方式。

## 主链路图怎么读

先按图从左到右看：

1. 用户发起 `/chat`。
2. `ActiveSceneChatService` 根据会话找到当前 scene。
3. `ChatService` 准备消息、request id、知识源和运行上下文。
4. `ModeSelector` 判断这次更像 ReAct 还是 Plan。
5. ReAct 或 Plan 只负责“怎么决定下一步”，真正执行工具还是交给 `ToolExecutor`。
6. 如果工具是 RAG，那么 RAG 内部还会继续做 query rewrite、检索、rerank、证据判断。
7. 最后把工具结果汇总成回答、引用和 trace。

这个图最容易误解的地方有两个：

- 顶层 Agent 不是 RAG。
- Agentic RAG 内部的多轮检索，不等于顶层 ReAct 的多轮 turn，也不等于 Plan 的多步 step。

## 入口层：`/chat` 先把一次请求变成可执行上下文

从代码上看，最先接住请求的是 [chat routes](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/api/chat/routes.py:46>)，然后进入 [ActiveSceneChatService](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/service.py:266>) 和 [ChatService](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/service.py:83>)。

你可以把这一步理解成“先把场景、上下文、工具范围整理好”，还没开始真正回答问题。

实际的 Agent 编排入口在 [ChatAgentRuntimeMixin](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/chat_service_parts/agent_runtime.py:44>)。这里会做三件事：

- 组装可用工具。
- 选择 ReAct 或 Plan。
- 把执行结果整理成最终可返回的结构。

如果你只想先抓主干，先看这三个文件就够了。

## 先选模式：简单问题用 ReAct，复杂问题用 Plan

模式选择器在 [ModeSelector](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/mode_selector.py:30>)，默认别名是 [MinimalModeSelector](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/mode_selector.py:115>)。

它做的事情很直白：

- 简单问题，走 ReAct。
- 需要分步骤、依赖执行、步骤汇总的问题，走 Plan。

可以把它理解成分流闸门，而不是智能回答器。

例子：

- “这份文档讲了什么？”通常走 ReAct。
- “先查三份资料，再对比，再输出计划”通常走 Plan。

## ReAct 怎么跑

ReAct 的核心代码在 [ReActRuntime](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/react_parts/runtime.py:42>)，下一步怎么选由 [LLMReActActionSelector](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/react_parts/selector.py:161>) 决定。

它的执行顺序可以直接记成五步：

1. 选择器输出一个结构化 action，比如 `tool_call`、`ask_user`、`final_answer`。
2. `ReActRuntime` 校验 action 是否合理。
3. `ToolExecutor` 执行真正的工具调用。
4. 工具结果写回 `ReActTurn` 和 `ReActRun`。
5. 如果还没结束，就进入下一轮；如果够了，就汇总成最终回答。

相关数据结构在 [ReActRun](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/contracts.py:141>) 和 [ReActTurn](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/contracts.py:114>)。

ReAct 的直觉很简单：

- 它不是先写完所有步骤。
- 它是“看一步，做一步，再决定下一步”。

所以 ReAct 更适合那些目标不完全固定、需要根据中间结果调整的问题。

## Plan 怎么跑

Plan 的两个核心类是 [MinimalPlanner](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/planner.py:70>) 和 [PlanExecutor](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/plan_executor.py:78>)。

Plan 先生成计划，再执行计划。它不是边走边想，而是先把任务拆成步骤。

对应的数据结构是 [PlanRun](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/contracts.py:178>) 和 [PlanStep](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/contracts.py:161>)。

你可以这样理解：

1. `MinimalPlanner` 根据用户目标、场景策略和可用工具，生成 step 列表。
2. 每个 step 会带自己的 `goal`、`tool_name`、`input` 和 `depends_on`。
3. `PlanExecutor` 只执行已经满足依赖的 step。
4. 每个 step 的结果会沉淀成 observation。
5. 最后再把多个 step 的结果汇总成最终回答。

Plan 更适合这类问题：

- “先查资料，再列对比表，再总结结论”
- “把任务拆成 3 步，逐步执行”
- “中间某一步需要人工确认，确认后继续”

## 为什么所有工具都要先过 ToolExecutor

真正的工具入口是 [ToolExecutor](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/tool_executor.py:24>)。

它的作用不是“帮模型想”，而是“帮系统守边界”：

- 只允许调用当前场景允许的工具。
- 先校验工具名，再校验输入 schema。
- 支持 `BaseTool`、`SceneTool`，也支持 RAG adapter 这类平台工具。

这很重要，因为顶层 Agent 不应该直接碰具体业务实现。
它只能说“我要用这个工具”，至于能不能用、参数对不对、权限对不对，都由 `ToolExecutor` 决定。

## RAG 为什么只是工具

RAG 的顶层工具包装在 [build_rag_tool_adapters](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/rag_tools.py:188>)。

如果走的是 Agentic RAG，真正的内部流程在 [AgenticRetriever](</d:/Programs/interview-projects/ai-rag-project/backend/platform/rag/orchestration/agentic.py:54>)，入口是 [retrieve_with_trace](</d:/Programs/interview-projects/ai-rag-project/backend/platform/rag/orchestration/agentic.py:76>)。

它内部一般会经历：

1. 改写问题。
2. 做文档或业务检索。
3. 重排结果。
4. 判断证据够不够。
5. 够就回答，不够就继续查、切工具，或者追问用户。

所以 RAG 不是顶层 ReAct，也不是顶层 Plan。
它只是顶层 Agent 调用的一个“证据收集工具”。

## 状态、checkpoint、HITL 怎么理解

运行状态的容器是 [RuntimeGraphState](</d:/Programs/interview-projects/ai-rag-project/backend/platform/workflow/langgraph/state.py:55>)，构造函数在 [build_runtime_graph_state](</d:/Programs/interview-projects/ai-rag-project/backend/platform/workflow/langgraph/state.py:120>) 和 [build_runtime_hitl_state](</d:/Programs/interview-projects/ai-rag-project/backend/platform/workflow/langgraph/state.py:82>)。

状态机规则在 [WorkflowStateMachine](</d:/Programs/interview-projects/ai-rag-project/backend/platform/workflow/state_machine.py:94>) 一带。它负责保证状态转移合法，比如：

- `running` 可以转到 `waiting_user`。
- `waiting_user` 可以恢复。
- 终态不能随便再改成运行中。

`waiting_user` 的意思不是失败，而是“暂停，等人来接着做”。

这个暂停点通常会带着：

- `react_run_id`
- `plan_run_id`
- `current_turn_id`
- `current_step_id`

用户 `approve` 或 `respond` 后，会继续同一个 run；`reject` 则会把这次运行收成 `cancelled`。

`ChatGraphRuntime` 负责把这些状态真正写进 checkpoint，见 [ChatGraphRuntime](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/graph_runtime.py:34>)、[invoke](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/graph_runtime.py:58>) 和 [start_stream_run](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/graph_runtime.py:99>)。

## 两个小例子

### 例子一：简单问题

用户问：“这份文档讲了什么？”

通常路径是：

1. `ModeSelector` 选 ReAct。
2. `LLMReActActionSelector` 选择一个 RAG 工具。
3. `ToolExecutor` 调用 RAG。
4. RAG 返回证据、引用和 trace。
5. ReAct 汇总成回答。

### 例子二：复杂任务

用户问：“先查三份资料，再做对比，最后输出一个执行计划。”

通常路径是：

1. `ModeSelector` 选 Plan。
2. `MinimalPlanner` 拆出多个 step。
3. `PlanExecutor` 按依赖顺序执行。
4. 每个 step 的结果写入 observation。
5. 最后把多个 step 的结果汇总成计划。

## 如果你要看源码，建议按这个顺序

1. [主链路图](./main-chat-agent-runtime-flow.mmd)
2. [chat routes](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/api/chat/routes.py:46>)
3. [ChatService](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/service.py:83>)
4. [ChatAgentRuntimeMixin](</d:/Programs/interview-projects/ai-rag-project/backend/application/runtime/chat_service_parts/agent_runtime.py:44>)
5. [ModeSelector](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/mode_selector.py:30>)
6. [ReActRuntime](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/react_parts/runtime.py:42>)
7. [MinimalPlanner](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/planner.py:70>)
8. [PlanExecutor](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/plan_executor.py:78>)
9. [ToolExecutor](</d:/Programs/interview-projects/ai-rag-project/backend/platform/agent_runtime/tool_executor.py:24>)
10. [AgenticRetriever](</d:/Programs/interview-projects/ai-rag-project/backend/platform/rag/orchestration/agentic.py:54>)

这样读下来，你会先明白“谁负责分流”，再明白“谁负责执行”，最后明白“RAG 为什么只是工具”。

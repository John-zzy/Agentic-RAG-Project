# PRD：LangChain middleware 与 LangGraph 现代化迁移

## 1. 文档信息

- 文档类型：PRD / 技术产品需求文档
- 需求名称：LangChain middleware 化与 LangGraph 最新特性现代化迁移
- 目标版本：分阶段交付，不绑定单一版本
- 当前状态：规划中
- 适用范围：后端 Agent Runtime、ChatGraph、ReAct、Plan、Agentic RAG、HITL、SSE、评测与运行时文档
- 约束来源：`AGENTS.md`、`README.md`、`docs/documents/` 当前架构说明

## 2. 背景

当前项目已经基于 LangGraph 构建了多层运行时：

- 顶层 `ChatGraph` 负责 `/chat` 会话准备、模式选择、ReAct / Plan 分支、HITL 等待、最终回答和持久化。
- `ReAct` 子图负责即时工具调用、澄清等待、工具观察记录和结果合成。
- `Plan` 子图负责多步计划、依赖顺序、重试、HITL 和结果合成。
- `Agentic RAG Retrieval Graph` 作为顶层 Agent 可调用的检索工具，负责 query rewrite、检索、rerank、证据充分性判断和 no-hit fallback。

当前依赖已经是 LangChain / LangGraph v1 系列：

- `langchain==1.3.1`
- `langchain-core==1.4.0`
- `langgraph==1.2.0`

因此迁移重点不是依赖大升级，而是把自研 Agent loop、工具拦截、模型保护、HITL、流式事件和 checkpoint 使用方式逐步切换到 LangChain middleware 与 LangGraph 最新推荐 API。

参考资料：

- LangChain middleware 概览：https://docs.langchain.com/oss/python/langchain/middleware
- LangChain middleware API reference：https://reference.langchain.com/python/langchain/agents/middleware
- LangChain custom middleware：https://docs.langchain.com/oss/python/langchain/middleware/custom
- LangGraph streaming：https://docs.langchain.com/oss/python/langgraph/streaming
- LangGraph human-in-the-loop：https://docs.langchain.com/oss/python/langgraph/human-in-the-loop
- LangGraph subgraphs：https://docs.langchain.com/oss/python/langgraph/use-subgraphs

## 3. 问题陈述

当前 runtime 能力完整，但存在以下工程问题：

1. ReAct loop、工具执行、模型保护、HITL 拦截、trace 汇总等逻辑部分自研，后续维护成本高。
2. 横切能力分散在 `ModelClient`、`ToolExecutor`、ReAct / Plan 节点、ChatService projection 中，不利于统一治理。
3. LangGraph 最新能力如 typed streaming、`GraphOutput`、`interrupt()` / `Command(resume=...)`、subgraph persistence 尚未形成稳定项目规范。
4. 大规模迁移存在回归风险，特别是 HITL resume、副作用幂等、citation / retrieval trace 和 SSE UI 协议。
5. 当前缺少 worktree 双轨开发和 baseline / candidate 并行验证流程，复杂迁移一旦出问题回滚成本偏高。

## 4. 目标

1. 将模型调用、工具调用、动态 prompt、权限、重试、审计、HITL 前置拦截等横切能力迁到 LangChain middleware。
2. 保留 LangGraph 作为显式业务编排层，尤其保留多分支、Plan、RAG 子图、checkpoint、HITL resume 和状态机语义。
3. 优先使用 LangGraph v1.2 之后推荐的 `GraphOutput`、typed streaming、`Command(resume=...)`、`interrupt()`、subgraph persistence 等能力。
4. 通过 feature flag 渐进替换，确保每一阶段可独立验证和回滚。
5. 使用 Git worktree 方式开发迁移分支，保留 baseline 分支用于并行测试和结果对比。
6. 在不改变 `/chat`、`/chat/resume`、SSE UI 协议和 eval 样本语义的前提下降低自研 runtime 复杂度。

## 5. 非目标

- 不一次性删除 `ChatGraph`、`Plan`、`Agentic RAG Retrieval Graph`。
- 不把 scene 业务工具或 API schema 下沉到 `platform`。
- 不把 SSE 输出改成直接透传 LangChain / LangGraph 内部事件。
- 不牺牲现有 `waiting_user`、`approve / reject / respond`、旧 `interrupt_id` 拒绝、工具副作用只执行一次等安全语义。
- 不在迁移期间大面积重写 RAG 检索、知识入库或前端 UI。
- 不在主工作区直接推进大规模实验性改造。

## 6. 用户故事

1. 作为开发者，我希望模型保护、工具策略和审计逻辑以 middleware 形式集中管理，以便新增策略时不需要改多个 graph 节点。
2. 作为维护者，我希望 ReAct runtime 可以逐步替换成 LangChain agent provider，以便减少自研循环和选择器维护成本。
3. 作为 reviewer，我希望迁移分支和 baseline 分支可以并行运行同一批测试，以便快速判断行为是否漂移。
4. 作为运维者，我希望迁移出问题时可以通过 worktree、feature flag 或分支回退快速恢复。
5. 作为前端或 API 调用方，我希望 `/chat`、`/chat/resume` 和 SSE 事件在迁移期间保持兼容。

## 7. 成功指标

| 指标 | 目标 |
| --- | --- |
| API 兼容性 | `/chat`、`/chat/resume` 请求与响应 schema 无破坏性变化。 |
| SSE 兼容性 | UI 安全事件集不变，不泄露工具参数、历史窗口或 raw trace。 |
| HITL 安全性 | stale checkpoint、terminal resume、approve 幂等、reject cancel 全部通过测试。 |
| RAG 可观测性 | `citations`、`retrieval_trace`、`final_decision` 字段不回退。 |
| 回归测试 | baseline / candidate 两分支核心测试均通过，关键样本输出可解释差异为 0 或已登记。 |
| 可回滚性 | 每阶段至少支持 feature flag 回滚或 worktree 分支回退。 |
| 代码维护收益 | 完成 ReAct legacy 清理后，预期净减少 500-1200 行，长期减少 1000-2000 行维护负担。 |

## 8. 当前实现盘点

| 模块 | 当前行数级别 | 迁移策略 |
| --- | ---: | --- |
| `backend/platform/agent_runtime/react/` | 约 2027 行 | 优先迁移。用 LangChain `create_agent` + middleware 替代自研 ReAct loop。 |
| `backend/platform/agent_runtime/plan/` | 约 1358 行 | 暂时保留。Plan 是显式多步依赖执行，不能简单压进 agent loop。 |
| `backend/platform/agent_runtime/chat_graph/` | 约 2323 行 | 保留并现代化。它是外层业务编排与 checkpoint 边界。 |
| `backend/platform/rag/orchestration/retrieval_graph/` | 约 938 行 | 保留并现代化。Agentic RAG 是工具内部子图。 |
| `backend/application/runtime/assembly/service_parts/` | 约 2667 行 | 分阶段适配 projection、trace、answer mode 和 middleware runtime facade。 |
| `backend/platform/agent_runtime/tool_executor.py` | 约 220 行 | 先保留为兼容层，再逐步把校验、审计、重试迁到 `wrap_tool_call`。 |
| `backend/platform/models/llm/client.py` | 约 230 行 | 保留模型路由，新增 middleware 适配入口。 |

## 9. 核心需求

### FR-1：迁移过程必须支持 worktree 双轨开发

需求：

- 必须使用 Git worktree 创建独立迁移工作树，不在主工作区直接推进大规模实验改造。
- 主工作区保留当前 baseline 或用户已有工作状态。
- candidate 工作树用于 middleware / LangGraph 现代化实现。
- baseline 与 candidate 可以同时安装依赖、运行测试、启动服务和生成评测结果。

建议目录：

```text
d:\Programs\interview-projects\
├─ ai-rag-project\                 # baseline / 当前主工作区
└─ ai-rag-project-langchain-mw\     # candidate worktree
```

建议命令：

```powershell
git branch migrate/langchain-middleware
git worktree add ..\ai-rag-project-langchain-mw migrate/langchain-middleware
```

验收：

- `git worktree list` 能看到 baseline 和 candidate 两个工作树。
- candidate 分支可独立运行测试。
- baseline 工作树不被 candidate 改造污染。

### FR-2：迁移过程必须支持两分支并行测试和结果对比

需求：

- baseline 和 candidate 必须运行同一组测试命令。
- 对于 eval / HTTP replay / SSE replay，必须生成可比较 artifact。
- candidate 输出可以与 baseline 有差异，但差异必须分类：
  - 预期差异：例如 trace metadata 多了 middleware 字段。
  - 非预期差异：例如 citation 丢失、状态机错误、SSE 事件缺失。
- 每阶段合并前必须提交对比摘要。

建议对比流程：

```powershell
# baseline 工作树
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agent_runtime_react.py backend\tests\test_agent_runtime_tools.py -q -c backend\tests\pytest.ini

# candidate worktree
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agent_runtime_react.py backend\tests\test_agent_runtime_tools.py -q -c backend\tests\pytest.ini
```

建议 artifact：

```text
backend/tests/artifacts/migration-baseline/
backend/tests/artifacts/migration-candidate/
backend/tests/artifacts/migration-diff/
```

验收：

- baseline / candidate 测试命令均记录结果。
- candidate 所有失败都有归因。
- `/chat`、`/chat/resume`、SSE、RAG trace 的关键输出差异有摘要。

### FR-3：所有行为变化必须由 feature flag 控制

需求：

- 新 provider、新 middleware、新 stream mapper 默认不得一次性替代 legacy。
- 每个阶段至少提供一种快速关闭方式。

建议配置：

```env
AI_RAG_AGENT_RUNTIME__REACT_PROVIDER=legacy
AI_RAG_AGENT_RUNTIME__ENABLE_MIDDLEWARE_GUARDS=false
AI_RAG_LANGGRAPH__STREAM_VERSION=v1
AI_RAG_LANGGRAPH__GRAPH_OUTPUT_MODE=legacy
```

验收：

- 默认配置下行为不变。
- candidate 可以通过配置切回 legacy。
- 测试覆盖 legacy 与 candidate provider 的核心路径。

### FR-4：LangChain middleware 基础设施

需求：

新增平台层 middleware 包：

```text
backend/platform/agent_runtime/middleware/
├─ __init__.py
├─ context.py
├─ model_guard.py
├─ tool_policy.py
├─ tool_observation.py
├─ dynamic_prompt.py
├─ hitl_gate.py
├─ trace.py
└─ factory.py
```

建议 middleware 职责：

| Middleware | LangChain hook | 作用 |
| --- | --- | --- |
| `RuntimeContextMiddleware` | `before_agent` / `after_agent` | 注入 `session_id`、`request_id`、scene、mounted knowledge sources、run metadata。 |
| `DynamicScenePromptMiddleware` | `dynamic_prompt` 或 `wrap_model_call` | 根据 scene、history、resume metadata 动态生成 system prompt。 |
| `ModelGuardMiddleware` | `wrap_model_call` | 统一空输出、模型失败分类、重试、provider fallback 入口。 |
| `ToolPolicyMiddleware` | `wrap_model_call` / `wrap_tool_call` | 动态收窄工具集合，执行 allowed tools、high risk tools、schema guard。 |
| `ToolObservationMiddleware` | `wrap_tool_call` | 将 LangChain `ToolMessage` / tool output 转回项目 `ToolObservation`。 |
| `HitlGateMiddleware` | `wrap_tool_call` | 高风险工具执行前触发 HITL 等待，不直接执行副作用。 |
| `RuntimeTraceMiddleware` | `before_model` / `after_model` / `wrap_tool_call` | 记录模型调用、工具调用、latency、trace metadata。 |

验收：

- middleware 可单测。
- legacy runtime 可复用 helper，但默认行为不变。
- 工具输出仍可归一化为 `ToolObservation`。

### FR-5：ReAct provider 双轨

需求：

- 新增 LangChain agent provider。
- 保留 legacy ReAct provider。
- 通过 `AI_RAG_AGENT_RUNTIME__REACT_PROVIDER` 选择 provider。

新增目录：

```text
backend/platform/agent_runtime/react_langchain/
├─ __init__.py
├─ factory.py
├─ runtime.py
├─ state.py
├─ projection.py
└─ tools.py
```

核心设计：

- 使用 `langchain.agents.create_agent` 创建 LangChain agent。
- agent model 使用 `ModelClient.get_chat_model(complexity)`。
- tools 来自 scene tools + RAG adapters，但通过 middleware 动态收窄。
- `react_langchain.runtime` 输出仍投影为项目现有 `ReActRun` 或兼容结构。

验收：

- legacy provider 行为不变。
- langchain provider 可完成基础 RAG 工具问答。
- 新 provider 输出可投影为 `documents`、`citations`、`retrieval_trace`、`tool_event`、`final_decision`、`answer_mode`。

### FR-6：LangGraph 最新特性现代化

需求：

- `ChatGraphRuntime.invoke` 兼容 `GraphOutput.value` / `.interrupts` 与旧 dict 输出。
- 内部逐步支持 typed streaming。
- HITL 优先使用 `interrupt()` / `Command(resume=...)` 语义，但必须保留当前项目自己的 resume 校验。
- subgraph persistence 使用方式形成项目规范。

验收：

- 同步 `/chat` 输出不变。
- `/chat?stream=true` UI 协议不变。
- checkpoint 最新等待点读取逻辑不变。
- `GraphOutput` / typed stream / `Command(resume=...)` 有测试覆盖。

### FR-7：HITL 安全语义必须保留

需求：

- `waiting_user` 是等待人工输入，不是失败。
- `reject` 或 cancel 进入 `cancelled`。
- `approve` 必须先消费等待点，再执行工具副作用。
- `/chat/resume` 必须校验最新 checkpoint 的 `interrupt_id`。
- ReAct / Plan HITL 恢复必须回到所属图节点继续执行。

验收：

- stale `interrupt_id` 被拒绝。
- terminal checkpoint resume 被拒绝。
- approve 只执行一次工具。
- reject 不执行工具。
- respond 能回到同一 run 消费用户补充信息。

### FR-8：SSE 与可观测性对齐

需求：

- 内部可以使用 LangGraph / LangChain typed events。
- 外部继续提供稳定 UI SSE 协议。
- 不向 UI SSE 泄露历史窗口、工具参数或 retrieval raw trace。

事件映射：

| 内部事件 | UI SSE |
| --- | --- |
| graph start | `start` |
| model chunk | `chunk` |
| safe thinking | `thinking` |
| interrupt | `waiting_user` |
| graph output | `done` |
| error | `error` |

验收：

- stream 和 non-stream 最终语义一致。
- eval replay 可对比 stream / non-stream。

## 10. 分阶段交付计划

### 阶段 0：基线审计、worktree 与保护网

改动：

- 建立 candidate worktree。
- 增加 runtime provider 配置占位，默认 legacy。
- 固定 baseline 测试命令和 artifact 目录。
- 补充 characterization tests。

验证：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_generic_assistant_hitl.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agent_runtime_react.py backend\tests\test_agent_runtime_plan.py backend\tests\test_agent_runtime_tools.py -q -c backend\tests\pytest.ini
```

### 阶段 1：middleware 基础设施

改动：

- 新增 `backend/platform/agent_runtime/middleware/`。
- 抽出 model guard、tool policy、tool observation、trace helper。
- legacy runtime 可以选择性复用 helper，但默认行为不变。

验证：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agent_runtime_tools.py backend\tests\test_model_guards.py -q -c backend\tests\pytest.ini
```

### 阶段 2：LangGraph API 现代化第一步

改动：

- `ChatGraphRuntime.invoke` 兼容 `GraphOutput`。
- 建立 typed stream mapper 内部适配层。
- 保留现有 UI SSE 协议。

验证：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

### 阶段 3：ReAct provider 双轨

改动：

- 新增 `react_langchain` provider。
- `react_branch` 按配置选择 legacy 或 langchain provider。
- 新增 provider contract tests。

验证：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agent_runtime_react.py backend\tests\test_agent_runtime_tools.py -q -c backend\tests\pytest.ini
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agentic_retrieval.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

### 阶段 4：ReAct HITL 迁移

改动：

- `HitlGateMiddleware` 接入高风险工具审批。
- LangGraph interrupt / resume 语义接入 ReAct provider。
- 保留 `ChatGraphRuntime.resume_hitl` 统一入口。

验证：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_generic_assistant_hitl.py backend\tests\test_agent_runtime_react.py -q -c backend\tests\pytest.ini
```

### 阶段 5：SSE 与可观测性对齐

改动：

- 内部 typed events 转 UI SSE。
- token、latency、tool count、retry count 写入 metadata 或 eval artifact。

验证：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py backend\tests\test_eval_assets.py -q -c backend\tests\pytest.ini
```

### 阶段 6：Agentic RAG 子图现代化

改动：

- `build_agentic_rag_graph` 输出适配 `GraphOutput`。
- query rewrite、sufficiency check、final evidence synthesis 接入共享 model guard。
- 检索 trace 字段保持兼容。

验证：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agentic_retrieval.py backend\tests\test_document_hybrid_retrieval.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

### 阶段 7：Plan 现代化

改动：

- `MinimalPlanner` 模型调用接入共享 guard。
- Plan step 工具执行逐步复用 middleware helper。
- Plan HITL resume 适配 LangGraph interrupt / Command 语义。

验证：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agent_runtime_plan.py backend\tests\test_langgraph_runtime.py -q -c backend\tests\pytest.ini
```

### 阶段 8：ReAct legacy 删除候选

删除候选：

- `backend/platform/agent_runtime/react/runtime.py`
- `backend/platform/agent_runtime/react/selector.py`
- `backend/platform/agent_runtime/react/tool_turns.py`
- `backend/platform/agent_runtime/react/graph/nodes/select_action.py`
- `backend/platform/agent_runtime/react/graph/nodes/validate_action.py`
- `backend/platform/agent_runtime/react/graph/nodes/route_action.py`
- `backend/platform/agent_runtime/react/graph/nodes/execute_tool.py`
- `backend/platform/agent_runtime/react/graph/nodes/record_observation.py`
- `backend/platform/agent_runtime/react/graph/nodes/loop_or_finish.py`

验收：

- 默认 provider 切到 `langchain`。
- legacy provider 保留至少一个短期回滚窗口。
- 两轮主要 eval 通过后再真正删除 legacy。

### 阶段 9：文档、图表和配置清理

更新：

- `README.md`
- `AGENTS.md`
- `backend/.env.example`
- `docs/documents/runtime/react-plan-agent-runtime.md`
- `docs/documents/runtime/main-chat-agent-runtime-flow.svg`
- `docs/documents/runtime/chatgraph-subgraphs.svg`
- `docs/documents/reference/api-list.md`
- `docs/documents/reference/data-model.md`

验收：

- 文档描述和代码默认 provider 一致。
- API 文档中 `/chat`、`/chat/resume` 字段无漂移。

## 11. 工作树开发规范

### 11.1 分支与 worktree

推荐分支：

```text
main 或 当前基线分支
migrate/langchain-middleware
```

推荐 worktree：

```powershell
git branch migrate/langchain-middleware
git worktree add ..\ai-rag-project-langchain-mw migrate/langchain-middleware
git worktree list
```

规则：

- baseline 工作树只做对照测试，不做迁移实验。
- candidate worktree 承载所有迁移代码。
- 每阶段完成后在 candidate 分支提交。
- 合并前必须记录 baseline / candidate 对比结果。

### 11.2 双分支测试矩阵

| 测试组 | baseline | candidate | 对比要求 |
| --- | --- | --- | --- |
| LangGraph runtime | 必跑 | 必跑 | 状态机、checkpoint、HITL 结果一致。 |
| ReAct runtime | 必跑 | 必跑 | legacy 阶段一致；candidate provider 差异需登记。 |
| Plan runtime | 必跑 | 必跑 | 计划、重试、HITL 语义一致。 |
| RAG retrieval | 必跑 | 必跑 | citation、trace、final decision 一致。 |
| Chat API / SSE | 必跑 | 必跑 | API schema 与 UI SSE 事件一致。 |
| Eval replay | 建议 | 建议 | 样本通过率、引用命中、stream 语义对比。 |

### 11.3 对比报告模板

每阶段完成后记录：

```text
阶段：
baseline commit：
candidate commit：
测试命令：
baseline 结果：
candidate 结果：
API 差异：
SSE 差异：
HITL 差异：
RAG trace / citation 差异：
预期差异：
非预期差异：
是否允许进入下一阶段：
```

## 12. 回滚策略

| 层级 | 回滚方式 |
| --- | --- |
| worktree | 删除 candidate worktree 或切回 baseline 工作树。 |
| Git 分支 | 停止合并 `migrate/langchain-middleware`，继续使用 baseline 分支。 |
| feature flag | `REACT_PROVIDER=legacy`、关闭 middleware guards、回退 stream version。 |
| runtime provider | 保留 legacy ReAct provider 至少一个短期窗口。 |
| checkpoint | 保持业务状态投影兼容，不把 LangChain 内部结构作为唯一恢复依据。 |

每阶段回滚点：

| 阶段 | 回滚方式 |
| --- | --- |
| 阶段 1 | 关闭 `ENABLE_MIDDLEWARE_GUARDS`，legacy runtime 不使用新 middleware。 |
| 阶段 2 | 回退到 dict-style graph invoke 和当前 stream mapper。 |
| 阶段 3 | `REACT_PROVIDER=legacy`。 |
| 阶段 4 | 关闭 ReAct HITL middleware，恢复 legacy ReAct resume graph。 |
| 阶段 6 | Agentic RAG 继续使用当前 `graph.invoke` 和 trace mapper。 |
| 阶段 7 | Plan 继续使用 `PlanExecutor` 当前工具执行路径。 |

## 13. 风险

1. LangChain `create_agent` 的内部消息和工具调用结构与当前 `ReActRun` 不完全一致，projection 需要谨慎。
2. 内置 `HumanInTheLoopMiddleware` 不能直接替代当前 HITL 安全语义，必须包装项目自己的 `interrupt_id` 和副作用幂等规则。
3. LangGraph typed streaming / `GraphOutput` 新旧 API 并存期间，测试需要覆盖两种返回格式。
4. 工具输出从 `ToolObservation` 到 LangChain `ToolMessage` 再回投影，可能丢失 citation / retrieval trace，需要专门测试。
5. 双轨期代码量会上升，阶段管理和 feature flag 命名必须清晰。
6. worktree 双分支测试会增加本地环境维护成本，需要统一 `.env`、数据库和 artifact 隔离策略。

## 14. 预计代码量变化

| 阶段 | 新增 | 删除 | 净变化 |
| --- | ---: | ---: | ---: |
| 阶段 1 | 600-1000 行 | 0-100 行 | 增加 |
| 阶段 2 | 300-600 行 | 100-300 行 | 小幅增加或持平 |
| 阶段 3 | 800-1500 行 | 0-300 行 | 增加 |
| 阶段 4 | 400-900 行 | 200-600 行 | 持平 |
| 阶段 8 | 0-200 行 | 1000-1800 行 | 明显减少 |
| 阶段 7 | 300-700 行 | 100-400 行 | 小幅增加或持平 |
| 阶段 6 | 200-500 行 | 100-300 行 | 持平 |

整体预期：

- 迁移中期代码会先增加约 1500-2500 行，因为会存在双轨和 adapter。
- 完成 ReAct legacy 清理后，预计净减少约 500-1200 行。
- 如果后续 Plan 的工具执行和模型 guard 也充分复用 middleware，长期可减少约 1000-2000 行维护负担。

## 15. 最终验收清单

- [ ] 已使用 Git worktree 建立 candidate 工作树。
- [ ] baseline / candidate 两分支可并行运行核心测试。
- [ ] 每阶段有测试对比报告。
- [ ] 默认 runtime 使用 LangChain middleware 化 ReAct provider。
- [ ] `ChatGraph`、`Plan`、`Agentic RAG` 仍由 LangGraph 显式编排。
- [ ] `/chat` 和 `/chat/resume` API schema 无破坏性变化。
- [ ] SSE UI 协议无破坏性变化。
- [ ] HITL stale checkpoint、terminal resume、approve 幂等、reject cancel 全部通过测试。
- [ ] RAG citation 和 retrieval trace 字段无回退。
- [ ] LangGraph `GraphOutput` / typed stream / `Command(resume=...)` 路径有测试覆盖。
- [ ] README、AGENTS、runtime 文档、API 文档、图表均同步。

## 16. 第一批任务

1. 创建 `migrate/langchain-middleware` 分支和 candidate worktree。
2. 固化 baseline / candidate 测试矩阵与 artifact 输出目录。
3. 新增 agent runtime 配置项与默认 legacy provider。
4. 新增 `backend/platform/agent_runtime/middleware/` 包和 middleware 单测。
5. 把 `ModelClient` 的 guard 能力抽成可被 middleware 复用的 adapter。
6. 为 `ToolExecutor` 增加 LangChain tool list 导出或 adapter，不改变现有执行路径。
7. 在 `ChatGraphRuntime.invoke` 中兼容 `GraphOutput.value` / `.interrupts`，但默认行为不变。
8. 增加一组 ReAct provider contract tests，为双轨迁移做准备。

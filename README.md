# AI RAG Project

面向企业知识问答与场景化智能助手的 **Agentic RAG / Agent Runtime** 示例项目。它不是单 Prompt Demo，而是一套可运行、可观测、可评测、可扩展的 RAG 应用底座。

项目已经打通知识文档入库、Hybrid Search、Agentic Retrieval、引用溯源、检索 Trace、SSE 流式输出、Human-in-the-Loop、Workflow State Machine、Evaluation Harness 和多场景扩展，适合用作 RAG 项目作品集、面试讲解样例或二次开发起点。

## 核心能力

- **统一对话入口**：`/chat` 按会话 scene 和 mounted knowledge sources 动态选择检索工具，支持普通 JSON 和 SSE 流式输出。
- **Agentic RAG 主链路**：先做 query rewrite 和工具决策，再多轮检索、判断证据是否足够，最后生成带引用的回答。
- **Hybrid Search**：文档检索支持语义召回、BM25 关键词召回、融合排序、相关性过滤和 no-hit fallback。
- **引用与可观测性**：回答返回结构化 `citations`、正文引用编号、`retrieval_trace`、rerank trace 和 SSE `tool / waiting_user / done` 事件。
- **Human-in-the-Loop**：支持澄清等待、工具审批、外部 API 审批，以及 `approve / reject / respond` 恢复。
- **Workflow State Machine**：运行时状态统一为 `created / planning / running / waiting_user / retrying / succeeded / failed / cancelled`，终态防重复恢复。
- **Knowledge Admin**：支持文件上传、预处理预览、正式入库、重处理、重切块、软删除和索引状态查看。
- **Evaluation Harness**：支持 minimal 与 retrieval benchmark 回放、baseline / candidate 对比和评测看板。
- **多场景扩展**：`generic_assistant` 是通用知识问答主线，`ecommerce` 是业务扩展示例，后端按 `platform / application / scenes` 分层。

## 项目截图

### 对话工作台

![对话工作台](./docs/documents/assets/images/api-tester-ui.png)

### 知识库管理

![知识库管理](./docs/documents/assets/images/knowledge-manager-ui.png)

## 架构概览

```text
backend/
├─ application/        # FastAPI runtime、API 路由、服务装配
├─ platform/           # 配置、模型、记忆、知识处理、RAG、工具协议、Workflow Runtime
└─ scenes/             # generic_assistant、ecommerce 等场景定义

frontend/              # 对话工作台、知识管理、评测看板
docs/documents/        # 面向阅读和 LLM 检索的分模块文档
openspec/              # 变更提案、规格与归档记录
```

核心链路：

```text
用户问题
  -> 会话与场景解析
  -> Query Rewrite
  -> Agentic Retrieval Tool Decision
  -> Hybrid Search / Business Tool
  -> 低相关过滤 / no-hit fallback / HITL interrupt
  -> LLM Answer
  -> citations + retrieval_trace + workflow state
```

架构图和明细文档见：[文档索引](./docs/documents/README.md)。

## 当前进展与后续计划

### 已完成主线

- [x] 三层后端结构：`platform / application / scenes`
- [x] 统一 `/chat`、`/chat/resume`、`/sessions`、`/files`、`/knowledge/documents` API
- [x] 会话级 `mounted_knowledge_sources` 挂载与 scene definition 候选工具解析
- [x] Agentic Retrieval：query rewrite、工具决策、多轮检索、no-hit fallback 与结构化 trace
- [x] 文档 Hybrid Search：语义召回 + 关键词召回 + 融合排序
- [x] 模型路由：LLM、Embedding、ReRank 统一配置
- [x] 结构化 citations、回答正文引用编号与 session 证据持久化
- [x] `/chat` 与 SSE 暴露 retrieval trace、runtime state 和 final state
- [x] Knowledge Admin：上传、预览、入库、重处理、重分块、软删除
- [x] Evaluation Harness：HTTP replay、SSE replay、benchmark artifact 对比
- [x] LangGraph Runtime 骨架：graph state、thread_id、checkpointer、stream event 映射
- [x] Human-in-the-Loop：interrupt/resume、工具审批、澄清等待、reject/cancel 边界
- [x] Workflow State Machine：状态枚举、合法转移、终态保护和 SSE/API 状态字段

### P0：把 RAG Runtime 升级为真正的 Agent Runtime

- [x] Runtime 边界修正：严格消费 `AgenticRetrievalOutcome.success`、`final_decision` 和 `follow_up_question`，确保 `ask_user` / `max_rounds_reached` 不误入证据回答链。
- [x] 请求上下文隔离：移除 `ChatService` 中 per-request mutable state，避免并发请求串写 `request_id`、时间戳和历史消息元数据。
- [x] LangGraph Runtime 骨架：接入 graph state、`thread_id`、checkpointer、stream event 映射和 graph run 生命周期管理。
- [x] Human-in-the-Loop：基于 LangGraph interrupt/resume 支持 `approve / reject / respond`，覆盖 generic 写操作测试工具、外部 API 测试工具和 `ask_user` 澄清场景；`edit` 保留协议占位。
- [x] Workflow State Machine：基于 LangGraph 节点和持久化状态表达 `created / planning / running / waiting_user / retrying / succeeded / failed / cancelled`。
- [ ] Planner / Executor：支持计划生成、步骤拆解、工具调用链执行、步骤结果沉淀和最终汇总，并保留人工介入点。
- [ ] Agentic RAG Subgraph：将 `AgenticRetriever` 中手写的 `while` 循环、`next_action` 路由、query rewrite、工具切换、rerank、充分性判断和 no-hit fallback 迁移为可复用 LangGraph 子图。
- [ ] LangChain / LangGraph 重构审计：逐项识别当前自造状态机、streaming glue、history glue、tool routing glue，能用 LangGraph graph / node / conditional edge / interrupt 表达的优先迁移。
- [ ] Business Handoff Subgraph：将 `generic_assistant` 到 `ecommerce` 的 handoff / followup 逻辑从 scene 内部判断迁移为 LangGraph router 或业务子图。
- [ ] Failure Recovery：为工具调用、模型调用和长链路任务补齐超时、重试、失败补偿、可恢复执行和幂等控制。
- [ ] Reflection / Critique：在多步任务中加入结果校验、失败原因归类和必要时的自我修正。

### P1：平台化 Tool、Memory 和 Evaluation 能力

- [ ] Tool Registry 平台化：统一工具注册、参数 schema、权限声明、Agent 白名单、MCP 暴露标记和运行结果协议，并保持与 LangChain `BaseTool` / `StructuredTool` 兼容。
- [ ] Tool 协议收敛：scene 工具以 LangChain `BaseTool` / `StructuredTool` 为主协议，自研 `ToolResult` 仅保留为业务 payload，逐步移除重复的 `RetrievalToolAdapter` / `RetrievalTool` 编排层协议。
- [ ] Tool Routing 重构：将当前 scene definition 和 `AgenticRetriever` 中分散的候选工具解析、白名单和切换逻辑，收敛到 Tool Registry + LangGraph 条件边。
- [ ] Tool Audit：记录工具调用输入摘要、输出摘要、耗时、错误类型、重试次数和权限判定结果。
- [ ] Memory 升级：从短窗口历史扩展到任务状态记忆、用户偏好、长期摘要和跨会话上下文；会话历史继续兼容 LangChain message history，长任务状态交给 LangGraph checkpointer。
- [ ] Query Rewrite 标准化：将手写 JSON 解析迁移到 LangChain structured output / output parser，继续保留关键 token 保护、unsafe rewrite 校验和 fallback 策略。
- [ ] Streaming 重构：将当前 `ChatStreamEvent` / SSE 手写事件与 LangChain callback、LangGraph stream events 对齐，统一输出 token、tool、interrupt、resume、done 和 error 事件。
- [ ] Workflow Evaluation：在现有 RAG 评测外，增加任务完成率、工具成功率、步骤失败率、恢复成功率和人工/LLM judge 评分。
- [ ] Eval 标准化：保留 HTTP replay + qrels 指标，同时接入 LLM-as-a-judge / LangSmith 风格数据集与实验对比，用于生成质量和多步任务质量评测。
- [ ] Cost & Latency Metrics：沉淀 token、模型成本、P50/P95 延迟、检索耗时、rerank 耗时和端到端耗时。
- [ ] 场景扩展模板：提供新增 scene / workflow / tool / eval sample 的标准目录、接口约束和测试样例。

### P1：继续强化 RAG 与知识库工程化

- [ ] 知识库批量重建索引、失败重试、索引状态诊断和上传文件清理。
- [ ] 增量更新与缓存：支持文档增量索引、检索结果缓存、Embedding 缓存和缓存失效策略。
- [ ] Retrieval Benchmark 扩充：扩大 qrels 样本，稳定 Precision / Recall / MRR / NDCG / no-hit false positive 指标。
- [ ] 关键词召回扩展：为当前小规模 BM25 实现预留持久化倒排索引或外部搜索引擎边界。
- [ ] 检索诊断 UI：展示 query rewrite、候选召回、过滤、rerank、citation 对齐和失败原因。

### P2：面向开源与生产部署补齐工程闭环

- [ ] 权限与安全：用户、角色、知识库权限、工具调用权限、API Key、敏感信息过滤和审计日志。
- [ ] 可观测性：结构化日志、OpenTelemetry Trace、模型调用 trace、工具调用 trace 和 eval run trace。
- [ ] 部署能力：Docker Compose、生产配置样例、健康检查、备份恢复和日志采集。
- [ ] 模型治理：模型路由策略、降级策略、超时重试、成本预算、调用审计和 provider fallback。
- [ ] 开源文档：补齐架构决策记录、插件开发指南、Workflow DSL 示例、贡献指南和 Roadmap。
- [ ] 企业集成：预留 SSO、对象存储、外部知识源、工单系统、CRM、内部搜索或支付/订单系统扩展点。

## Quick Start

示例使用 PowerShell，命令默认在仓库根目录执行。

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
```

在 `backend\.env` 中配置模型 Key：

```env
AI_RAG_MODELS__SIMPLE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__MODERATE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__COMPLEX__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__EMBEDDING__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__RERANK__API_KEY=your-dashscope-api-key
AI_RAG_APP__ACTIVE_SCENE=generic_assistant
AI_RAG_VECTOR_STORE__PROVIDER=chroma
```

启动后端：

```powershell
backend\.venv\Scripts\python.exe backend\run.py
```

访问入口：

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- 对话工作台: `http://127.0.0.1:8000/frontend/api-tester.html`
- 知识库管理: `http://127.0.0.1:8000/frontend/knowledge-manager.html`
- 评测看板: `http://127.0.0.1:8000/frontend/eval-dashboard.html`

## 常用接口

- `POST /chat`：统一对话入口，支持 `stream=true` SSE。
- `POST /chat/resume`：恢复 HITL 等待点，支持 `approve / reject / respond`。
- `GET /sessions` / `POST /sessions`：会话创建、查询与场景绑定。
- `POST /files/upload`：上传知识文件。
- `POST /knowledge/documents/preprocess-preview`：知识文件预处理预览。
- `POST /knowledge/documents`：确认入库并发布索引。
- `GET /evals/latest` / `POST /evals/runs`：读取或触发评测回放。

接口和数据模型明细见：[API 文档](./docs/documents/reference/api-list.md)、[数据模型](./docs/documents/reference/data-model.md)。

## 测试与评测

运行后端测试：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

常用回归：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agentic_retrieval.py backend\tests\test_chat_api.py backend\tests\test_document_hybrid_retrieval.py -q -c backend\tests\pytest.ini
```

运行评测：

```powershell
backend\.venv\Scripts\python.exe backend\evals\run_http_eval.py --base-url http://127.0.0.1:8000 --sample-set minimal --output backend\data\evals\latest.json
```

## 文档入口

- [文档总索引](./docs/documents/README.md)
- [系统架构图](./docs/documents/architecture/system-overview.svg)
- [LangGraph Runtime 图](./docs/documents/architecture/langgraph-runtime-current.svg)
- [知识管理流程图](./docs/documents/knowledge/knowledge-document-flow.svg)
- [Agentic RAG 流程图](./docs/documents/rag/agentic-rag-retrieval-flow.svg)
- [Agentic RAG 设计说明](./docs/documents/rag/agentic-rag.md)
- [常见坑与排障](./docs/documents/operations/common-pitfalls.md)

## 项目定位

这个项目的重点不是“能回答一句话”，而是展示一套可解释、可回归、可继续扩展的 Agentic RAG Runtime：回答有引用，检索有 Trace，状态有治理，人工介入有边界，调参有评测，场景能扩展，知识能管理。

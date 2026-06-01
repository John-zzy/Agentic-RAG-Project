# AI RAG Project

面向企业知识问答与场景化智能助手的 **Agentic RAG Runtime**。它不是一个单 Prompt Demo，而是一套可运行、可观测、可评测、可扩展的 RAG 应用底座。

项目围绕真实工程链路设计：知识文档入库、Hybrid Search、Agentic Retrieval、引用溯源、检索 Trace、SSE 流式输出、Evaluation Harness 和多场景扩展都已经打通，适合用作 RAG 项目作品集、面试讲解样例或二次开发起点。

## Highlights

- **Agentic RAG 主链路**：统一 `/chat` 入口，按会话 scene 和 mounted knowledge sources 动态选择检索工具。
- **Hybrid Search**：文档检索支持语义召回、BM25 关键词召回与融合排序，可按 scene policy 控制 `top_k`、阈值和 no-hit fallback。
- **模型路由与 ReRank**：LLM、Embedding、ReRank 模型统一由配置路由，检索结果支持真实重排和 trace 观测。
- **Retrieval Trace 可观测性**：单次请求可看到 query rewrite、tool decision、候选数量、过滤数量、top chunk score、citations 与 `knowledge_used`。
- **Knowledge Admin**：支持文件上传、预处理预览、正式入库、重处理、重切块、软删除和索引状态查看。
- **Evaluation Harness**：内置 minimal 与 retrieval benchmark 回放，支持 baseline / candidate 对比，便于量化调参效果。
- **多场景扩展**：`generic_assistant` 作为通用知识问答主线，`ecommerce` 作为业务扩展示例，架构按 `platform / application / scenes` 分层。
- **可运行前端**：提供对话工作台、知识库管理页和评测看板，不只是后端接口集合。

## 当前进展与后续开发计划

本节是统一功能清单：所有功能项都在这里维护，已完成标记为 `[x]`，待完成标记为 `[ ]`。当以下所有功能均为 `[x]` 时，即表示本项目开发完成。

项目下一阶段的目标是从 **Agentic RAG Runtime** 演进为可开源复用的 **Agent Runtime / Workflow 平台**：既能展示 RAG 深度优化，也能展示 Planning、Memory、Tool Use、Human-in-the-Loop、Evaluation Harness、可观测和稳定性工程这些 Agent 岗位高频能力。LangGraph 会在 P0 阶段作为 Workflow Runtime 底座引入，用于承载状态持久化、interrupt/resume、人审节点和可组合子图，而不是等到后期作为可选重构。

### 平台底座与 RAG 主链

- [x] 三层后端结构：`platform / application / scenes`
- [x] 统一 `/chat`、`/sessions`、`/files`、`/knowledge/documents` API
- [x] `generic_assistant` 独立 docs-first 检索链路
- [x] 场景扩展雏形：`generic_assistant` 主线与 `ecommerce` 业务扩展示例按 `platform / application / scenes` 分层
- [x] 会话级 `mounted_knowledge_sources` 挂载与 scene definition 候选工具解析
- [x] Agentic Retrieval 主链：query rewrite、工具决策、多轮检索、no-hit fallback 与结构化 `retrieval_trace`
- [x] 文档 `Hybrid Search`：语义召回 + 关键词召回 + 融合排序
- [x] 模型路由配置：统一管理 LLM、Embedding 与 ReRank 模型参数
- [x] DashScope Embedding 接入：区分 query embedding 与 document embedding
- [x] ReRank 实际接入：支持检索结果重排、分数回填和未启用时的分数字段清理
- [x] scene retrieval policy：控制 `top_k`、相关性阈值、召回策略、no-hit 策略和 ReRank 开关
- [x] no-hit fallback：无明确文档意图或过滤后无证据时返回 `knowledge_used=false` 与空 citations
- [x] 结构化 `citations`、回答正文引用编号与 session `retrieval_snippets` 持久化
- [x] `/chat` 与 SSE 暴露结构化 `retrieval_trace`，包含 rerank 执行状态和候选数量变化
- [x] Evaluation Harness 覆盖 minimal 回放、SSE 回放、baseline / candidate artifact 对比
- [x] 知识文档预处理预览、注册、重处理、重分块、软删除与文件维度索引视图
- [x] `Chroma` / `Elasticsearch` 可切换向量存储，`SQLite` 会话持久化

### P0：把 RAG Runtime 升级为真正的 Agent Runtime

- [x] Runtime 边界修正：严格消费 `AgenticRetrievalOutcome.success`、`final_decision` 和 `follow_up_question`，确保 `ask_user` / `max_rounds_reached` 不误入证据回答链
- [x] 请求上下文隔离：移除 `ChatService` 中 per-request mutable state，避免并发请求串写 `request_id`、时间戳和历史消息元数据
- [x] LangGraph Runtime 骨架：接入 graph state、`thread_id`、checkpointer、stream event 映射和 graph run 生命周期管理
- [ ] Human-in-the-Loop：基于 LangGraph interrupt/resume 支持 `approve / edit / reject / respond`，优先覆盖写操作工具、外部 API 调用和 `ask_user` 澄清场景
- [ ] Workflow State Machine：基于 LangGraph 节点和持久化状态表达 `created / planning / running / waiting_user / retrying / succeeded / failed / cancelled`
- [ ] Planner / Executor：支持计划生成、步骤拆解、工具调用链执行、步骤结果沉淀和最终汇总，并保留人工介入点
- [ ] Agentic RAG Subgraph：将 `AgenticRetriever` 中手写的 `while` 循环、`next_action` 路由、query rewrite、工具切换、rerank、充分性判断和 no-hit fallback 迁移为可复用 LangGraph 子图
- [ ] LangChain / LangGraph 重构审计：逐项识别当前自造状态机、streaming glue、history glue、tool routing glue，能用 LangGraph graph / node / conditional edge / interrupt 表达的优先迁移
- [ ] Business Handoff Subgraph：将 `generic_assistant` 到 `ecommerce` 的 handoff / followup 逻辑从 scene 内部判断迁移为 LangGraph router 或业务子图
- [ ] Failure Recovery：为工具调用、模型调用和长链路任务补齐超时、重试、失败补偿、可恢复执行和幂等控制
- [ ] Reflection / Critique：在多步任务中加入结果校验、失败原因归类和必要时的自我修正

### P1：平台化 Tool、Memory 和 Evaluation 能力

- [ ] Tool Registry 平台化：统一工具注册、参数 schema、权限声明、Agent 白名单、MCP 暴露标记和运行结果协议，并保持与 LangChain `BaseTool` / `StructuredTool` 兼容
- [ ] Tool 协议收敛：scene 工具以 LangChain `BaseTool` / `StructuredTool` 为主协议，自研 `ToolResult` 仅保留为业务 payload，逐步移除重复的 `RetrievalToolAdapter` / `RetrievalTool` 编排层协议
- [ ] Tool Routing 重构：将当前 scene definition 和 `AgenticRetriever` 中分散的候选工具解析、白名单和切换逻辑，收敛到 Tool Registry + LangGraph 条件边
- [ ] Tool Audit：记录工具调用输入摘要、输出摘要、耗时、错误类型、重试次数和权限判定结果
- [ ] Memory 升级：从短窗口历史扩展到任务状态记忆、用户偏好、长期摘要和跨会话上下文；会话历史继续兼容 LangChain message history，长任务状态交给 LangGraph checkpointer
- [ ] Query Rewrite 标准化：将手写 JSON 解析迁移到 LangChain structured output / output parser，继续保留关键 token 保护、unsafe rewrite 校验和 fallback 策略
- [ ] Streaming 重构：将当前 `ChatStreamEvent` / SSE 手写事件与 LangChain callback、LangGraph stream events 对齐，统一输出 token、tool、interrupt、resume、done 和 error 事件
- [ ] Workflow Evaluation：在现有 RAG 评测外，增加任务完成率、工具成功率、步骤失败率、恢复成功率和人工/LLM judge 评分
- [ ] Eval 标准化：保留 HTTP replay + qrels 指标，同时接入 LLM-as-a-judge / LangSmith 风格数据集与实验对比，用于生成质量和多步任务质量评测
- [ ] Cost & Latency Metrics：沉淀 token、模型成本、P50/P95 延迟、检索耗时、rerank 耗时和端到端耗时
- [ ] 场景扩展模板：提供新增 scene / workflow / tool / eval sample 的标准目录、接口约束和测试样例

### P1：继续强化 RAG 与知识库工程化

- [ ] 知识库批量重建索引、失败重试、索引状态诊断和上传文件清理
- [ ] 增量更新与缓存：支持文档增量索引、检索结果缓存、Embedding 缓存和缓存失效策略
- [ ] Retrieval Benchmark 扩充：扩大 qrels 样本，稳定 Precision / Recall / MRR / NDCG / no-hit false positive 指标
- [ ] 关键词召回扩展：为当前小规模 BM25 实现预留持久化倒排索引或外部搜索引擎边界
- [ ] 检索诊断 UI：展示 query rewrite、候选召回、过滤、rerank、citation 对齐和失败原因

### P2：面向开源与生产部署补齐工程闭环

- [ ] 权限与安全：用户、角色、知识库权限、工具调用权限、API Key、敏感信息过滤和审计日志
- [ ] 可观测性：结构化日志、OpenTelemetry Trace、模型调用 trace、工具调用 trace 和 eval run trace
- [ ] 部署能力：Docker Compose、生产配置样例、健康检查、备份恢复和日志采集
- [ ] 模型治理：模型路由策略、降级策略、超时重试、成本预算、调用审计和 provider fallback
- [ ] 开源文档：补齐架构决策记录、插件开发指南、Workflow DSL 示例、贡献指南和 Roadmap
- [ ] 企业集成：预留 SSO、对象存储、外部知识源、工单系统、CRM、内部搜索或支付/订单系统扩展点

## UI Preview

### 对话工作台

![对话工作台](./docs/images/api-tester-ui.png)

### 知识库管理

![知识库管理](./docs/images/knowledge-manager-ui.png)

## Architecture

```text
backend/
├─ application/        # FastAPI runtime、API 路由、服务装配
├─ platform/           # 配置、模型、记忆、知识处理、RAG、工具协议
└─ scenes/             # generic_assistant、ecommerce 等场景定义

frontend/              # 调试工作台、知识管理、评测看板
docs/                  # 架构图、接口文档、数据模型、设计说明
openspec/              # 变更提案、规格与归档记录
```

核心链路：

```text
用户问题
  -> 会话与场景解析
  -> Query Rewrite
  -> Agentic Retrieval Tool Decision
  -> Hybrid Search
  -> 低相关过滤 / no-hit fallback
  -> LLM Answer
  -> citations + retrieval_trace + evaluation artifact
```

## Quick Start

> 示例使用 PowerShell，命令默认在仓库根目录执行。

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

## Key APIs

- `POST /chat`：统一对话入口，支持 `stream=true` SSE。
- `GET /sessions` / `POST /sessions`：会话创建、查询与场景绑定。
- `POST /files/upload`：上传知识文件。
- `POST /knowledge/documents/preprocess-preview`：知识文件预处理预览。
- `POST /knowledge/documents`：确认入库并发布索引。
- `GET /evals/latest` / `POST /evals/runs`：读取或触发评测回放。

## Observability

普通请求直接查看 `/chat` JSON 响应：

```json
{
  "answer": "...",
  "knowledge_used": true,
  "citations": [
    {
      "index": 1,
      "source_name": "knowledge.md",
      "citation_id": "chunk-1",
      "snippet": "..."
    }
  ],
  "retrieval_trace": {
    "original_query": "...",
    "rewritten_query": "...",
    "final_decision": "answer_with_evidence",
    "tool_call_count": 1,
    "raw_candidates_count": 8,
    "filtered_candidates_count": 3,
    "top_k_chunks": [
      {
        "citation_id": "chunk-1",
        "score": 0.82
      }
    ]
  }
}
```

流式请求查看 SSE 事件：

- `tool.retrieval_trace`
- `done.retrieval_trace`

Evaluation Harness 回放后查看：

- `backend\data\evals\latest.json`
- `backend\data\evals\latest.md`

## Tests

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

## Documentation

- [系统架构图](./docs/architecture.svg)
- [知识管理流程图](./docs/knowledge-document-flow.svg)
- [Agentic RAG 流程图](./docs/agentic-rag-retrieval-flow.svg)
- [接口文档](./docs/api-list.md)
- [数据模型](./docs/data-model.md)
- [Agentic RAG 设计说明](./docs/agentic_rag.md)
- [Evaluation Harness 说明](./backend/evals/evaluation-harness.md)

## Positioning

这个项目的重点不是“能回答一句话”，而是展示一套可解释、可回归、可继续扩展的 RAG Runtime：

- 回答有引用
- 检索有 Trace
- 调参有评测
- 场景能扩展
- 知识能管理

# AI RAG Project Agent Guide

本文件服务 AI Agent：用于快速建立代码导航、理解架构边界、找到改动入口并避免常见误改。更细的背景资料不要堆在这里，优先走 `docs/documents/README.md` 渐进阅读。

## 文档导航

- 文档总索引：[docs/documents/README.md](./docs/documents/README.md)
- 系统架构图：[docs/documents/architecture/system-overview.svg](./docs/documents/architecture/system-overview.svg)
- Main Chat Agent Runtime Flow：[docs/documents/runtime/main-chat-agent-runtime-flow.svg](./docs/documents/runtime/main-chat-agent-runtime-flow.svg)
- ChatGraph SubGraphs：[docs/documents/runtime/chatgraph-subgraphs.svg](./docs/documents/runtime/chatgraph-subgraphs.svg)
- 知识管理流程图：[docs/documents/knowledge/knowledge-document-flow.svg](./docs/documents/knowledge/knowledge-document-flow.svg)
- Agentic RAG 流程图：[docs/documents/rag/agentic-rag-retrieval-flow.svg](./docs/documents/rag/agentic-rag-retrieval-flow.svg)
- API 文档：[docs/documents/reference/api-list.md](./docs/documents/reference/api-list.md)
- 数据模型：[docs/documents/reference/data-model.md](./docs/documents/reference/data-model.md)
- 常见坑与排障：[docs/documents/operations/common-pitfalls.md](./docs/documents/operations/common-pitfalls.md)

## 30 秒导航

如果你要改下面这些能力，优先看这些文件。

### 聊天主链路

- `backend/application/runtime/service.py`
- `backend/application/runtime/assembly/service_parts/agent_runtime.py`
- `backend/application/runtime/assembly/service_parts/turn_preparation.py`
- `backend/application/runtime/assembly/runtime_factory.py`
- `backend/application/runtime/api/chat/routes.py`
- `backend/application/runtime/api/chat/schemas.py`

### Application Runtime Assembly

- `backend/application/runtime/assembly/service_factory.py`（创建统一场景聊天服务）
- `backend/application/runtime/assembly/runtime_factory.py`（`ChatGraphRuntime`，负责 checkpoint / HITL / run lifecycle）
- `backend/application/runtime/assembly/service_parts/`（ChatService 的准备、Agent 执行、回答、引用、HITL、响应组装拆分）
- `backend/application/runtime/assembly/runtime_parts/`（Graph runtime 的状态、answer graph、HITL、state store 拆分）

### Agent Runtime / ChatGraph

- `backend/platform/agent_runtime/chat_graph/graph.py`（顶层 ChatGraph 拓扑）
- `backend/platform/agent_runtime/chat_graph/nodes/`（prepare/select/route/branch/synthesis/persist 节点）
- `backend/platform/agent_runtime/react/graph/graph.py`（ReAct 子图）
- `backend/platform/agent_runtime/react/graph/nodes/`（ReAct action、tool、wait、final 节点）
- `backend/platform/agent_runtime/plan/graph/graph.py`（Plan 子图）
- `backend/platform/agent_runtime/plan/graph/nodes/`（plan create、step execute、retry、wait、synthesis 节点）
- `backend/platform/agent_runtime/tool_executor.py`
- `backend/platform/agent_runtime/rag_tools.py`

### Workflow / LangGraph Runtime

- `backend/platform/workflow/state_machine.py`
- `backend/platform/workflow/langgraph/state.py`
- `backend/platform/workflow/langgraph/lifecycle.py`
- `backend/platform/workflow/langgraph/checkpointer.py`
- `backend/application/runtime/assembly/runtime_factory.py`

### 场景定义与 prompt

- `backend/scenes/generic_assistant/definition.py`
- `backend/scenes/ecommerce/definition.py`（可选演示场景，不是平台主线）
- `backend/scenes/registry.py`（默认 scene 与 business extension 组合装配入口）

### Agentic Retrieval

- `backend/platform/rag/orchestration/agentic.py`
- `backend/platform/rag/orchestration/decisions.py`
- `backend/platform/rag/contracts.py`
- `backend/platform/rag/pre_retrieval/query_rewrite.py`
- `backend/platform/rag/retrieval/documents/service.py`

### 工具协议与工具实现

- `backend/platform/tools/base.py`（工具基础协议、上下文、结果结构）
- `backend/platform/tools/adapters.py`（StructuredTool / RetrievalTool adapter）
- `backend/platform/tools/registry.py`（工具注册元数据、Agent 白名单、MCP 暴露标记）
- `backend/scenes/generic_assistant/tools/`（通用知识助手具体工具）
- `backend/scenes/ecommerce/tools/`（电商具体工具）

### 文档 Hybrid Search

- `backend/platform/rag/retrieval/documents/service.py`
- `backend/platform/rag/retrieval/documents/semantic.py`
- `backend/platform/rag/retrieval/documents/keyword.py`
- `backend/platform/rag/retrieval/documents/fusion.py`

### 知识文档写流程

- `backend/platform/knowledge/documents/application_service.py`
- `backend/platform/knowledge/documents/publisher.py`
- `backend/platform/knowledge/processing/`

### 会话与记忆

- `backend/platform/memory/base/session_store.py`
- `backend/platform/memory/chat/prompt_context.py`

### 启动与依赖注入

- `backend/run.py`
- `backend/application/runtime/api/app.py`
- `backend/application/runtime/bootstrap.py`
- `backend/application/runtime/assembly/service_factory.py`
- `backend/application/runtime/assembly/runtime_factory.py`

## 项目定位

这是一个面向多场景智能助手的 RAG / Agent Runtime 示例项目。当前主线是：

- 提供可运行的 FastAPI 后端
- 提供按会话绑定场景的统一 `/chat` 入口
- 支持本地知识文件上传、文档索引、向量检索和回答生成
- 用 `generic_assistant` 验证平台主链路
- 用 `ecommerce` 作为可选演示场景，展示“平台能力 + 场景扩展”的组织方式

## 架构边界

### 分层边界

- `platform` 放通用底层能力，不感知具体业务场景。
- `application` 放运行时装配与 API 暴露。
- `scenes` 放具体场景定义、prompt、scene policy 和业务工具。
- 不要把运行时装配逻辑塞进 `platform` 或 `__init__.py`。
- `__init__.py` 保持轻量，避免引入循环依赖。

### RAG 与 Knowledge 边界

- 文档召回、关键词召回、Hybrid Search 统一进 `platform.rag.retrieval.documents`。
- rerank 边界统一进 `platform.rag.post_retrieval`。
- `platform.knowledge` 只承接知识管理、底层存储和仓储访问，不新增面向 scene/chat 的文档召回业务 API。
- 文档检索统一通过 `DocumentRetrievalService` 进入，不要在 scene 里重造文档召回入口。
- Agentic RAG 是 `platform.rag.orchestration` 下的编排层，不是与 Modular RAG 并列竞争的架构。

### Workflow Runtime 边界

- Workflow run state 使用 `backend/platform/workflow/state_machine.py` 的统一枚举和转移校验。
- `succeeded / failed / cancelled` 是终态，不能继续 resume、retry 或重新写成 running。
- `waiting_user` 表示等待人工输入，不是失败。
- HITL `reject` 或 cancel 进入 `cancelled`，不是模型或工具错误。
- LangGraph checkpoint 保存可恢复运行时状态；`sessions.status` 只表示聊天会话是否 active/expired，不能替代 workflow run state。

### 会话、场景与知识源

- 新会话默认场景由 `AI_RAG_APP__ACTIVE_SCENE` 控制。
- 新会话默认挂载知识源是 `["documents"]`。
- 可在 `POST /sessions` 里通过 `mounted_knowledge_sources` 扩展到 `["documents", "ecommerce"]`。
- `scene` 负责 prompt 与运行时风格，知识源是否可用由会话挂载配置决定。
- candidate retrieval tools 由 `SceneDefinition` 根据 `mounted_knowledge_sources` 解析；runtime 不硬编码 knowledge source 到 tool name 的映射。

### Tools 边界

- `backend/platform/tools` 只放中立协议、adapter 和 registry，不引入具体业务工具。
- 具体工具按 scene 维护，例如 `backend/scenes/generic_assistant/tools/` 和 `backend/scenes/ecommerce/tools/`。
- 每个逻辑工具对应一个独立类。
- scene definition 负责工具装配和范围控制；runtime 不从全局工具池自由选择业务工具。

## 常见任务

### 改 `/chat` 主链路

- 先看 `backend/application/runtime/service.py`
- Agent 执行看 `backend/application/runtime/assembly/service_parts/agent_runtime.py`
- turn 准备看 `backend/application/runtime/assembly/service_parts/turn_preparation.py`
- ChatGraph 持久化与 HITL runtime 看 `backend/application/runtime/assembly/runtime_factory.py`
- 再看 `backend/application/runtime/api/chat/routes.py`
- 响应结构改动再看 `backend/application/runtime/api/chat/schemas.py`

### 改 ReAct / Plan / ChatGraph 编排

- 顶层图先看 `backend/platform/agent_runtime/chat_graph/graph.py`
- ReAct 子图看 `backend/platform/agent_runtime/react/graph/`
- Plan 子图看 `backend/platform/agent_runtime/plan/graph/`
- 工具执行入口看 `backend/platform/agent_runtime/tool_executor.py`
- RAG 工具 adapter 看 `backend/platform/agent_runtime/rag_tools.py`
- 测试优先看 `backend/tests/test_agent_runtime_react.py`、`backend/tests/test_agent_runtime_plan.py`、`backend/tests/test_agent_runtime_tools.py`、`backend/tests/test_agent_runtime_mode_selector.py`

### 改 Workflow State Machine

- 先看 `backend/platform/workflow/state_machine.py`
- 生命周期接入看 `backend/platform/workflow/langgraph/lifecycle.py`
- runtime checkpoint / HITL 接入看 `backend/application/runtime/assembly/runtime_factory.py`
- API/SSE 字段映射看 `backend/application/runtime/service.py`、`backend/application/runtime/stream_events.py` 和 `backend/application/runtime/api/chat/schemas.py`
- 测试优先看 `backend/tests/test_langgraph_runtime.py`、`backend/tests/test_generic_assistant_hitl.py`、`backend/tests/test_chat_api.py`

### 改文档检索或 Hybrid Search

- 先看 `backend/platform/rag/retrieval/documents/service.py`
- 语义召回看 `semantic.py`
- 关键词召回看 `keyword.py`
- 融合排序看 `fusion.py`

### 改知识文档入库、重处理、发布

- 先看 `backend/platform/knowledge/documents/application_service.py`
- 发布切换看 `backend/platform/knowledge/documents/publisher.py`
- 预处理逻辑看 `backend/platform/knowledge/processing/`

### 改 scene prompt、tool 或场景编排

- 主看 `backend/scenes/generic_assistant/definition.py`
- 演示场景看 `backend/scenes/ecommerce/definition.py`
- 默认场景组合装配看 `backend/scenes/registry.py`

## 数据位置

- 会话记忆默认落在 `backend/data/sessions.db`
- LangGraph checkpoint 默认落在 `backend/data/langgraph.db`
- 文件上传目录默认在 `backend/data/files`
- 向量存储默认是 Chroma，可切换到 Elasticsearch
- Elasticsearch 本地运行说明见 `devops/elasticsearch/README.md`

## 文档同步

如果改动了架构、接口、数据模型、运行方式、状态语义或环境变量，优先同步检查：

- `README.md`
- `AGENTS.md`
- `backend/.env.example`
- `docs/documents/reference/api-list.md`
- `docs/documents/reference/data-model.md`
- `docs/documents/architecture/*.mmd` 与对应 `.svg`
- 对应模块文档，例如 `docs/documents/rag/`、`docs/documents/knowledge/`、`docs/documents/operations/`

## 最小运行与验证

以下命令默认在仓库根目录执行。

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
backend\.venv\Scripts\python.exe backend\run.py
```

全量测试：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

单文件示例：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

## 修改纪律

- 使用 `rg` 或 `rg --files` 搜索文本和文件。
- 使用 `apply_patch` 做手工文件修改。
- 避免大面积无关格式化 diff。
- 不要回退用户已有改动。
- 修改 `__init__.py` 时保持最小化，避免循环导入。
- PowerShell 5.1 读取中文或 Markdown 时显式使用 `-Encoding UTF8`。
- 更多高频错误和排障规则见：[常见坑与排障](./docs/documents/operations/common-pitfalls.md)。

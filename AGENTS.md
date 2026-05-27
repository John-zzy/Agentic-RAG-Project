# AI RAG Project Agent Guide

本文件主要服务 AI Agent，用于快速建立代码导航、理解架构边界、找到改动入口并避免常见误改。

如需补充设计背景，再看 `docs/`：

- 架构图：[docs/architecture.svg](./docs/architecture.svg)
- 知识管理流程图：[docs/knowledge-document-flow.svg](./docs/knowledge-document-flow.svg)
- Agentic RAG 流程图：[docs/agentic-rag-retrieval-flow.svg](./docs/agentic-rag-retrieval-flow.svg)
- 接口文档：[docs/api-list.md](./docs/api-list.md)
- 数据模型：[docs/data-model.md](./docs/data-model.md)
- Agentic RAG 说明：[docs/agentic_rag.md](./docs/agentic_rag.md)

## 30 秒导航

如果你要改下面这些能力，优先看这些文件：

- 聊天主链路：
  - `backend/application/runtime/service.py`
  - `backend/application/runtime/api/chat/routes.py`
  - `backend/application/runtime/api/chat/schemas.py`
- 场景定义与 prompt：
  - `backend/scenes/generic_assistant/definition.py`
  - `backend/scenes/ecommerce/definition.py`（可选演示场景，不是平台主线）
  - `backend/scenes/registry.py`（默认 scene 与 business extension 组合装配入口）
- Agentic Retrieval：
  - `backend/platform/rag/orchestration/agentic.py`
  - `backend/platform/rag/orchestration/decisions.py`
  - `backend/platform/rag/contracts.py`
  - `backend/platform/rag/pre_retrieval/query_rewrite.py`
  - `backend/platform/rag/retrieval/documents/service.py`
- 文档 Hybrid Search：
  - `backend/platform/rag/retrieval/documents/service.py`
  - `backend/platform/rag/retrieval/documents/semantic.py`
  - `backend/platform/rag/retrieval/documents/keyword.py`
  - `backend/platform/rag/retrieval/documents/fusion.py`
- 知识文档写流程：
  - `backend/platform/knowledge/documents/application_service.py`
  - `backend/platform/knowledge/documents/publisher.py`
  - `backend/platform/knowledge/processing/`
- 知识文档读流程：
  - `backend/platform/knowledge/documents/query_service.py`
- 会话与记忆：
  - `backend/platform/memory/base/session_store.py`
  - `backend/platform/memory/chat/prompt_context.py`
- 启动与依赖注入：
  - `backend/run.py`
  - `backend/application/runtime/api/app.py`
  - `backend/application/runtime/bootstrap.py`

## 项目定位

这是一个面向多场景智能助手的 RAG / Agent Runtime 示例项目。当前主线是：

- 提供可运行的 FastAPI 后端
- 提供按会话绑定场景的统一 `/chat` 入口
- 支持本地知识文件上传、文档索引、向量检索和回答生成
- 用 `generic_assistant` 验证平台主链路
- 用 `ecommerce` 作为可选演示场景，展示“平台能力 + 场景扩展”的组织方式

它不是单一 Prompt Demo，而是一个可继续演进的场景化 RAG / Agent Runtime 起点。

## 架构地图

后端按 `platform / application / scenes` 三层组织：

- `platform`
  - 通用底层能力，不感知具体业务场景
  - 包括配置、模型路由、会话记忆、知识处理、RAG 核心、工具协议
  - `platform/search_foundation` 是其子模块，不是独立顶层
- `application`
  - 运行时装配与 API 暴露
  - 包括启动、依赖注入、Chat service 组装、FastAPI 路由注册
- `scenes`
  - 具体场景定义
  - 当前以 `generic_assistant` 为主，`ecommerce` 为可选演示场景

## 常见任务

### 改 `/chat` 主链路

- 先看 `backend/application/runtime/service.py`
- 再看 `backend/application/runtime/api/chat/routes.py`
- 响应结构改动再看 `backend/application/runtime/api/chat/schemas.py`

### 改文档检索或 Hybrid Search

- 先看 `backend/platform/rag/retrieval/documents/service.py`
- 语义召回看 `backend/platform/rag/retrieval/documents/semantic.py`
- 关键词召回看 `backend/platform/rag/retrieval/documents/keyword.py`
- 融合排序看 `backend/platform/rag/retrieval/documents/fusion.py`

### 改 Agentic Retrieval 决策

- 先看 `backend/platform/rag/orchestration/agentic.py`
- 决策类型在 `backend/platform/rag/orchestration/decisions.py`
- 共享检索协议在 `backend/platform/rag/contracts.py`
- 查询改写协议在 `backend/platform/rag/pre_retrieval/query_rewrite.py`
- scene 级策略在 `backend/scenes/*/definition.py`

### 改知识文档入库、重处理、发布

- 先看 `backend/platform/knowledge/documents/application_service.py`
- 发布切换看 `backend/platform/knowledge/documents/publisher.py`
- 预处理逻辑看 `backend/platform/knowledge/processing/`

### 改会话、历史、挂载知识源

- 先看 `backend/platform/memory/base/session_store.py`
- prompt 上下文组装看 `backend/platform/memory/chat/prompt_context.py`
- 会话 API 看 `backend/application/runtime/api/chat/routes.py`

### 改 scene prompt、tool 或场景编排

- 主看 `backend/scenes/generic_assistant/definition.py`
- 演示场景看 `backend/scenes/ecommerce/definition.py`
- 默认场景组合装配看 `backend/scenes/registry.py`
- 抽象定义在 `backend/scenes/base.py`

## 架构边界

### 分层边界

- 优先遵守 `platform / application / scenes` 的分层边界。
- 不要把运行时装配逻辑塞进 `platform` 或 `__init__.py`。
- `__init__.py` 保持轻量，避免引入循环依赖。
- `platform.search_foundation` 只放共享类型、默认算法和基础读侧契约，不要引入 `platform.knowledge` 或 `platform.rag` 依赖。

### RAG 与 Knowledge 边界

- 文档召回、关键词召回、Hybrid Search 统一进 `platform.rag.retrieval.documents`，rerank 边界统一进 `platform.rag.post_retrieval`。
- Agentic RAG 是 `platform.rag.orchestration` 下的编排层，不是与 Modular RAG 并列竞争的架构。
- `platform.knowledge` 只承接知识管理、底层存储和仓储访问，不要新增面向 scene/chat 的文档召回业务 API。
- 文档检索统一通过 `DocumentRetrievalService` 进入，不要在 scene 里重造文档召回入口。
- 如果后续为 `products/reviews` 增加关键词召回或 Hybrid Search，继续放在 `platform.rag`。
- `orders/inventory/detail` 这类结构化能力优先保持为 structured tools，经由 Agentic Retrieval 编排。

### 会话、场景与知识源

- 新会话默认场景由 `AI_RAG_APP__ACTIVE_SCENE` 控制。
- 新会话默认挂载知识源是 `["documents"]`，可在 `POST /sessions` 里通过 `mounted_knowledge_sources` 显式扩展到 `["documents", "ecommerce"]`。
- 日常切换场景优先走会话级 API 或前端选择，不要把改环境变量当主流程。
- `scene` 负责 prompt 与运行时风格，知识源是否可用由会话挂载配置决定，不要再把二者视为同一个开关。
- candidate retrieval tools 由 `SceneDefinition` 根据 `mounted_knowledge_sources` 解析；不要再在 runtime 中硬编码 knowledge source 到 tool name 的映射。
- `generic_assistant` 持有默认 docs-first 主链；`ecommerce` 等业务场景通过 business extension 接入，不要再让 generic 反向依赖业务默认 judge / rewriter / tool builder。

### 数据位置

- 会话记忆默认落在 `backend/data/sessions.db`
- 文件上传目录默认在 `backend/data/files`
- 向量存储默认是 Chroma，可切换到 Elasticsearch

### 文档同步

- 如果改动了架构、接口、数据模型、运行方式或环境变量，优先同步检查：
  - `README.md`
  - `AGENTS.md`
  - `backend/.env.example`
  - `docs/api-list.md`
  - `docs/data-model.md`
  - `docs/*.mmd` 与对应 `.svg`

## 最小运行与验证

以下命令默认在仓库根目录执行。

### 安装依赖

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
```

### 最小环境变量

```env
AI_RAG_MODELS__SIMPLE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__MODERATE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__COMPLEX__API_KEY=your-dashscope-api-key
AI_RAG_APP__ACTIVE_SCENE=generic_assistant
AI_RAG_VECTOR_STORE__PROVIDER=chroma
```

### 启动后端

```powershell
backend\.venv\Scripts\python.exe backend\run.py
```

### 运行测试

- 全量测试：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

- 单文件示例：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

## 修改注意事项

- 使用 `apply_patch` 做手工文件修改
- 避免大面积无关格式化 diff
- 修改 `__init__.py` 时保持最小化，避免引入运行时依赖导致循环导入
- 如果改动了架构、启动方式、环境变量或测试命令，要同步检查 `README.md`、`AGENTS.md`、`backend/.env.example`
- 知识文档写流程统一接到 `KnowledgeDocumentApplicationService` 和 `KnowledgeDocumentPublisher`，不要恢复聚合式文档服务

## 高频错误

- 使用了错误的 Python 解释器，而不是 `backend\.venv\Scripts\python.exe`
- 改完实际代码后，没有同步更新文档和环境样例
- 在 `__init__.py` 中引入运行时装配，导致循环导入
- 用不精确的覆盖式写文件方式修改内容，导致文件损坏或内容串乱
- 架构相关改动后，没有补跑受影响测试或全量测试

## Encoding And Patch Discipline

- This repo's code and docs should be treated as UTF-8 unless the file itself clearly proves otherwise.
- In this Windows PowerShell 5.1 environment, `Get-Content` without `-Encoding` may decode files with the system ANSI code page (`gb2312` here), which will garble UTF-8 Chinese text. Do not use default decoding when reading source files that may contain non-ASCII text.
- When reading text files for inspection or patch preparation, explicitly use UTF-8, for example: `Get-Content -Raw -Encoding UTF8 <file>`.
- If a command writes text files directly, explicitly use UTF-8 as well. Never rely on PowerShell 5.1 default file encoding for source code, HTML, Markdown, JSON, YAML, or config files.
- If terminal output shows mojibake, first determine whether the file bytes are valid UTF-8 before assuming the file content is corrupted. Distinguish "wrong decode while reading" from "actual file damage".
- Never build `apply_patch` context from garbled terminal output. Re-read the file with explicit UTF-8 and anchor patches on stable exact text.
- A failed `apply_patch` does not justify a whole-file rewrite by itself. First re-read the live file, shrink the patch hunk, and retry with stable anchors.
- Before patching a file that was recently edited, re-read its current contents from disk. Do not trust earlier copied snippets after structural changes.
- Do not escalate localized changes into full-file rewrites unless the user explicitly approves it, or the file is already inconsistent enough that targeted patching is no longer safe.

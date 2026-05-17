# AI RAG Project Agent Guide

本文件用于帮助 AI / 开发者快速进入这个仓库。内容保持精简，重点说明项目定位、架构、关键模块、约定和运行方式。

详细设计与清单请直接看 `docs/`：

- 架构图：[docs/architecture.svg](./docs/architecture.svg)
- 模块依赖图：[docs/module-deps.svg](./docs/module-deps.svg)
- 外部依赖图：[docs/external-deps.svg](./docs/external-deps.svg)
- 接口清单：[docs/api-list.md](./docs/api-list.md)
- 数据模型：[docs/data-model.md](./docs/data-model.md)
- 数据模型 ER 图：[docs/data-model-er.svg](./docs/data-model-er.svg)
- Agentic RAG 说明：[docs/agentic_rag.md](./docs/agentic_rag.md)

## 项目定位

这是一个面向多场景智能助手的 RAG 示例项目，当前核心目标是：

- 提供一个可运行的 FastAPI 后端
- 支持按会话绑定场景的统一 `/chat` 入口
- 支持本地知识文件上传、文档索引、向量检索和回答生成
- 用 `generic_assistant` 与 `ecommerce` 两个场景演示“平台能力 + 场景扩展”的结构

从设计上，它不是单一 Prompt Demo，而是一个可继续演进的场景化 RAG / Agent Runtime 起点。

## 核心架构

后端按三层组织：

- `platform`
  - 放通用底层能力，不感知具体业务场景。
  - 包括配置、模型路由、会话记忆、知识处理、RAG 核心、工具协议。
- `application`
  - 放运行时装配与 API 暴露。
  - 包括启动、依赖注入、Chat service 组装、FastAPI 路由注册。
- `scenes`
  - 放具体场景定义。
  - 当前内置 `generic_assistant` 和 `ecommerce`。

如果要看图而不是读代码，优先看：

- [docs/architecture.svg](./docs/architecture.svg)
- [docs/module-deps.svg](./docs/module-deps.svg)

## 关键模块

### `backend/run.py`

后端启动入口。负责启动 FastAPI 应用。

### `backend/application/runtime`

运行时装配层。

- `bootstrap.py`
  - 初始化当前激活场景和运行时摘要。
- `service.py`
  - 统一聊天主链路，核心类是 `ActiveSceneChatService` 和 `ChatService`。
- `api/app.py`
  - 创建 FastAPI 应用并注册路由。
- `api/chat/`
  - 聊天、场景、会话接口。
- `api/file/`
  - 文件上传、列表、删除、下载接口。
- `api/knowledge/`
  - 知识文档预处理预览、注册、列表、详情、删除、重处理、重分块接口。

### `backend/platform`

平台公共能力层。

- `config/`
  - 环境变量、模型配置、向量库配置。
- `models/`
  - 模型路由与 LLM 客户端封装。
- `memory/`
  - SQLite 会话、轮次、消息历史持久化。
- `knowledge/`
  - 通用知识文件读取、预处理、切块、索引管理、向量存储抽象。
  - `base/store.py` 里已经拆出 `KnowledgeRetriever` 和 `KnowledgeDocumentRepository` 两套接口。
  - `processing/` 负责标准化、规则清洗、预览、统计和 provenance 元数据生成。
  - `documents/application_service.py` 负责预处理预览、注册、删除、重处理、重切块这类写流程。
  - `documents/query_service.py` 负责文档列表、详情和文件索引状态聚合查询。
  - `documents/publisher.py` 负责新版本发布、旧版本失活、失败恢复和清理。
  - `documents/mappers.py` 负责 DTO 映射，不要再把映射逻辑塞回应用服务。
- `rag/`
  - 检索编排、Sufficiency 判断、Query Rewrite 等 Agentic RAG 核心协议。
- `tools/`
  - 通用工具协议与结构化工具封装。

### `backend/scenes`

场景层。

- `generic_assistant/`
  - 通用文档问答场景。
- `ecommerce/`
  - 电商演示场景，包含商品、评论、订单、库存、知识文档等多源检索与工具能力。
- `base.py`
  - 场景抽象定义。

## 关键约定

### 代码组织

- 优先遵守 `platform / application / scenes` 的分层边界。
- 不要把运行时装配逻辑塞进 `platform` 或 `__init__.py`。
- `__init__.py` 保持轻量，避免引入循环依赖。

### 会话与场景

- 新会话默认场景由 `AI_RAG_APP__ACTIVE_SCENE` 控制。
- 新会话默认挂载知识源是 `["documents"]`，可在 `POST /sessions` 里通过 `mounted_knowledge_sources` 显式扩展到 `["documents", "ecommerce"]`。
- 日常切换场景优先走会话级 API 或前端选择，不要把改环境变量当主流程。
- `scene` 负责 prompt 与运行时风格，知识源是否可用由会话挂载配置决定，不要再把二者视为同一个开关。

### 数据与存储

- 会话记忆默认落在 `backend/data/sessions.db`
- 文件上传目录默认在 `backend/data/files`
- 向量存储默认是 Chroma，可切换到 Elasticsearch

### 文档与图资产

- 如果改动了架构、接口、数据模型、运行方式或环境变量，优先同步检查：
  - `README.md`
  - `AGENTS.md`
  - `backend/.env.example`
  - `docs/api-list.md`
  - `docs/data-model.md`
  - `docs/*.mmd` 与对应 `.svg`

### 编码与修改方式

- 仓库文本文件默认按 UTF-8 处理。
- 在 PowerShell 5.1 下读源码或 Markdown 时显式使用 `-Encoding UTF8`。
- 做局部修改优先保持最小 diff，不要无关重排。

## 怎么跑

以下命令默认在仓库根目录执行。

### 1. 创建虚拟环境并安装依赖

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
```

### 2. 配置最小环境变量

至少配置：

```env
AI_RAG_MODELS__SIMPLE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__MODERATE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__COMPLEX__API_KEY=your-dashscope-api-key
AI_RAG_APP__ACTIVE_SCENE=generic_assistant
AI_RAG_VECTOR_STORE__PROVIDER=chroma
```

### 3. 启动后端

```powershell
backend\.venv\Scripts\python.exe backend\run.py
```

默认地址：

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- API 调试页: `http://127.0.0.1:8000/frontend/api-tester.html`
- 知识库管理页: `http://127.0.0.1:8000/frontend/knowledge-manager.html`

知识库管理页当前流程：

- 上传 `json`、`csv`、`txt`、`md` 文件后会自动打开“数据预处理”弹窗
- 通过 `preprocess-preview` 预览规则、样本和统计，再确认正式入库
- 未入库但可处理文件状态为 `awaiting_processing`
- `pdf`、`docx`、`xlsx` 当前允许上传，但不会进入预处理与索引链路

### 4. 切换到 Elasticsearch

本地启动：

```powershell
docker compose -f docs/elasticsearch/docker-compose.yml up -d
```

对应说明见：

- [docs/elasticsearch/README.md](./docs/elasticsearch/README.md)

### 5. 运行测试

全量后端测试：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

单文件示例：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

## 禁区

- 待补充

## 历史包袱

### 修改时的注意事项

- 使用 `apply_patch` 做手工文件修改
- 避免大面积无关格式化 diff
- 修改 `__init__.py` 时保持最小化，避免引入运行时依赖导致循环导入
- 如果改动了架构、启动方式、环境变量或测试命令，要同步检查 `README.md`、`AGENTS.md`、`backend/.env.example`
- 知识文档写流程统一接到 `KnowledgeDocumentApplicationService` 和 `KnowledgeDocumentPublisher`，不要恢复聚合式文档服务

### 高频错误

- 使用了错误的 Python 解释器，而不是 `backend\.venv\Scripts\python.exe`
- 改完实际代码后，没有同步更新文档和环境样例
- 在 `__init__.py` 中引入运行时装配，导致循环导入
- 用不精确的覆盖式写文件方式修改内容，导致文件损坏或内容串乱
- 架构相关改动后，没有补跑受影响测试或全量测试

### Encoding And Patch Discipline

- This repo's code and docs should be treated as UTF-8 unless the file itself clearly proves otherwise.
- In this Windows PowerShell 5.1 environment, `Get-Content` without `-Encoding` may decode files with the system ANSI code page (`gb2312` here), which will garble UTF-8 Chinese text. Do not use default decoding when reading source files that may contain non-ASCII text.
- When reading text files for inspection or patch preparation, explicitly use UTF-8, for example: `Get-Content -Raw -Encoding UTF8 <file>`.
- If a command writes text files directly, explicitly use UTF-8 as well. Never rely on PowerShell 5.1 default file encoding for source code, HTML, Markdown, JSON, YAML, or config files.
- If terminal output shows mojibake, first determine whether the file bytes are valid UTF-8 before assuming the file content is corrupted. Distinguish "wrong decode while reading" from "actual file damage".
- Never build `apply_patch` context from garbled terminal output. Re-read the file with explicit UTF-8 and anchor patches on stable exact text.
- A failed `apply_patch` does not justify a whole-file rewrite by itself. First re-read the live file, shrink the patch hunk, and retry with stable anchors.
- Before patching a file that was recently edited, re-read its current contents from disk. Do not trust earlier copied snippets after structural changes.
- Do not escalate localized changes into full-file rewrites unless the user explicitly approves it, or the file is already inconsistent enough that targeted patching is no longer safe.

# AI RAG Project

一个面向多场景智能助手的 RAG 示例项目。仓库当前已经收敛为清晰的三层后端架构：

- `platform`：平台级通用能力
- `application`：运行时装配与 API 暴露
- `scenes`：具体业务场景实现

当前默认场景为 `generic_assistant`，同时保留 `ecommerce` 作为电商演示场景。项目适合作为以下工作的起点：

- 搭建一个可运行的 FastAPI + RAG 后端
- 扩展新的会话场景、检索策略和场景工具
- 接入本地知识文档并完成向量检索
- 演示多场景会话路由与会话级场景切换

## 项目概览

当前项目已经具备一个可运行的多场景 RAG 后端，当前核心能力如下：

- 统一 `/chat` 入口，按会话绑定 `scene` 处理请求
- `/chat` 支持 `stream=true` 的 SSE 最终回答流式输出
- 会话创建、查询、删除与 `mounted_knowledge_sources` 知识源挂载
- 本地知识文件上传、预处理预览、正式入库、重处理与重切块
- 文档切块、索引、语义检索与 `documents/chunks` 的 `Hybrid Search`
- Agentic Retrieval 编排，支持“文档优先 + 按需切换场景工具”
- 结构化 `citations`、回答正文引用编号与 session `retrieval_snippets` 落库
- `Chroma` / `Elasticsearch` 可切换向量存储，以及基于 `SQLite` 的会话记忆
- 内置 `generic_assistant` 与 `ecommerce` 两个场景，用于验证平台能力与场景扩展方式

当前定位更偏“场景化 RAG / Agent Runtime 底座”，还不是完整产品。

## 当前进展

当前主链路已经不是“单一文档检索 Demo”，而是一个完成过一轮边界收敛的 scene-based Agentic RAG Runtime：

- [x] 三层后端结构：`platform / application / scenes`
- [x] 统一 `/chat`、`/sessions`、`/files`、`/knowledge/documents` API
- [x] `generic_assistant` 独立 docs-first 检索链路
- [x] `GenericAssistantSufficiencyJudge` 与 `GenericAssistantQueryRewriter`
- [x] 会话级 `mounted_knowledge_sources` 挂载与 scene definition 候选工具解析
- [x] `ecommerce` 以 business extension 方式接入 generic 主链，而不是反向成为默认依赖
- [x] 文档 `Hybrid Search`：语义召回 + 关键词召回 + 融合排序
- [x] 知识文档预处理预览、注册、重处理、重分块、软删除与文件维度索引视图
- [x] 结构化 `citations`、回答正文引用编号与 session `retrieval_snippets` 持久化
- [x] `Chroma` / `Elasticsearch` 可切换向量存储，`SQLite` 会话持久化

## 后续重点

当前 README 已同步到最近一轮 generic/ecommerce 解耦后的状态。后续优先事项聚焦在“补能力”而不是“继续拆边界”：

- `Tool Registry`、函数调用协议、多步执行与任务状态管理
- `WebSocket` 双向流式协议
- RAG 评测脚手架、ReRank 与更系统的效果验证
- 鉴权、观测、部署与长期任务框架
- 正式产品界面替代当前调试页

## 设计文档

如果你希望先从设计层面理解这个项目，而不是直接读代码，可以先看下面的文档：

- [系统架构图](./docs/architecture.svg)
- [知识管理流程图](./docs/knowledge-document-flow.svg)
- [Agentic RAG 流程图](./docs/agentic-rag-retrieval-flow.svg)
- [接口文档](./docs/api-list.md)
- [数据模型](./docs/data-model.md)
- [Agentic RAG 设计说明](./docs/agentic_rag.md)：解释多轮召回、扩展 handoff、证据聚合和最终回答生成链路

## 系统架构

后端采用 `platform / application / scenes` 三层结构：

### 1. Platform

`backend/platform` 提供与具体场景无关的底层能力，主要包括：

- 配置加载与模型路由
- LLM 客户端封装
- 会话存储与聊天上下文
- 通用知识文档处理
- RAG 检索核心协议与实现
- `platform/retrieval` 作为 `knowledge` 与 `rag` 共享的中立检索底座

当前关键边界：

- 文档检索统一收口到 `platform.rag`
- `platform.knowledge` 负责知识资产管理、预处理、发布和存储访问
- `products/reviews` 这类文本型知识若继续补 Hybrid Search，应继续放在 `platform.rag`
- `orders/inventory/detail` 这类结构化能力优先保持为 structured tools，经由 Agentic Retrieval 编排

### 2. Application

`backend/application/runtime` 负责运行时装配，主要包括：

- 应用启动引导
- 默认 scene 选择
- Chat service 组装与依赖注入
- FastAPI 应用与 API 路由注册

### 3. Scenes

`backend/scenes` 放置具体场景定义，主要包括：

- 场景提示词与定义
- 场景级检索工具与决策逻辑
- 场景知识组织方式与场景特有服务

## 目录结构

```text
.
├─ backend/
│  ├─ application/
│  │  └─ runtime/                 # 运行时装配、服务编排、API 入口
│  ├─ platform/
│  │  ├─ config/                  # 配置与模型路由
│  │  ├─ knowledge/               # 通用知识文档处理与索引
│  │  ├─ memory/                  # 会话存储与聊天上下文
│  │  ├─ models/                  # 模型抽象与 LLM 客户端
│  │  ├─ rag/                     # RAG 核心协议与检索实现
│  │  ├─ retrieval/               # knowledge / rag 共享的中立检索底座
│  │  └─ tools/                   # 通用工具协议
│  ├─ scenes/
│  │  ├─ generic_assistant/       # 通用助手场景
│  │  └─ ecommerce/               # 电商演示场景
│  ├─ tests/                      # 后端测试
│  ├─ data/                       # 本地数据与持久化目录
│  ├─ .env.example
│  ├─ requirements.txt
│  └─ run.py                      # 后端启动入口
├─ frontend/                      # 调试用静态页面
├─ docs/                          # 补充文档
├─ openspec/                      # 变更提案与规格文档
├─ AGENTS.md                      # 面向 AI Agent 的快速指引
└─ README.md
```

## 运行环境准备

以下命令默认在仓库根目录执行，示例使用 PowerShell。

### 1. 创建虚拟环境并安装依赖

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
```

### 2. 配置环境变量

至少需要在 `backend\.env` 中配置模型 API Key：

```env
AI_RAG_MODELS__SIMPLE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__MODERATE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__COMPLEX__API_KEY=your-dashscope-api-key
AI_RAG_APP__ACTIVE_SCENE=generic_assistant
AI_RAG_VECTOR_STORE__PROVIDER=chroma
```

说明：

- `AI_RAG_APP__ACTIVE_SCENE` 表示“新会话默认场景”
- 日常切换场景时，优先通过会话级 API 或前端选择，而不是频繁手工改环境变量

### 3. 启动后端

请直接使用虚拟环境中的解释器：

```powershell
backend\.venv\Scripts\python.exe backend\run.py
```

默认访问地址：

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- API 调试页: `http://127.0.0.1:8000/frontend/api-tester.html`
- 知识库管理页: `http://127.0.0.1:8000/frontend/knowledge-manager.html`

## 界面预览

下面两张图展示了前端调试页和知识库管理页的默认界面，便于快速了解整体交互入口。

### 智能客服工作台

![智能客服工作台](./docs/images/api-tester-ui.png)

### 知识库管理页

![知识库管理页](./docs/images/knowledge-manager-ui.png)

## 接口与使用说明

当前主要接口分为四类：

- 聊天与会话：`/chat`、`/sessions`、`/scenes`
- 文件管理：`/files`
- 知识文档：`/knowledge/documents`
- 健康检查：`/health`

典型流程：

1. 启动服务
2. 通过 `POST /sessions` 创建会话并指定场景，可选传入 `mounted_knowledge_sources`
3. 通过 `POST /chat` 发起对话；运行时会根据会话挂载源动态组装 candidate tools
   传 `stream=true` 时，接口会以 SSE 返回 `start`、`chunk`、`done`、`error` 四类事件，仅最终回答阶段流式输出
4. 如需知识增强，先通过 `POST /files/upload` 上传知识文件
5. 通过 `POST /knowledge/documents/preprocess-preview` 预览清洗规则、样本和统计
6. 通过 `POST /knowledge/documents` 确认入库；后续按需调用 `.../reprocess` 或 `.../rechunk`
7. 查看 `/chat` 响应中的 `citations` 与回答正文里的 `[1]`、`[2]` 编号，完成来源追溯

知识库管理页当前交互：

- 上传 `json`、`csv`、`txt`、`md` 后会自动打开“数据预处理”弹窗
- 弹窗中可查看 `supported_rules`、切换 `processing_rules`、刷新预览并确认入库
- 未入库但可处理的文件状态为 `awaiting_processing`
- `pdf`、`docx`、`xlsx` 当前允许上传，但仅显示 `unsupported`，不会进入预处理与索引链路

## 向量存储

默认使用 `chroma`：

```env
AI_RAG_VECTOR_STORE__PROVIDER=chroma
AI_RAG_VECTOR_STORE__CHROMA__PERSIST_DIRECTORY=backend/data/.chroma
```

如需切换到 `Elasticsearch`：

```env
AI_RAG_VECTOR_STORE__PROVIDER=elasticsearch
AI_RAG_VECTOR_STORE__ELASTICSEARCH__URL=http://127.0.0.1:9200
```

本地启动 Elasticsearch：

```powershell
docker compose -f docs\elasticsearch\docker-compose.yml up -d
```

## 测试

全量后端测试：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

单文件测试示例：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

最小 SSE 验证：

```powershell
$body = @{ message = "推荐续航好的手机"; stream = $true } | ConvertTo-Json
Invoke-WebRequest -Uri http://127.0.0.1:8000/chat -Method Post -ContentType "application/json" -Body $body
```

说明：

- `stream=true` 的响应头应包含 `Content-Type: text/event-stream`
- 事件顺序应为 `start -> chunk... -> done`
- 客户端应以 `done.answer` 作为最终权威文本；若中途失败，会收到 `error` 事件

## 开发说明

- 后端顶层代码以 `application / platform / scenes` 三层为核心组织方式
- 修改架构、启动方式、环境变量或测试命令时，应同步更新 `README.md`、`AGENTS.md` 和 `backend/.env.example`
- `__init__.py` 应保持轻量，避免在包初始化阶段引入运行时装配逻辑
- 知识文档写流程统一由 `KnowledgeDocumentApplicationService` 编排，并通过 `KnowledgeDocumentPublisher` 发布新版本；不要恢复聚合式文档服务
- 文档预处理能力位于 `backend/platform/knowledge/processing`，新增规则或统计逻辑优先落在该层

## 适用场景

如果你想基于这个仓库继续扩展，通常会从以下方向入手：

- 新增一个 `scene`，构建新的行业助手
- 扩展 `platform/knowledge`，增加新的文档处理能力
- 扩展 `platform/rag`，调整检索策略
- 增加前端页面或接入自己的业务 UI

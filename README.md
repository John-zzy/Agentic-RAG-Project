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

当前后端提供统一的聊天、会话、文件和知识文档接口，并支持基于场景的能力编排。

核心能力包括：

- 统一 `/chat` 入口，按会话绑定场景处理请求
- 会话创建、查询、删除与场景选择
- 本地知识文件上传、下载、删除与索引管理
- 文档切块、重建索引、向量检索
- `Chroma` 与 `Elasticsearch` 两种向量存储实现
- 基于 `SQLite` 的会话记忆

当前内置场景：

- `generic_assistant`：通用助手场景，依赖通用知识与会话记忆
- `ecommerce`：电商演示场景，包含商品、评价、订单、库存等检索与工具能力

## 设计文档

如果你希望先从设计层面理解这个项目，而不是直接读代码，可以先看下面的文档：

- [Agentic RAG 设计说明](./docs/agentic_rag.md)：解释本项目在多轮召回、工具切换、query 改写、证据聚合和最终回答生成上的完整链路

## 系统架构

后端采用三层结构，每层职责明确：

### 1. Platform

`backend/platform` 提供与具体场景无关的底层能力，包括：

- 配置加载与模型路由
- LLM 客户端封装
- 会话存储与聊天上下文
- 通用知识文档处理
- RAG 检索核心协议与实现

### 2. Application

`backend/application/runtime` 负责运行时装配，包括：

- 应用启动引导
- active scene 默认选择
- Chat service 组装
- FastAPI 应用与 API 路由注册

### 3. Scenes

`backend/scenes` 放置具体场景定义，包括：

- 场景提示词与定义
- 场景级检索工具
- 场景知识组织方式
- 场景特有的工具与服务

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
2. 通过 `POST /sessions` 创建会话并指定场景
3. 通过 `POST /chat` 发起对话
4. 如需知识增强，先上传文件或创建知识文档索引

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

## 开发说明

- 后端顶层代码以 `application / platform / scenes` 三层为核心组织方式
- 修改架构、启动方式、环境变量或测试命令时，应同步更新 `README.md`、`AGENTS.md` 和 `backend/.env.example`
- `__init__.py` 应保持轻量，避免在包初始化阶段引入运行时装配逻辑

## 适用场景

如果你想基于这个仓库继续扩展，通常会从以下方向入手：

- 新增一个 `scene`，构建新的行业助手
- 扩展 `platform/knowledge`，增加新的文档处理能力
- 扩展 `platform/rag`，调整检索策略
- 增加前端页面或接入自己的业务 UI

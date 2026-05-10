# AI RAG Project

一个面向电商客服场景的 AI RAG 项目，用于演示“知识库检索 + 会话记忆 + 大模型回答”这一条最小可运行链路。当前仓库以后端 MVP 为主，覆盖商品导购、评价检索、多轮对话、会话管理等基础能力，适合作为面试展示、RAG 工程练习和后续多 Agent 扩展的起点。

## 项目亮点

- 基于 `FastAPI` 提供可直接联调的对话 API
- 支持商品与评价知识的 RAG 检索增强回答
- 支持上传本地知识文件并通过独立管理页完成建索引、重建和删除
- 内置 `Chroma` 与 `Elasticsearch` 两种向量存储后端
- 使用 `SQLite` 持久化会话上下文，支持多轮对话
- 提供静态 API 测试页，便于本地快速验证接口
- 代码结构清晰，已拆分为 API、知识库、记忆、模型路由、配置等模块

## 当前功能

目前已经实现的能力：

- `POST /chat`：接收用户问题，执行检索并返回回答
- `GET /health`：健康检查
- `POST /sessions`：创建会话
- `GET /sessions/{session_id}`：查看会话历史
- `DELETE /sessions/{session_id}`：删除会话
- 启动时自动预加载本地商品与评价数据到知识库
- `POST /files/upload`、`GET /files`、`DELETE /files/{filename}`：上传和管理本地知识文件
- `POST /knowledge/documents`、`GET /knowledge/documents`、`GET /knowledge/documents/files`：注册知识文档、查看文档列表和按文件聚合索引状态
- `GET /knowledge/documents/{document_id}`、`POST /knowledge/documents/{document_id}/rechunk`、`DELETE /knowledge/documents/{document_id}`：查看详情、重建切分和删除知识文档
- 本地挂载前端测试页：`/frontend/api-tester.html`
- 本地挂载知识库管理页：`/frontend/knowledge-manager.html`

当前尚未完成的部分：

- 多 Agent 协同路由
- MCP 工具调用链
- 完整监控与生产化部署流程

## 技术栈

- Python 3.11+
- FastAPI
- LangChain / LangGraph
- ChromaDB / Elasticsearch
- SQLite
- DashScope 兼容模型接口
- Pytest

## 目录结构

```text
.
├── backend/                # 后端主代码
│   ├── api/                # 路由、Schema、聊天服务
│   ├── config/             # 应用配置
│   ├── knowledge/          # 知识库、向量检索、预加载
│   ├── memory/             # 会话存储与上下文构建
│   ├── models/             # 模型路由与客户端封装
│   ├── tests/              # pytest 测试
│   ├── .env.example        # 环境变量示例
│   └── run.py              # 启动入口
├── frontend/               # 对话测试页与知识库管理页
├── docs/elasticsearch/     # 本地 Elasticsearch docker compose
├── openspec/               # 需求变更与实现任务文档
└── README.md
```

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd ai-rag-project
```

### 2. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

如果你使用 Git Bash 或 macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

### 3. 配置环境变量

复制示例文件并填写模型 API Key：

```powershell
Copy-Item backend\.env.example backend\.env
```

至少需要配置：

```env
AI_RAG_MODELS__SIMPLE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__MODERATE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__COMPLEX__API_KEY=your-dashscope-api-key
AI_RAG_VECTOR_STORE__PROVIDER=chroma
```

默认使用 `chroma`。如果要切换到 Elasticsearch，请改为：

```env
AI_RAG_VECTOR_STORE__PROVIDER=elasticsearch
AI_RAG_VECTOR_STORE__ELASTICSEARCH__URL=http://127.0.0.1:9200
```

### 4. 启动项目

```powershell
python backend\run.py
```

启动后默认监听：

- API: `http://127.0.0.1:8000`
- Swagger 文档: `http://127.0.0.1:8000/docs`
- 对话测试页: `http://127.0.0.1:8000/frontend/api-tester.html`
- 知识库管理页: `http://127.0.0.1:8000/frontend/knowledge-manager.html`

## 知识文档管理

项目当前支持把上传到 `backend/data/files` 的文本类文件注册到“我的文档知识库”，并通过独立管理页完成索引管理。

- 支持上传的文件类型：`.json`、`.txt`、`.md`、`.csv`、`.pdf`、`.docx`、`.xlsx`
- 当前支持建索引的文件类型：`.json`、`.txt`、`.md`、`.csv`
- 上传文件和建索引是两步操作：先通过 `/files/upload` 上传，再通过 `/knowledge/documents` 注册入库
- 文档管理接口默认懒加载，单独访问聊天 API 时不会因为文档索引初始化失败而阻塞启动

一个典型流程如下：

1. 打开 `http://127.0.0.1:8000/frontend/knowledge-manager.html`
2. 上传知识文件到 `backend/data/files`
3. 在文件列表中选择可索引文件，填写 `namespace`、`chunk_size`、`chunk_overlap`
4. 点击“建索引”或对已有文档执行“重建”

`namespace` 需要使用仅包含小写字母、数字和下划线的 slug，例如 `faq`、`manual_v2`。

## 可选：启动 Elasticsearch

如果你想验证 Elasticsearch 向量检索后端，可以先启动本地容器：

```powershell
docker compose -f docs\elasticsearch\docker-compose.yml up -d
```

停止容器：

```powershell
docker compose -f docs\elasticsearch\docker-compose.yml down
```

## API 示例

请求：

```http
POST /chat
Content-Type: application/json
```

```json
{
  "message": "推荐一款续航好的安卓手机",
  "session_id": "optional-session-id",
  "stream": false
}
```

响应：

```json
{
  "session_id": "6b4d3d6d5e3947d49e3d5e2ed5b1b0f1",
  "request_id": "4b2b8b471a9f4f0ea1f6fe8b74a9194a",
  "answer": "推荐 P001，续航表现较好。",
  "knowledge_used": true,
  "citations": [
    {
      "citation_id": "P001",
      "namespace": "products",
      "snippet": "P001 手机，续航强，电池 5000mAh。",
      "score": 0.92
    }
  ]
}
```

## 测试

运行单元测试：

```powershell
python -m pytest backend\tests -q -c backend\tests\pytest.ini
```

## 项目状态

这是一个正在持续补齐功能的 MVP。当前重点是把单体 RAG 闭环打稳，包括知识检索、会话记忆和接口联调；多 Agent、MCP 工具层、监控与更完整的工程化能力仍在后续规划中。

如果你想继续了解后端实现细节，可以查看 [backend/README.md](backend/README.md)。

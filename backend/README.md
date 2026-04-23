# AI RAG Project Backend

基于 LangGraph 的电商多 Agent 智能客服后端 MVP，面向导购、订单查询、售后处理三类核心客服场景。当前阶段已完成基础项目初始化、模拟数据准备、知识库模块，以及基线对话 API（用户输入 -> 检索 -> Prompt 上下文 -> 回答）。后续将逐步补齐 MCP 工具层、多 Agent 协同、监控与完整生产化能力。

## 项目目标

- 通过多 Agent 协同处理导购、订单、售后等电商客服任务
- 使用本地模拟数据构建可检索的商品与评价知识库
- 提供可演示、可扩展、便于面试展示的后端工程骨架
- 保持本地优先：FastAPI + ChromaDB/Elasticsearch + SQLite

## 当前目录结构

```text
backend/
├── agents/         # 多 Agent 实现（待补充）
├── api/            # FastAPI 应用入口、路由与聊天编排
├── config/         # 统一配置管理
├── data/           # 模拟商品、订单、评价数据
├── knowledge/      # 知识库与向量检索模块（已支持 Chroma + Elasticsearch）
├── mcp/            # MCP 工具层与服务端（待补充）
├── memory/         # 会话存储与 Prompt 上下文管理
├── models/         # 模型路由与调用客户端
├── monitoring/     # Prometheus 指标与监控（待补充）
├── tests/          # 测试包、pytest 配置与测试辅助文件
├── config/settings.py
├── requirements.txt
└── README.md
```

## 已完成内容

### 1. 基础项目初始化

- 创建后端模块目录与 `__init__.py`
- 创建 `requirements.txt`
- 创建统一配置文件 `config/settings.py`

### 2. 模拟数据准备

当前已提供 3 份本地 JSON 数据，可作为后续知识库构建、工具调用和接口联调的数据源。

#### `data/products.json`

- 包含 20 个模拟商品
- 覆盖笔记本、平板、手机、耳机、音箱、显示器、键盘、鼠标、智能家居、智能穿戴等品类
- 每个商品包含：
  - `product_id`
  - `name`
  - `category`
  - `description`
  - `price`
  - `currency`
  - `specs`
  - `inventory`

其中库存信息已内嵌到商品数据中，满足模拟库存场景需要。

#### `data/orders.json`

- 包含 10 个模拟订单
- 覆盖以下订单状态：
  - `待付款`
  - `已发货`
  - `运输中`
  - `已签收`
- 每个订单包含用户、商品项、金额、地址、物流等字段，便于后续订单查询与售后工具使用

#### `data/reviews.json`

- 包含与商品关联的用户评价数据
- 每条评价包含：
  - `review_id`
  - `product_id`
  - `rating`
  - `title`
  - `content`
  - `user_name`
  - `created_at`

这些评价数据可直接用于后续向量化、商品口碑摘要和推荐增强

### 3. 知识库向量存储（Chroma + Elasticsearch）

当前知识库模块已经支持：

- 统一 `VectorStore` 抽象和 `VectorStoreFactory`
- `ChromaVectorStore` 与 `ElasticsearchVectorStore` 双后端实现
- `knowledge/service.py` 统一检索与增量更新服务入口（仅依赖 `VectorStore` 接口）
- `knowledge/extractor.py` 对商品与评价 JSON 的标准化文档转换
- `knowledge/loader.py` 通过工厂完成 `products.json` 与 `reviews.json` 的预加载
- 本地哈希 embedding，用于开发阶段验证向量存储与检索流程

与 OpenSpec `ecommerce-customer-service-agent` 的 3.x 任务对齐关系如下：

- 3.1：`knowledge/store.py` 已实现 Chroma 初始化、collection 管理、文档 upsert 与语义检索
- 3.2：`knowledge/extractor.py` 已实现商品与评价 JSON 的标准化字段提取与结构化文档构建
- 3.3：`knowledge/loader.py` 已实现 `products.json` 与 `reviews.json` 的预加载向量化脚本入口 `preload_knowledge_base`
- 3.4：`knowledge/service.py` 已实现商品/评价数据的增量 upsert 接口（`upsert_products`、`upsert_reviews`）

### 4. 基线对话 API（Chat RAG Baseline）

当前已实现最小可交付对话闭环：

- `POST /chat`：接收用户问题，执行知识库检索，构建 Prompt 上下文并生成回答
- `GET /health`：健康检查
- 会话持久化：按 `session_id` 保存每轮 `user_message/assistant_answer/retrieval_snippets/timestamp`
- Prompt 组装：合并历史窗口、本轮输入和检索片段，支持窗口裁剪
- LangChain RAG 编排：通过 `BaseRetriever + create_retrieval_chain` 组合执行检索增强回答
- LangChain 模型执行：通过 `PromptTemplate + ChatOpenAI + StrOutputParser` 执行链完成同步调用
- 默认同步调用：`POST /chat` 当前默认非流式返回，`stream=true` 会返回未启用提示

关键实现文件：

- `api/app.py`、`api/routes.py`、`api/chat_service.py`、`api/schemas.py`、`api/prompts.py`
- `knowledge/retriever.py`（`KnowledgeBaseRetriever`）
- `memory/session_store.py`、`memory/prompt_context.py`
- `models/client.py`（LangChain 模型调用客户端）

## 环境准备

### Python 解析器

使用 Miniconda3 的 Python 解析器创建本地 `venv` 虚拟环境。

Windows 示例：

```bash
"C:/Users/<your-user>/miniconda3/python.exe" -m venv .venv
```

如果 Miniconda3 已加入 PATH，也可以直接使用：

```bash
python -m venv .venv
```

### 激活虚拟环境

Git Bash：

```bash
source .venv/Scripts/activate
```

PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

### 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 运行测试

```bash
python -m pytest backend/tests -q -c backend/tests/pytest.ini -p no:tmpdir --ignore backend/tests/.pytest_tmp_local
```

### 配置环境变量

在 `backend/.env` 中配置 DashScope API Key：

```env
AI_RAG_MODELS__SIMPLE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__MODERATE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__COMPLEX__API_KEY=your-dashscope-api-key
AI_RAG_VECTOR_STORE__PROVIDER=chroma
AI_RAG_VECTOR_STORE__CHROMA__PERSIST_DIRECTORY=backend/data/.chroma
AI_RAG_VECTOR_STORE__ELASTICSEARCH__URL=http://localhost:9200
```

如果三个复杂度统一使用同一个 Key，以上三个值可以保持一致。
向量库默认使用 `chroma`。切换到 Elasticsearch 时，将 `AI_RAG_VECTOR_STORE__PROVIDER=elasticsearch` 并补齐 ES 连接参数。

推荐切换方式与约束：
1. 前期开发/初始化阶段使用 `chroma`，先完成数据预加载与本地联调。
2. 后期需要索引治理时切到 `elasticsearch`，并确保 `http://localhost:9200`（或目标 ES 地址）可达。
3. provider 切换只改变后端实现，不改变上层调用契约（`search/upsert/delete` 接口保持一致）。
4. 当前不会自动迁移 Chroma 历史数据到 Elasticsearch；切换后需重新预加载或使用独立迁移脚本。

## 当前依赖

- `langchain`
- `langchain-openai`
- `langgraph`
- `chromadb`
- `elasticsearch`
- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `pydantic-settings`
- `python-dotenv`
- `prometheus-client`
- `dashscope`

## 配置说明

核心配置位于 `config/settings.py`，当前包含：

- 应用基础配置：名称、环境、监听地址、端口
- 数据目录配置：`data_dir`
- 临时数据目录约定：Chroma 向量文件等运行时数据统一落到 `backend/data/`
- 统一向量存储配置：`vector_store.provider`、Top-K、商品/评价命名空间的 collection 与 index 命名
- ChromaDB 持久化目录配置：`vector_store.chroma.persist_directory`
- Elasticsearch 连接与索引前缀配置：`vector_store.elasticsearch.url`、认证信息、请求超时、`index_prefix`
- SQLite 会话存储路径
- 会话超时与窗口大小
- 简单 / 中等 / 复杂任务的模型路由占位配置（当前默认使用 DashScope 的 `qwen-turbo`、`qwen-plus`、`qwen-max`）
- 模型调用执行方式：LangChain（非直接 OpenAI SDK 调用）

### 知识库存储抽象

当前已在 `knowledge/store.py` 中定义：

- `VectorStore`：统一的向量库抽象接口
- `VectorStoreDocument` / `VectorSearchResult`：上层通用文档与检索结果模型
- `VectorStoreFactory`：根据 `vector_store.provider` 选择后端实现
- `ChromaVectorStore`：默认本地向量存储实现
- `ElasticsearchVectorStore`：Elasticsearch 索引管理与检索实现
- `KnowledgeService`：统一检索与增量更新服务，屏蔽底层后端差异
- `preload_knowledge_base`：基于统一工厂的知识库预加载入口

### 测试目录约定

- 所有测试文件、pytest 配置和测试辅助代码统一放在 `backend/tests/`
- 测试运行时产物统一放在 `backend/tests/artifacts/`、`backend/tests/.pytest_tmp/` 和 `backend/tests/.pytest_cache/`

## 后续开发建议

推荐按以下顺序继续推进：

1. 继续完善知识库迁移能力（例如 Chroma -> Elasticsearch 的离线迁移脚本）
2. 实现 MCP 工具层，先打通商品查询、库存查询、订单状态查询
3. 引入多 Agent 路由与 Agent 间上下文交接
4. 接入监控指标（`/metrics`）与成本统计
5. 接入 Hermes 类似流程，实现智能客服自主迭代

## API 基线说明

当前已实现的基线接口：

- `POST /chat`：多轮对话主入口
- `GET /health`：健康检查

`POST /chat` 请求示例：

```json
{
  "message": "推荐一款续航好的安卓手机",
  "session_id": "optional-session-id",
  "stream": false
}
```

`POST /chat` 响应示例：

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

调试与人工复核建议：

- `knowledge_used=true` 表示本轮回答已使用检索知识；`false` 表示走无命中降级回答
- `citations` 用于查看回答引用来源（商品/评价）与片段内容，便于人工核验

当前基线明确不包含：

- 多 Agent 路由与 Agent 交接
- MCP 工具调用
- 监控大盘与复杂成本统计

## 说明

- 当前数据均为本地模拟数据，仅用于开发与演示
- 当前 README 反映的是基线 API 阶段状态，后续随着多 Agent、MCP、监控模块落地，需要继续同步更新
- 当前尚未提供统一 `run.py` 一键启动入口，可先使用 `uvicorn backend.api.app:app --reload` 启动 API

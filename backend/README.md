# AI RAG Project Backend

基于 LangGraph 的电商多 Agent 智能客服后端 MVP，面向导购、订单查询、售后处理三类核心客服场景。当前阶段已完成基础项目初始化与模拟数据准备，后续将逐步补齐知识库、MCP 工具层、会话记忆、模型路由、监控与 API。

## 项目目标

- 通过多 Agent 协同处理导购、订单、售后等电商客服任务
- 使用本地模拟数据构建可检索的商品与评价知识库
- 提供可演示、可扩展、便于面试展示的后端工程骨架
- 保持本地优先：FastAPI + ChromaDB + SQLite

## 当前目录结构

```text
backend/
├── agents/         # 多 Agent 实现（待补充）
├── api/            # FastAPI 应用入口与路由（待补充）
├── config/         # 统一配置管理
├── data/           # 模拟商品、订单、评价数据
├── knowledge/      # 知识库与向量检索模块（待补充）
├── mcp/            # MCP 工具层与服务端（待补充）
├── memory/         # 会话存储与记忆管理（待补充）
├── models/         # 模型路由配置（待补充）
├── monitoring/     # Prometheus 指标与监控（待补充）
├── tests/          # 测试（待补充）
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

### 配置环境变量

在 `backend/.env` 中配置 DashScope API Key：

```env
AI_RAG_MODELS__SIMPLE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__MODERATE__API_KEY=your-dashscope-api-key
AI_RAG_MODELS__COMPLEX__API_KEY=your-dashscope-api-key
```

如果三个复杂度统一使用同一个 Key，以上三个值可以保持一致。

## 当前依赖

- `langchain`
- `langgraph`
- `chromadb`
- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `pydantic-settings`
- `python-dotenv`
- `prometheus-client`
- `openai`
- `dashscope`

## 配置说明

核心配置位于 `config/settings.py`，当前包含：

- 应用基础配置：名称、环境、监听地址、端口
- 数据目录配置：`data_dir`
- ChromaDB 持久化目录与 collection 名称
- SQLite 会话存储路径
- 会话超时与窗口大小
- 简单 / 中等 / 复杂任务的模型路由占位配置（当前默认使用 DashScope 的 `qwen-turbo`、`qwen-plus`、`qwen-max`）

## 后续开发建议

推荐按以下顺序继续推进：

1. 实现 `knowledge/store.py` 与 `knowledge/extractor.py`
2. 编写模拟数据预加载逻辑，将商品与评价写入 ChromaDB
3. 实现 MCP 工具层，先打通商品查询、库存查询、订单状态查询
4. 补充 FastAPI 应用入口与聊天接口
5. 最后接入监控与测试

## API 规划

当前仓库尚未完成 API 层实现，规划中的接口包括：

- `POST /chat`：多轮对话主入口
- `GET /health`：健康检查
- `GET /metrics`：Prometheus 指标
- 会话管理接口：创建、查询、清理会话

## 说明

- 当前数据均为本地模拟数据，仅用于开发与演示
- 当前 README 反映的是 MVP 初始阶段状态，后续随着知识库、MCP、API 与测试模块落地，需要继续同步更新
- 当前尚未提供一键启动入口，待 `run.py` 与 API 层完成后补齐

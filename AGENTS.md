# Repository Guidelines

## 维护原则
这份文档的首要目标是让 AI 快速定位需要修改的代码位置，而不是泛泛列举目录。接到任务后，先根据“任务到代码位置”找到最小改动面，再读取相关文件；不要从仓库根目录递归扫 `backend/.venv/`、`backend/data/` 或 `backend/tests/artifacts/`。

## 任务到代码位置
改 API 路由、请求参数、响应字段或 HTTP 错误处理时，优先看 `backend/api/routes.py` 和 `backend/api/schemas.py`。`routes.py` 定义 `/health`、`/chat`、`/sessions` 等 FastAPI endpoint；`schemas.py` 定义 `ChatRequest`、`ChatResponse`、会话响应模型。对应测试通常在 `backend/tests/test_chat_api.py`。

改聊天主流程、RAG 编排、引用返回、会话读写串联或模型调用时，优先看 `backend/api/chat_service.py`。它是 `ChatService` 的主实现，连接 `KnowledgeService`、`SQLiteSessionStore`、`PromptContextBuilder` 和模型链。提示词模板在 `backend/api/prompts.py`，上下文裁剪逻辑在 `backend/memory/prompt_context.py`。

改会话创建、恢复、过期、历史窗口、SQLite 持久化或旧库迁移时，优先看 `backend/memory/session_store.py`。会话窗口只影响提示词上下文时，同时看 `backend/memory/prompt_context.py`。对应测试在 `backend/tests/test_session_store.py` 和 `backend/tests/test_prompt_context_builder.py`。

改知识库检索、文档入库、过滤、引用片段或向量库抽象时，优先看 `backend/knowledge/service.py` 和 `backend/knowledge/store.py`。`service.py` 是业务入口，`store.py` 定义 `VectorStore` 合约以及 `ChromaVectorStore`、`ElasticsearchVectorStore`、`VectorStoreFactory`。数据预加载在 `backend/knowledge/loader.py`，商品和评论转文档在 `backend/knowledge/extractor.py`，LangChain retriever 适配在 `backend/knowledge/retriever.py`。

改 Chroma 或 Elasticsearch 行为时，主要看 `backend/knowledge/store.py`。Chroma 相关实现集中在 `ChromaVectorStore`，Elasticsearch 相关实现集中在 `ElasticsearchVectorStore`，provider 选择在 `VectorStoreFactory.create()`。对应测试在 `backend/tests/test_knowledge_chroma.py`、`backend/tests/test_knowledge_elasticsearch.py`、`backend/tests/test_knowledge_vector_store_contract.py`。

改模型供应商、模型路由、API Key、流式输出或模型客户端时，优先看 `backend/models/router.py`、`backend/models/client.py`、`backend/config/model_routing.json` 和 `backend/config/settings.py`。`router.py` 负责按任务复杂度选择模型，`client.py` 负责实际模型调用封装，`settings.py` 负责从 `.env` 和 JSON 配置装载模型端点。

改环境变量、默认路径、端口、会话配置、向量库配置或配置解析时，优先看 `backend/config/settings.py`。不要在业务代码里硬编码路径或配置值；应通过 `AppSettings`、`settings.session`、`settings.vector_store` 或相关配置模型传入。

改应用启动、生命周期、静态页面挂载或启动前预加载时，优先看 `backend/run.py` 和 `backend/api/app.py`。`run.py` 负责 runtime bootstrap、知识库预加载和 uvicorn 启动；`app.py` 负责创建 FastAPI app、注册 router、挂载 `frontend/` 到 `/frontend`。

改静态调试页面时，只看 `frontend/api-tester.html`。它是当前前端入口，用来手动调用后端 API；后端静态挂载逻辑在 `backend/api/app.py`。

改示例数据或本地知识源时，看 `backend/data/products.json`、`backend/data/reviews.json`、`backend/data/orders.json`。这些是样例数据文件；不要把运行生成的 `sessions.db`、`.chroma/`、`elasticsearch/` 当作源码改动。

改 Elasticsearch 本地开发编排时，看 `docs/elasticsearch/docker-compose.yml` 和 `docs/elasticsearch/README.md`。不要直接修改 `backend/data/elasticsearch/` 下的运行数据。

改需求或设计文档时，看 `openspec/changes/<change-id>/proposal.md`、`design.md`、`tasks.md` 和对应 `specs/**/spec.md`。已归档内容在 `openspec/changes/archive/`，除非任务明确要求，不要修改归档变更。

## 修改入口速查
新增或调整 endpoint：`backend/api/routes.py` -> `backend/api/schemas.py` -> `backend/tests/test_chat_api.py`。

新增聊天返回字段：`backend/api/schemas.py` -> `backend/api/chat_service.py` -> `backend/tests/test_chat_api.py`。

调整 RAG 检索策略：`backend/api/chat_service.py` -> `backend/knowledge/service.py` -> `backend/knowledge/store.py` -> 知识库测试。

调整提示词：`backend/api/prompts.py`；如果涉及历史上下文，再看 `backend/memory/prompt_context.py`。

调整会话生命周期：`backend/memory/session_store.py` -> `backend/config/settings.py` -> `backend/tests/test_session_store.py`。

新增向量库 provider：`backend/knowledge/store.py` 的 `VectorStore` 合约和 `VectorStoreFactory` -> `backend/config/settings.py` 的配置模型 -> provider 合约测试。

调整模型选择：`backend/models/router.py` -> `backend/config/model_routing.json` -> `backend/config/settings.py`。

调整模型调用实现：`backend/models/client.py`；涉及调用方行为时再看 `backend/api/chat_service.py`。

调整启动时初始化：`backend/run.py`；涉及 app 生命周期或静态挂载时再看 `backend/api/app.py`。

## 不要优先查看或修改的目录
`backend/.venv/` 是本地虚拟环境，不属于项目源码。`backend/data/.chroma/`、`backend/data/elasticsearch/`、`backend/data/sessions.db` 是运行时数据。`backend/tests/artifacts/` 是测试产物。`__pycache__/` 是 Python 缓存。除非用户明确要求处理运行数据或环境问题，不要读取、编辑或提交这些内容。

## 构建、测试与开发命令
先创建虚拟环境并安装后端依赖：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

本地启动 API：

```powershell
python backend\run.py
```

运行测试：

```powershell
python -m pytest backend\tests -q -c backend\tests\pytest.ini
```

如需启用 Elasticsearch，本地启动命令如下：

```powershell
docker compose -f docs\elasticsearch\docker-compose.yml up -d
```

## 代码风格与命名约定
沿用当前 Python 风格：4 空格缩进、补全类型标注、函数和模块使用 `snake_case`、类使用 `PascalCase`。优先写职责单一的小函数，导入保持显式，路径处理优先使用 `Path`，不要硬编码字符串路径。仓库当前未提交格式化或 lint 配置，新增代码时以相邻文件风格为准。
新增或修改类、方法时，应补充简洁明确的中文注释或 docstring，说明该类/方法的职责、使用场景或关键输入输出；避免只写重复代码字面的无效注释。目标是让后续阅读者不需要反推实现细节，也能快速理解“这个类/方法是干什么的”。

## 测试规范
使用 `pytest`，测试文件统一放在 `backend/tests/` 下，命名采用 `test_*.py`，例如 `test_chat_api.py`、`test_session_store.py`。测试文件名应尽量对应被测模块。调用外部服务的用例请标记 `@pytest.mark.integration`；默认测试配置会跳过这类用例。

## 安全与配置提示
敏感配置放在 `backend/.env`，不要提交 API Key。开发环境默认使用 `chroma`；切换到 Elasticsearch 时，设置 `AI_RAG_VECTOR_STORE__PROVIDER=elasticsearch`，并同时配置 `AI_RAG_VECTOR_STORE__ELASTICSEARCH__URL`。

## LLM 编码行为约束
以下约束来自 `CLAUDE.md`，用于降低常见 LLM 编码错误。与项目特定说明冲突时，优先遵守本文件中更具体的项目规则。

### 先思考再编码
不要假设，不要隐藏困惑，明确暴露权衡。实现前应显式说明假设；如果存在多种解释，应先列出而不是静默选择；如果有更简单的方案，应主动说明；如果需求不清楚，应停下来指出不清楚之处并询问。

### 简单优先
只写解决问题所需的最少代码，不做推测性扩展。不要添加未被要求的功能，不要为一次性代码抽象，不要加入未被要求的灵活性或配置能力，不要为不可能发生的场景添加复杂错误处理。如果代码明显可以从 200 行简化到 50 行，应重写为更简单的实现。

### 外科手术式修改
只修改必须修改的内容，只清理自己造成的问题。编辑现有代码时，不要顺手改进相邻代码、注释或格式；不要重构未损坏的代码；保持现有风格，即使个人偏好不同；发现无关死代码时，只说明，不要删除。若本次修改导致导入、变量或函数变为未使用，应清理这些由本次修改造成的孤儿代码。

每一行变更都应能直接追溯到用户请求。

### 目标驱动执行
把任务转换为可验证目标并循环到验证完成。例如，“添加校验”应对应“为非法输入写测试，然后让测试通过”；“修复 bug”应对应“写出能复现问题的测试，然后让测试通过”；“重构 X”应对应“确保重构前后测试通过”。

多步骤任务应给出简短计划：

```text
1. [步骤] -> verify: [检查]
2. [步骤] -> verify: [检查]
3. [步骤] -> verify: [检查]
```

强成功标准能支持独立迭代；弱标准，例如“让它能工作”，需要先澄清。

这些约束的目标是减少不必要 diff、避免过度复杂实现，并让澄清问题发生在实现前而不是返工后。

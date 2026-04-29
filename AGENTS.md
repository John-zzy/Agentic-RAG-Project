# Repository Guidelines

## 项目结构与模块组织
`backend/` 是核心应用目录。`api/` 负责 FastAPI 路由与聊天编排，`knowledge/` 负责检索与向量库集成，`memory/` 管理会话状态，`models/` 处理模型路由，`config/` 维护配置。测试集中在 `backend/tests/`。`frontend/` 目前只有一个静态调试页 `api-tester.html`，启动后挂载到 `/frontend`。`docs/elasticsearch/` 保存本地 Elasticsearch 编排文件，`openspec/` 用于跟踪需求变更与实现任务。

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

## 测试规范
使用 `pytest`，测试文件统一放在 `backend/tests/` 下，命名采用 `test_*.py`，例如 `test_chat_api.py`、`test_session_store.py`。测试文件名应尽量对应被测模块。调用外部服务的用例请标记 `@pytest.mark.integration`；默认测试配置会跳过这类用例。

## 安全与配置提示
敏感配置放在 `backend/.env`，不要提交 API Key。开发环境默认使用 `chroma`；切换到 Elasticsearch 时，设置 `AI_RAG_VECTOR_STORE__PROVIDER=elasticsearch`，并同时配置 `AI_RAG_VECTOR_STORE__ELASTICSEARCH__URL`。

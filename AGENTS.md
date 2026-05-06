# AI RAG Project

一个面向电商客服场景的 AI RAG 项目，用于演示“知识库检索 + 会话记忆 + 大模型回答”这一条最小可运行链路。当前仓库以后端 MVP 为主，覆盖商品导购、评价检索、多轮对话、会话管理等基础能力，适合作为面试展示、RAG 工程练习和后续多 Agent 扩展的起点。

## 项目亮点

- 基于 `FastAPI` 提供可直接联调的对话 API
- 支持商品与评价知识的 RAG 检索增强回答
- 内置 `Chroma` 与 `Elasticsearch` 两种向量存储后端
- 使用 `SQLite` 持久化会话上下文，支持多轮对话
- 提供静态 API 测试页，便于本地快速验证接口
- 代码结构清晰，已拆分为 API、知识库、记忆、模型路由、配置等模块

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
├── frontend/               # 静态 API 测试页
├── docs/elasticsearch/     # 本地 Elasticsearch docker compose
├── openspec/               # 需求变更与实现任务文档
└── README.md
```

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

# 代码风格约束
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

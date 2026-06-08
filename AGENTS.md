# AI RAG Project Agent Guide

本文件服务 AI Agent：快速定位改动入口、遵守架构边界、避免高频误改。完整背景和图谱从 [docs/documents/README.md](./docs/documents/README.md) 渐进阅读；排障细节见 [common-pitfalls.md](./docs/documents/operations/common-pitfalls.md)。

## 文档索引

- 文档总索引：[docs/documents/README.md](./docs/documents/README.md)
- 系统架构图：[system-overview.svg](./docs/documents/architecture/system-overview.svg)
- Main Chat Runtime：[main-chat-agent-runtime-flow.svg](./docs/documents/runtime/main-chat-agent-runtime-flow.svg)
- ChatGraph SubGraphs：[chatgraph-subgraphs.svg](./docs/documents/runtime/chatgraph-subgraphs.svg)
- 知识流程：[knowledge-document-flow.svg](./docs/documents/knowledge/knowledge-document-flow.svg)
- Agentic RAG：[agentic-rag-retrieval-flow.svg](./docs/documents/rag/agentic-rag-retrieval-flow.svg)
- API 文档：[api-list.md](./docs/documents/reference/api-list.md)
- 数据模型：[data-model.md](./docs/documents/reference/data-model.md)
- 常见坑与排障：[common-pitfalls.md](./docs/documents/operations/common-pitfalls.md)

## 项目定位

多场景 RAG / Agent Runtime 示例项目，主线是 FastAPI 后端、统一 `/chat` 入口、本地知识文件上传与索引、向量检索和回答生成。`generic_assistant` 验证平台主链路，`ecommerce` 是“平台能力 + 场景扩展”的演示场景。

## 30 秒代码导航

先按目录定位层级，再用 `rg` 追具体类和函数。

```text
backend/
├─ run.py                         # 本地启动入口
├─ application/runtime/           # API 暴露、启动装配、ChatService 应用层
│  ├─ api/                        # FastAPI app、/chat、/sessions 等路由与 schema
│  ├─ assembly/                   # runtime/service factory、settings adapter、依赖注入
│  └─ service.py                  # /chat 主服务编排入口
├─ platform/                      # 通用平台能力，不感知具体业务 scene
│  ├─ agent_runtime/              # ChatGraph、ReAct、Plan、tool executor、RAG tool adapter
│  ├─ workflow/                   # workflow 状态机、LangGraph lifecycle/checkpoint
│  ├─ rag/                        # Agentic RAG、query rewrite、document retrieval、rerank 边界
│  ├─ knowledge/                  # 知识文档管理、发布、处理、底层仓储
│  ├─ memory/                     # session store、chat prompt context
│  ├─ tools/                      # 中立工具协议、adapter、registry
│  ├─ models/                     # 模型客户端与模型抽象
│  ├─ config/                     # 配置结构与读取
│  └─ search_foundation/          # 搜索底座能力
├─ scenes/                        # 场景定义、prompt、policy、业务工具
│  ├─ registry.py                 # 默认 scene 与扩展组合入口
│  ├─ generic_assistant/          # 平台主链路验证场景
│  └─ ecommerce/                  # 可选演示场景
├─ tests/                         # 后端测试
├─ evals/                         # 评测资产
└─ data/                          # 本地运行数据：sessions、checkpoint、files、vector store
```

常用下钻入口：

- 改 `/chat`：从 `backend/application/runtime/service.py`、`backend/application/runtime/api/chat/`、`backend/application/runtime/assembly/service_parts/` 开始。
- 改 ChatGraph / ReAct / Plan：从 `backend/platform/agent_runtime/chat_graph/`、`backend/platform/agent_runtime/react/graph/`、`backend/platform/agent_runtime/plan/graph/` 开始。
- 改 Workflow / HITL：从 `backend/platform/workflow/state_machine.py`、`backend/platform/workflow/langgraph/`、`backend/platform/agent_runtime/chat_graph/runtime.py` 开始。
- 改 RAG / Hybrid Search：从 `backend/platform/rag/orchestration/` 和 `backend/platform/rag/retrieval/documents/` 开始。
- 改知识文档入库：从 `backend/platform/knowledge/documents/` 和 `backend/platform/knowledge/processing/` 开始。
- 改 scene prompt 或业务工具：从 `backend/scenes/generic_assistant/definition.py`、`backend/scenes/ecommerce/definition.py` 和各自 `tools/` 开始。

## 架构边界

- `platform` 放通用底层能力，不感知具体业务场景。
- `application` 放运行时装配、API facade、settings adapter 和具体依赖注入。
- `scenes` 放场景定义、prompt、scene policy 和业务工具。
- `platform.agent_runtime.chat_graph` 可放中立 ChatGraph runtime、节点和状态投影。
- 不要把 API schema、scene 业务实现或配置读取塞进 `platform` 或 `__init__.py`。
- `__init__.py` 保持轻量，避免循环导入。

## RAG / Knowledge 边界

- 文档召回、关键词召回、Hybrid Search 统一进 `platform.rag.retrieval.documents`。
- rerank 边界统一进 `platform.rag.post_retrieval`。
- 文档检索统一通过 `DocumentRetrievalService` 进入，不要在 scene 里重造入口。
- `platform.knowledge` 只承接知识管理、底层存储和仓储访问，不新增面向 scene/chat 的检索 API。
- Agentic RAG 是 `platform.rag.orchestration` 下的编排层，不是与 Modular RAG 并列竞争的架构。

## Workflow / HITL 边界

- Workflow run state 使用 `backend/platform/workflow/state_machine.py` 的统一枚举和转移校验。
- `succeeded / failed / cancelled` 是终态，不能 resume、retry 或重新写成 `running`。
- `waiting_user` 是等待人工输入，不是失败。
- HITL `reject` 或 cancel 进入 `cancelled`，不是模型或工具错误。
- `sessions.status` 只表示聊天会话 active/expired，不能替代 workflow run state。
- `/chat/resume` 必须校验最新 checkpoint 的 `interrupt_id`，不能接受旧等待点。
- ReAct / Plan HITL 恢复要回到所属图节点继续执行，不要在 application 层用临时 handler 绕过 waiting turn/step。

## Session / Scene / Tools

- 新会话默认 scene 由 `AI_RAG_APP__ACTIVE_SCENE` 控制。
- 新会话默认挂载知识源是 `["documents"]`。
- `POST /sessions` 可通过 `mounted_knowledge_sources` 扩展到 `["documents", "ecommerce"]`。
- scene 负责 prompt 与运行时风格；知识源是否可用由会话挂载配置决定。
- candidate retrieval tools 由 `SceneDefinition` 根据 `mounted_knowledge_sources` 解析；runtime 不硬编码 knowledge source 到 tool name 的映射。
- `backend/platform/tools` 只放中立协议、adapter 和 registry，不引入具体业务工具。
- 每个逻辑工具对应一个独立类；scene definition 负责工具装配和范围控制。

## 常见坑索引

- 环境：运行后端和测试优先用 `backend\.venv\Scripts\python.exe`；命令默认在仓库根目录执行。
- 配置：本地启动前复制 `backend\.env.example` 到 `backend\.env`；DashScope key 至少覆盖 simple、moderate、complex、embedding，启用 rerank 时补 rerank key。
- 编码：仓库按 UTF-8 处理；Windows PowerShell 5.1 读中文或 Markdown 用 `Get-Content -Raw -Encoding UTF8 <file>`，写中文文件也显式指定 UTF-8，避免 `Set-Content` / `Out-File` 默认编码写出乱码或 BOM 差异。
- Git：工作区可能已有用户改动；先看 `git status --short`，不要执行 `git reset --hard` 或 `git checkout --` 回退无关内容。
- 文档：架构、接口、数据模型、运行方式、状态语义或环境变量变化后，同步检查 `README.md`、`AGENTS.md`、`backend/.env.example` 和 `docs/documents/`。
- API 字段：优先同步 `docs/documents/reference/api-list.md` 和 `docs/documents/reference/data-model.md`。
- 图表：Mermaid 图移动或改名后，同时维护 `.mmd` 和对应 `.svg`，并更新 README / AGENTS / 文档索引链接。
- 事实来源：历史方案在 `docs/plan/`，不要当成当前实现事实；当前事实优先看 `docs/documents/` 和代码。
- 基础设施：Docker Compose、本地基础设施和部署脚本统一放 `devops/`。
- SSE：`/chat?stream=true` 面向 UI，只依赖 `start`、`chunk`、可选安全 `thinking`、`waiting_user`、`done`、`error`；不要把历史窗口、工具参数或 retrieval trace 作为业务事件推给 UI。
- 生图：用户明确要求 `gpt-image-2` / `gpt-image2` 时按 `imagegen` skill 的 CLI fallback 处理；详细限制见 common-pitfalls。

## 数据位置

- 会话记忆：`backend/data/sessions.db`
- LangGraph checkpoint：`backend/data/langgraph.db`
- 文件上传目录：`backend/data/files`
- 向量存储默认 Chroma，可切换 Elasticsearch。
- Elasticsearch 本地说明：`devops/elasticsearch/README.md`

## 最小运行

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
backend\.venv\Scripts\python.exe backend\run.py
```

## 验证命令

全量测试：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

单文件示例：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

改 `/chat`、SSE、HITL 或 Workflow State Machine 后至少跑：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_generic_assistant_hitl.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

改 RAG 检索、rerank、citation 或 trace 后至少跑：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agentic_retrieval.py backend\tests\test_document_hybrid_retrieval.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

## 修改纪律

- 使用 `rg` 或 `rg --files` 搜索文本和文件。
- 使用 `apply_patch` 做手工文件修改。
- 避免大面积无关格式化 diff。
- 不要回退用户已有改动。
- 修改 `__init__.py` 时保持最小化，避免循环导入。
- PowerShell 5.1 处理中文文件时读写都显式用 `-Encoding UTF8`；不要基于乱码终端输出制作 patch。
- 更完整排障规则见 [common-pitfalls.md](./docs/documents/operations/common-pitfalls.md)。

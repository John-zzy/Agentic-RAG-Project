# 常见坑与排障

本文记录项目开发和 Agent 协作中最高频的错误。遇到异常时先检查这些项，避免把环境问题误判成业务缺陷。

## Python 环境

- 运行测试和后端时优先使用 `backend\.venv\Scripts\python.exe`，不要混用系统 Python。
- 安装依赖后仍报包缺失，先确认当前 shell 是否激活了 `backend\.venv`。
- 后端命令默认在仓库根目录执行；相对路径依赖这一约定。

## 配置与模型 Key

- 本地启动前需要从 `backend\.env.example` 复制 `backend\.env`。
- DashScope 相关 key 至少要覆盖 simple、moderate、complex、embedding；启用 rerank 时还要配置 rerank key。
- `AI_RAG_APP__ACTIVE_SCENE` 只控制新会话默认 scene，不是日常切换场景的主流程。

## 文档和索引

- 架构、接口、数据模型、运行方式或状态语义变化后，要同步检查 `README.md`、`AGENTS.md` 和 `docs/documents/` 下的模块文档。
- API 字段变化优先同步 `docs/documents/reference/api-list.md` 和 `docs/documents/reference/data-model.md`。
- Mermaid 图移动或改名后，要同时维护 `.mmd` 和对应 `.svg`，并更新 README / AGENTS / 文档索引链接。
- 历史方案文档在 `docs/plan/`，不要把旧方案当成当前实现事实；当前事实优先看 `docs/documents/` 和代码。

## 编码与 PowerShell

- 本仓库源码和文档按 UTF-8 处理。
- 在 Windows PowerShell 5.1 中，读取中文文档要显式使用 `Get-Content -Raw -Encoding UTF8 <file>`。
- 如果终端输出出现乱码，先判断是不是读取编码错误，不要直接认定文件已损坏。
- 不要基于乱码输出制作 patch；重新用 UTF-8 读取后再修改。

## Git 工作区

- 工作区可能已有用户或其他任务留下的未提交改动，不要执行 `git reset --hard` 或 `git checkout --` 回退它们。
- 做代码或文档修改前先看 `git status --short`，只处理当前任务相关文件。
- 如果同一个文件已有脏改，先读取当前内容再 patch，不要用旧片段覆盖。

## 架构边界

- 不要把运行时装配逻辑放进 `platform` 或 `__init__.py`。
- `platform.knowledge` 只负责知识管理和底层存储，不要新增面向 chat 的检索入口。
- 文档检索统一走 `DocumentRetrievalService`。
- scene 负责 prompt、scene policy 和业务工具装配；runtime 不硬编码业务工具选择。
- ToolRegistry 只维护工具注册、分组、白名单和 MCP 暴露标记，不承载业务逻辑。

## Workflow / HITL 状态

- `sessions.status` 只表示聊天会话 active/expired，不能替代 workflow run state。
- `waiting_user` 是等待人工输入，不是失败。
- `reject` 或 cancel 进入 `cancelled`，不是 `failed`。
- `succeeded / failed / cancelled` 是终态，不能继续 resume、retry 或写回 running。
- `/chat/resume` 必须校验最新 checkpoint 中的 `interrupt_id`，不能接受旧等待点。

## 测试

- 全量测试命令：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

- 改 `/chat`、SSE、HITL 或 Workflow State Machine 后，至少跑：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_generic_assistant_hitl.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

- 改 RAG 检索、rerank、citation 或 trace 后，至少跑：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agentic_retrieval.py backend\tests\test_document_hybrid_retrieval.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

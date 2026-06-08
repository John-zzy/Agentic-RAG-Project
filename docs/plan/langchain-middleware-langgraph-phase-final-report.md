# LangChain Middleware 与 LangGraph 现代化 Final Phase 报告

## 阶段信息

- 阶段：9.x Baseline / Candidate 验证与 10.x 收口
- OpenSpec change：`modernize-langchain-middleware-langgraph`
- Baseline worktree：`D:/Programs/interview-projects/ai-rag-project`
- Candidate worktree：`D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware`
- Baseline commit：`ef1a106b439e7fb508fbf7af1532a90e52de969a`
- Candidate commit：`ef1a106b439e7fb508fbf7af1532a90e52de969a + working tree migration changes`
- 报告日期：2026-06-08
- Go / No-Go：Go，未发现阻塞 `/chat`、`/chat/resume`、SSE、HITL、citation 或 retrieval trace 的 unexpected 差异。

## 范围

- 本阶段完成任务：9.1-9.4、10.1-10.6。
- 本阶段涉及模块：`backend/platform/agent_runtime/`、`backend/platform/workflow/langgraph/`、`backend/platform/rag/orchestration/`、`backend/application/runtime/`、`frontend/knowledge-manager.html`、相关测试与文档。
- 本阶段未覆盖事项：未启动 baseline / candidate 双服务做 HTTP eval replay；本轮以 pytest regression、API/SSE/HITL contract tests、local API signature introspection 和 OpenSpec validate 作为收口证据。

## 运行环境

| Track | Host | Port | Data dir | Session SQLite | LangGraph SQLite | Artifact dir |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | `127.0.0.1` | `8000` | `D:/Programs/interview-projects/ai-rag-project/backend/data` | `backend/data/sessions.db` | `backend/data/langgraph.db` | `backend/tests/artifacts/langchain-middleware-langgraph/baseline` |
| Candidate | `127.0.0.1` | `8010` | `D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware/backend/data` | `backend/data/sessions.db` | `backend/data/langgraph.db` | `backend/tests/artifacts/langchain-middleware-langgraph/candidate` |

## API 版本核对

本地候选环境依赖版本：

- `langchain==1.3.1`
- `langchain-core==1.4.0`
- `langgraph==1.2.0`

本地签名核对：

- `langchain.agents.create_agent(model, tools=None, *, system_prompt=None, middleware=(), response_format=None, state_schema=None, context_schema=None, checkpointer=None, store=None, interrupt_before=None, interrupt_after=None, ...)`
- `AgentMiddleware.wrap_model_call(request, handler) -> ModelResponse | AIMessage | ExtendedModelResponse`
- `AgentMiddleware.wrap_tool_call(request: ToolCallRequest, handler) -> ToolMessage | Command`
- `StateGraph(state_schema, context_schema=None, *, input_schema=None, output_schema=None, ...)`
- `Command(..., resume=...)`
- `interrupt(value)`

官方文档交叉核对：

- LangChain v1 agents 文档以 `create_agent` 作为 agent runtime 入口，并支持 middleware、custom state 和 context schema。
- LangGraph interrupts 文档以 `interrupt()` 暂停、checkpointer/thread id 保存状态、`Command(resume=...)` 恢复执行；文档同时说明恢复会从所在节点开头重新运行，因此项目必须继续保留工具副作用幂等与审批前不执行副作用的安全规则。

## 测试命令

Baseline：

```powershell
D:\Programs\interview-projects\ai-rag-project\backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_generic_assistant_hitl.py backend\tests\test_chat_api.py backend\tests\test_agent_runtime_react.py backend\tests\test_agent_runtime_plan.py backend\tests\test_agent_runtime_tools.py backend\tests\test_agentic_retrieval.py -q -c backend\tests\pytest.ini
```

Candidate：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
backend\.venv\Scripts\python.exe -m compileall -q backend\platform\agent_runtime\react backend\platform\agent_runtime\middleware backend\platform\memory\base\session_store.py
openspec validate modernize-langchain-middleware-langgraph
```

## 测试结果

| Track | Command | Result | Duration | Artifact |
| --- | --- | --- | --- | --- |
| Baseline | Phase 0 核心矩阵 | `failed`，204 passed / 1 failed / 26 warnings | 38.06s | `docs/plan/langchain-middleware-langgraph-phase-0-report.md` |
| Candidate | Full pytest | `480 passed, 3 skipped, 4 deselected` | 80.45s | terminal output |
| Candidate | compileall | passed | <1s | terminal output |
| Candidate | Focused runtime regression | `151 passed` | 6.93s | terminal output |
| Candidate | openspec validate | `Change 'modernize-langchain-middleware-langgraph' is valid` | <3s | terminal output |

## 行为差异

| 领域 | 差异 | 分类 | 处理状态 |
| --- | --- | --- | --- |
| API | `/chat` 与 `/chat/resume` schema 保持稳定；内部 ReAct provider 投影为 provider-neutral `ReActRun`。 | expected | 已由 `test_chat_api.py`、`test_react_provider.py` 覆盖。 |
| SSE | UI SSE 仅保留 `start`、`chunk`、安全 `thinking`、`waiting_user`、`done`、`error`；不再要求 `history` / `tool` 业务事件。 | expected | baseline 已记录旧测试期望差异；candidate 已更新并通过。 |
| HITL | `waiting_user` 使用 LangGraph interrupt 语义恢复，但项目继续先校验最新 checkpoint、`interrupt_id`、terminal state、allowed action 和幂等。 | expected | 已由 LangGraph runtime、generic assistant HITL、Plan/ReAct provider tests 覆盖。 |
| RAG trace | Agentic RAG 改用 typed graph invocation 和 shared model guard helper，保留 final decision、answer mode、retrieval trace。 | expected | 已由 `test_agentic_retrieval.py`、tool adapter contract tests 覆盖。 |
| Citation | ToolObservation、LangChain tool artifact 和 session store citation normalization 保留 citation fields；legacy snippet fallback 恢复 `source_kind/source_name/chunk_id`。 | expected | 已由 full pytest 与 session store regression 覆盖。 |
| Eval replay outputs | 本轮未启动 baseline / candidate HTTP 服务回放。 | unresolved | 不阻塞代码收口；归入后续人工双服务验收。 |

## Legacy 删除与清理证据

- ChatGraph ReAct 分支已固定通过 `ReActRuntime` 和 `ReActProviderFactory` 构建 `create_agent` provider。
- `backend/platform/agent_runtime/react/factory.py` 使用 `create_agent(..., middleware=[model, tool], state_schema=..., context_schema=..., checkpointer=...)`。
- 已移除 legacy provider runtime selection 配置；没有新增 provider、middleware guard、stream version 或 graph output mode 切换。
- 旧 `backend/platform/agent_runtime/react/` 文件仍保留为历史单测和后续删除候选，不再作为 ChatGraph ReAct 主路径。
- `backend/platform/agent_runtime/__init__.py` 已改为轻量合同导出与延迟加载，避免迁移后 RAG / middleware / Plan 之间的循环导入。

## 回归控制状态

- Candidate 分支状态：完成代码迁移、测试补齐、文档同步和最终验证。
- Legacy 删除候选状态：运行时选择路径已清理；旧 ReAct loop 文件列入后续可删除候选。
- 数据库迁移状态：无 schema 破坏性迁移；candidate 使用独立 `backend/data`。
- Checkpoint 兼容状态：继续保存项目 runtime projection，不把 LangChain raw state 作为唯一恢复依据。

## 结论

- 通过条件：核心 API/SSE/HITL/RAG/Plan/ReAct/tool/session/knowledge tests 全量通过，OpenSpec validate 通过后即可归档。
- 阻塞项：无代码阻塞项；HTTP eval replay 双服务对比未执行，保留为人工验收补充项。
- 下一步：归档前可选执行 baseline/candidate HTTP eval replay，并在 `backend/data/evals/langchain-middleware-langgraph/` 写入 replay diff。

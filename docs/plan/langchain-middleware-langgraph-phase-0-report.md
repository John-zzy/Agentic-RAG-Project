# LangChain Middleware 与 LangGraph 现代化迁移 Phase 0 记录

## 1.1 Worktree 创建记录

- 日期：2026-06-08
- OpenSpec change：`modernize-langchain-middleware-langgraph`
- Baseline worktree：`D:/Programs/interview-projects/ai-rag-project`
- Candidate worktree：`D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware`
- Candidate branch：`migrate/langchain-middleware`
- Baseline commit：`ef1a106b439e7fb508fbf7af1532a90e52de969a`
- Candidate commit：`ef1a106b439e7fb508fbf7af1532a90e52de969a`

`git worktree list` 输出：

```text
D:/Programs/interview-projects/ai-rag-project                                         ef1a106 [master]
D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware  ef1a106 [migrate/langchain-middleware]
```

## 1.2 Candidate 隔离路径定义

Candidate 环境隔离约定已记录在 `docs/plan/langchain-middleware-langgraph-candidate-env.md`。

关键路径：

- Candidate runtime data dir：`D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware/backend/data`
- Candidate session SQLite：`D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware/backend/data/sessions.db`
- Candidate LangGraph checkpoint SQLite：`D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware/backend/data/langgraph.db`
- Candidate Chroma persist directory：`D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware/backend/data/.chroma`
- Candidate upload files directory：`D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware/backend/data/files`
- Candidate service port：`8010`
- Candidate eval artifact directory：`D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware/backend/data/evals/langchain-middleware-langgraph/candidate`
- Candidate pytest artifact directory：`D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware/backend/tests/artifacts/langchain-middleware-langgraph/candidate`

## 1.4 阶段报告模板

阶段报告模板已记录在 `docs/plan/langchain-middleware-langgraph-phase-report-template.md`。

模板覆盖 baseline commit、candidate commit、测试命令、baseline 结果、candidate 结果、API / SSE / HITL / RAG trace / citation 差异分类、回滚状态和 go/no-go 结论。

## 1.5 Baseline 测试记录

Baseline 测试在新模式重构实现前运行并记录在本节。

计划命令：

```powershell
D:\Programs\interview-projects\ai-rag-project\backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_generic_assistant_hitl.py backend\tests\test_chat_api.py backend\tests\test_agent_runtime_react.py backend\tests\test_agent_runtime_plan.py backend\tests\test_agent_runtime_tools.py backend\tests\test_agentic_retrieval.py -q -c backend\tests\pytest.ini
```

结果：`failed`

摘要：

- 通过：204
- 失败：1
- 警告：26
- 耗时：38.06s

失败项：

- `backend/tests/test_generic_assistant_hitl.py::test_generic_assistant_sse_emits_waiting_user_for_clarification`

失败摘要：

```text
assert ['start', 'waiting_user'] == ['start', 'history', 'tool', 'waiting_user']
```

说明：该失败发生在新模式重构实现前，记录为 baseline 已存在差异；当前实现实际只发出 `start` 与 `waiting_user` SSE 事件，而测试仍期待旧的 `history` / `tool` 业务事件。

## 1.3 无配置切换约束

本迁移不新增 provider、middleware guard、stream version 或 graph output mode 的运行时配置项。

后续实现直接按 LangChain middleware 与现代 LangGraph 新模式重构；回归控制通过 candidate worktree、阶段报告和 baseline / candidate 对比完成。

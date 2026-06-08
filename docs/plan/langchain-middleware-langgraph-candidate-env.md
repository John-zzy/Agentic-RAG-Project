# LangChain Middleware 与 LangGraph 现代化候选环境约定

本文定义 `modernize-langchain-middleware-langgraph` 迁移期间 candidate worktree 的隔离运行路径。目标是让 baseline 与 candidate 可以并行测试、回放和对比，避免 SQLite、Chroma、上传文件、eval artifact 或服务端口互相污染。

## Worktree

- Baseline worktree：`D:/Programs/interview-projects/ai-rag-project`
- Candidate worktree：`D:/Programs/interview-projects/ai-rag-project/.worktree/migrate-langchain-middleware`
- Candidate branch：`migrate/langchain-middleware`

## Candidate Runtime 路径

以下路径均以 candidate worktree 为根目录解析。

| 用途 | 路径 |
| --- | --- |
| Runtime data dir | `backend/data` |
| 上传文件目录 | `backend/data/files` |
| Session SQLite | `backend/data/sessions.db` |
| LangGraph checkpoint SQLite | `backend/data/langgraph.db` |
| Tool idempotency SQLite | `backend/data/langgraph.db` |
| Chroma persist directory | `backend/data/.chroma` |
| Eval artifact directory | `backend/data/evals` |
| Pytest runtime artifact directory | `backend/tests/artifacts` |

说明：

- `AppSettings.data_dir` 默认解析到 candidate worktree 的 `backend/data`。
- `SQLiteSessionStore` 默认使用 candidate worktree 的 `backend/data/sessions.db`。
- `ChatGraphRuntime.from_settings()` 使用 `AppSettings.data_dir / "langgraph.db"`，因此 checkpoint 与 tool idempotency store 均落在 candidate worktree 的 `backend/data/langgraph.db`。
- Chroma 默认路径为 candidate worktree 的 `backend/data/.chroma`；若显式覆盖，必须继续指向 candidate worktree 内部路径。

## 服务端口

| Track | Host | Port | Base URL |
| --- | --- | --- | --- |
| Baseline | `127.0.0.1` | `8000` | `http://127.0.0.1:8000` |
| Candidate | `127.0.0.1` | `8010` | `http://127.0.0.1:8010` |

Candidate 启动命令：

```powershell
D:\Programs\interview-projects\ai-rag-project\backend\.venv\Scripts\python.exe backend\run.py --host 127.0.0.1 --port 8010
```

如需使用环境变量覆盖端口：

```powershell
$env:AI_RAG_PORT = "8010"
D:\Programs\interview-projects\ai-rag-project\backend\.venv\Scripts\python.exe backend\run.py
```

## 测试与对比产物目录

| 产物类型 | Baseline 路径 | Candidate 路径 | Diff 路径 |
| --- | --- | --- | --- |
| Pytest logs / snapshots | `backend/tests/artifacts/langchain-middleware-langgraph/baseline` | `backend/tests/artifacts/langchain-middleware-langgraph/candidate` | `backend/tests/artifacts/langchain-middleware-langgraph/diff` |
| HTTP eval replay | `backend/data/evals/langchain-middleware-langgraph/baseline` | `backend/data/evals/langchain-middleware-langgraph/candidate` | `backend/data/evals/langchain-middleware-langgraph/diff` |
| Phase reports | `docs/plan/langchain-middleware-langgraph-phase-*.md` | `docs/plan/langchain-middleware-langgraph-phase-*.md` | `docs/plan/langchain-middleware-langgraph-phase-*.md` |

Baseline 与 candidate 各自在自己的 worktree 中写入同名相对路径；diff 产物优先在 candidate worktree 中生成，便于随迁移分支审阅。

## Eval Replay 示例

Baseline：

```powershell
D:\Programs\interview-projects\ai-rag-project\backend\.venv\Scripts\python.exe backend\evals\run_http_eval.py --base-url http://127.0.0.1:8000 --sample-set minimal --output backend\data\evals\langchain-middleware-langgraph\baseline\minimal.json
```

Candidate：

```powershell
D:\Programs\interview-projects\ai-rag-project\backend\.venv\Scripts\python.exe backend\evals\run_http_eval.py --base-url http://127.0.0.1:8010 --sample-set minimal --output backend\data\evals\langchain-middleware-langgraph\candidate\minimal.json --compare-to backend\data\evals\langchain-middleware-langgraph\baseline\minimal.json
```

## 运行约束

- Candidate 服务、测试和 eval 必须从 candidate worktree 目录执行。
- Candidate 不复用 baseline 的 `backend/data`、`backend/tests/artifacts` 或服务端口。
- 若后续显式设置 `AI_RAG_DATA_DIR` 或 `AI_RAG_VECTOR_STORE__CHROMA__PERSIST_DIRECTORY`，路径必须解析到 candidate worktree 内部。
- 本迁移不新增 provider、middleware guard、stream version 或 graph output mode 的运行时配置切换；本文件只定义迁移期间的运行环境隔离约定。

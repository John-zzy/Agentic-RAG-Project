# Repository Guidelines

## Project Structure & Module Organization
`backend/` contains the application code. Key modules are `api/` for FastAPI routes and chat orchestration, `knowledge/` for retrieval and vector-store integrations, `memory/` for session state, `models/` for model routing, and `config/` for settings. Tests live in `backend/tests/`. `frontend/` currently contains a single static API tester page, `api-tester.html`, which is mounted at `/frontend`. `docs/elasticsearch/` holds the local Elasticsearch compose setup, and `openspec/` tracks change proposals and implementation tasks.

## Build, Test, and Development Commands
Create a virtual environment, then install dependencies from the backend:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

Run the API locally with bootstrap loading:

```powershell
python backend\run.py
```

Run tests:

```powershell
python -m pytest backend\tests -q -c backend\tests\pytest.ini
```

Start the optional Elasticsearch stack:

```powershell
docker compose -f docs\elasticsearch\docker-compose.yml up -d
```

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints, `snake_case` for functions and modules, `PascalCase` for classes, and small, focused helpers. Keep imports explicit and prefer `Path` over hard-coded file strings. No formatter or linter is checked in yet, so match nearby code before introducing new patterns.

## Testing Guidelines
Use `pytest` and place tests under `backend/tests/` as `test_*.py`. Mirror the unit under test when naming files, for example `test_chat_api.py` or `test_session_store.py`. Mark external-integration cases with `@pytest.mark.integration`; the default test config excludes them.

## Commit & Pull Request Guidelines
Recent history uses short, date-prefixed snapshot commits such as `2026.04.23 ai rag project`. Keep commits focused and descriptive; prefer one logical change per commit. PRs should include a concise summary, note any config or API changes, link the related OpenSpec item or issue, and add request/response examples or screenshots when UI or API behavior changes.

## Security & Configuration Tips
Store secrets in `backend/.env` and never commit API keys. Default development uses `chroma`; switch to Elasticsearch with `AI_RAG_VECTOR_STORE__PROVIDER=elasticsearch` and the matching `AI_RAG_VECTOR_STORE__ELASTICSEARCH__URL`.

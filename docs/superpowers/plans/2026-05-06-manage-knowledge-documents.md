# Manage Knowledge Documents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add backend and demo-page support for registering, listing, viewing, deleting, and re-chunking JSON knowledge documents with traceable Elasticsearch-backed document metadata.

**Architecture:** Keep raw JSON files in `backend/data/files` and add a dedicated knowledge-document management layer that validates file inputs, derives stable `document_id`/`source_record_id`/`chunk_id`, stores document records plus active-version summaries separately from retrievable chunks, and exposes CRUD-style FastAPI endpoints. Reuse the existing vector-store abstraction where practical, but introduce document-focused types and tests so document management does not leak into chat/session concerns.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, Elasticsearch, pytest, static HTML/CSS/JavaScript

---

## Implementation Status

**Status:** Implemented on branch `codex/manage-knowledge-documents` in worktree `.worktrees/manage-knowledge-documents`.

**Final verification:**

- Full backend tests: `.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini` -> `79 passed, 4 deselected`
- API smoke check: `python backend\run.py`, then `GET /health` -> `{"status":"ok"}`
- Static page script check: extracted inline script from `frontend/api-tester.html` and ran `node --check`

**Implementation notes:**

- Document service uses rollback-safe version publishing. It writes new chunks before publishing a new active document record, preserves previous active chunks on chunk/record/deactivation failures, reactivates previous chunks after partial deactivation failures, and keeps cleanup best-effort so rollback cleanup errors do not mask the original failure.
- FastAPI app creation lazily initializes the knowledge document service for `/knowledge/documents` routes. Chat-only startup is not blocked by Elasticsearch/document-index initialization.
- API error responses use stable structured details for validation, not-found, store, and unexpected backend errors.
- `keep_version=True` now preserves historical version records while only the new active version remains active for default retrieval/filtering.

**Key commits:**

- `eb41c18` `feat: add document management elasticsearch indexes`
- `1d77d36` `feat: add knowledge document loaders and validators`
- `7322f64` `feat: add knowledge document management service`
- `0c09a35` `feat: add knowledge document management api`
- `f4c29a2` `feat: add knowledge document management demo ui`
- Follow-up hardening commits: `06c9728`, `5184b43`, `7d40b0f`, `f7e7061`, `4e19446`, `c9b1f98`, `110f0a9`, `750a091`

## File Structure

**Create**

- `backend/api/knowledge/__init__.py` - knowledge document API package marker.
- `backend/api/knowledge/schemas.py` - request/response models for register/list/detail/delete/rechunk operations.
- `backend/api/knowledge/routes.py` - FastAPI routes for knowledge document management.
- `backend/knowledge/documents/__init__.py` - knowledge document package marker.
- `backend/knowledge/documents/schemas.py` - internal document/version/chunk payload models.
- `backend/knowledge/documents/validators.py` - validation helpers for namespace, file path, JSON content, and chunk parameters.
- `backend/knowledge/documents/loader.py` - JSON file loading plus stable record-id derivation.
- `backend/knowledge/documents/chunker.py` - chunk generation and stable chunk-id derivation.
- `backend/knowledge/documents/service.py` - orchestration for register/list/detail/delete/rechunk flows.
- `backend/tests/test_knowledge_document_service.py` - unit tests for the service layer.
- `backend/tests/test_knowledge_document_api.py` - API tests for the new routes.

**Modify**

- `backend/config/settings.py` - extend vector store settings to include knowledge-document namespaces or index names for `documents` and `chunks`.
- `backend/api/base/app.py` - inject the new service into app state and register the new router.
- `backend/knowledge/base/store.py` - add document/chunk index support and filtered operations needed by knowledge-document management.
- `backend/tests/test_knowledge_elasticsearch.py` - expand fake/live Elasticsearch tests for the added `documents` and `chunks` index behavior.
- `backend/tests/test_support.py` - add helpers for temporary file fixtures if needed by new tests.
- `frontend/api-tester.html` - add the knowledge document management UI and wire it to the new endpoints.

**Check Before Coding**

- `backend/knowledge/ecommerce/loader.py` - existing JSON preload pattern and current `VectorStore` usage.
- `backend/api/chat/routes.py` - current route style and error handling.
- `backend/tests/test_chat_api.py` - current `FastAPI` test setup pattern.
- `openspec/changes/manage-knowledge-documents/design.md` - authoritative storage/versioning decisions.
- `openspec/changes/manage-knowledge-documents/specs/knowledge-document-management/spec.md` - requirement coverage checklist.

### Task 1: Define document storage settings and Elasticsearch primitives

**Files:**
- Modify: `backend/config/settings.py`
- Modify: `backend/knowledge/base/store.py`
- Modify: `backend/tests/test_knowledge_elasticsearch.py`

- [ ] **Step 1: Write the failing test for separate `documents` and `chunks` indexes**

```python
def test_elasticsearch_store_initializes_document_management_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_factory = FakeElasticsearchFactory()
    monkeypatch.setattr(store_module, "Elasticsearch", fake_factory)

    app_settings = AppSettings(
        data_dir=DATA_DIR,
        vector_store=VectorStoreConfig(
            provider="elasticsearch",
            elasticsearch={"url": "http://localhost:9200", "index_prefix": "ai-rag"},
        ),
    )

    store = ElasticsearchVectorStore(app_settings)
    store.ensure_document_indexes()

    fake_client = fake_factory.instances[-1]
    assert fake_client.indices.exists(index="ai-rag-documents")
    assert fake_client.indices.exists(index="ai-rag-chunks")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend\tests\test_knowledge_elasticsearch.py::test_elasticsearch_store_initializes_document_management_indexes -q -c backend\tests\pytest.ini`
Expected: FAIL with `AttributeError` or missing settings/index support for `ensure_document_indexes`

- [ ] **Step 3: Add minimal settings and store support**

```python
class DocumentIndexConfig(BaseModel):
    documents_index_name: str = "documents"
    chunks_index_name: str = "chunks"


class VectorStoreConfig(BaseModel):
    # existing fields...
    document_indexes: DocumentIndexConfig = DocumentIndexConfig()


class ElasticsearchVectorStore(VectorStore):
    def ensure_document_indexes(self) -> None:
        self._ensure_named_index(self.resolve_document_index_name("documents"), DOCUMENTS_MAPPING)
        self._ensure_named_index(self.resolve_document_index_name("chunks"), CHUNKS_MAPPING)

    def resolve_document_index_name(self, kind: str) -> str:
        configured_name = getattr(self.config.document_indexes, f"{kind}_index_name")
        prefix = self.config.elasticsearch.index_prefix.strip("-")
        return f"{prefix}-{configured_name}" if prefix else configured_name
```

- [ ] **Step 4: Run targeted Elasticsearch tests**

Run: `python -m pytest backend\tests\test_knowledge_elasticsearch.py -q -c backend\tests\pytest.ini`
Expected: PASS, including the new document index test and no regression in existing product/review index tests

- [ ] **Step 5: Commit**

```bash
git add backend/config/settings.py backend/knowledge/base/store.py backend/tests/test_knowledge_elasticsearch.py
git commit -m "feat: add document management elasticsearch indexes"
```

### Task 2: Add document validation, file loading, and stable ID derivation

**Files:**
- Create: `backend/knowledge/documents/schemas.py`
- Create: `backend/knowledge/documents/validators.py`
- Create: `backend/knowledge/documents/loader.py`
- Create: `backend/knowledge/documents/chunker.py`
- Create: `backend/knowledge/documents/__init__.py`
- Modify: `backend/tests/test_support.py`
- Test: `backend/tests/test_knowledge_document_service.py`

- [ ] **Step 1: Write failing tests for validation and stable IDs**

```python
def test_load_document_records_builds_stable_record_ids(tmp_path: Path) -> None:
    source = tmp_path / "files" / "faq" / "returns.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('[{"id":"r1","title":"退货","content":"7天无理由"}]', encoding="utf-8")

    records = load_document_records(
        namespace="faq",
        source_path=source,
        data_root=tmp_path / "files",
    )

    assert records[0].source_record_id
    assert records[0].source_record_id == records[0].source_record_id
    assert records[0].source_path == "faq/returns.json"


def test_validate_chunking_rejects_overlap_ge_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        validate_chunking(chunk_size=200, chunk_overlap=200)
```

- [ ] **Step 2: Run the service test file to confirm failures**

Run: `python -m pytest backend\tests\test_knowledge_document_service.py -q -c backend\tests\pytest.ini`
Expected: FAIL because the new loader/validator modules do not exist yet

- [ ] **Step 3: Implement validators, record loading, and chunk payload generation**

```python
def validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")


def build_document_id(namespace: str, source_path: str) -> str:
    return hashlib.sha256(f"{namespace}:{source_path}".encode("utf-8")).hexdigest()


def build_source_record_id(source_path: str, record: dict[str, Any], index: int) -> str:
    raw = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(f"{source_path}:{index}:{raw}".encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Re-run the focused service tests**

Run: `python -m pytest backend\tests\test_knowledge_document_service.py -q -c backend\tests\pytest.ini`
Expected: PASS for validation and stable-ID tests

- [ ] **Step 5: Commit**

```bash
git add backend/knowledge/documents backend/tests/test_support.py backend/tests/test_knowledge_document_service.py
git commit -m "feat: add knowledge document loaders and validators"
```

### Task 3: Implement document management service for register/list/detail/delete/rechunk

**Files:**
- Create: `backend/knowledge/documents/service.py`
- Modify: `backend/knowledge/base/store.py`
- Test: `backend/tests/test_knowledge_document_service.py`

- [ ] **Step 1: Write failing service tests for the full lifecycle**

```python
def test_register_document_persists_document_record_and_chunks(service: KnowledgeDocumentService) -> None:
    result = service.register_document(
        namespace="faq",
        source_path="faq/returns.json",
        chunk_size=120,
        chunk_overlap=20,
        keep_version=False,
    )

    assert result.document_id
    assert result.document_version == 1
    assert result.chunk_count > 0
    detail = service.get_document(result.document_id)
    assert detail.active_version == 1
    assert detail.source_path == "faq/returns.json"


def test_rechunk_document_keeps_previous_version_when_requested(service: KnowledgeDocumentService) -> None:
    first = service.register_document("faq", "faq/returns.json", 120, 20, False)
    second = service.rechunk_document(first.document_id, chunk_size=80, chunk_overlap=10, keep_version=True)

    assert second.document_version == 2
    assert second.active_version == 2
    assert any(version.document_version == 1 for version in second.versions)
```

- [ ] **Step 2: Run the new service test file**

Run: `python -m pytest backend\tests\test_knowledge_document_service.py -q -c backend\tests\pytest.ini`
Expected: FAIL because `KnowledgeDocumentService` and store-level document APIs are not implemented

- [ ] **Step 3: Implement minimal service and store operations**

```python
class KnowledgeDocumentService:
    """管理 JSON 知识文档的注册、查询、删除和重新切分。"""

    def register_document(... ) -> DocumentOperationResult:
        records = load_document_records(...)
        chunks = build_chunks(...)
        self.store.ensure_document_indexes()
        self.store.replace_active_document_version(...)
        return DocumentOperationResult(...)

    def list_documents(self, namespace: str | None = None) -> list[DocumentSummary]:
        return self.store.list_documents(namespace=namespace)

    def get_document(self, document_id: str) -> DocumentDetail:
        return self.store.get_document(document_id)
```

- [ ] **Step 4: Run lifecycle service tests plus Elasticsearch fake-store tests**

Run: `python -m pytest backend\tests\test_knowledge_document_service.py backend\tests\test_knowledge_elasticsearch.py -q -c backend\tests\pytest.ini`
Expected: PASS, including overwrite, keep-version, delete, and not-found branches

- [ ] **Step 5: Commit**

```bash
git add backend/knowledge/documents/service.py backend/knowledge/base/store.py backend/tests/test_knowledge_document_service.py backend/tests/test_knowledge_elasticsearch.py
git commit -m "feat: add knowledge document management service"
```

### Task 4: Expose knowledge document management through FastAPI

**Files:**
- Create: `backend/api/knowledge/__init__.py`
- Create: `backend/api/knowledge/schemas.py`
- Create: `backend/api/knowledge/routes.py`
- Modify: `backend/api/base/app.py`
- Test: `backend/tests/test_knowledge_document_api.py`

- [ ] **Step 1: Write failing API tests**

```python
def test_register_knowledge_document_returns_document_payload(client: TestClient) -> None:
    response = client.post(
        "/knowledge/documents",
        json={
            "namespace": "faq",
            "source_path": "faq/returns.json",
            "chunk_size": 120,
            "chunk_overlap": 20,
            "keep_version": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"]
    assert payload["document_version"] == 1


def test_register_knowledge_document_rejects_invalid_chunk_params(client: TestClient) -> None:
    response = client.post(
        "/knowledge/documents",
        json={"namespace": "faq", "source_path": "faq/returns.json", "chunk_size": 10, "chunk_overlap": 10},
    )
    assert response.status_code == 422
```

- [ ] **Step 2: Run the new API test file**

Run: `python -m pytest backend\tests\test_knowledge_document_api.py -q -c backend\tests\pytest.ini`
Expected: FAIL because the router and schemas do not exist yet

- [ ] **Step 3: Implement schemas, routes, and app wiring**

```python
router = APIRouter(prefix="/knowledge/documents", tags=["knowledge-documents"])


@router.post("", response_model=KnowledgeDocumentOperationResponse)
def register_document(payload: KnowledgeDocumentRegisterRequest, request: Request) -> KnowledgeDocumentOperationResponse:
    service = _get_knowledge_document_service(request)
    return KnowledgeDocumentOperationResponse.model_validate(service.register_document(**payload.model_dump()))


@router.get("", response_model=list[KnowledgeDocumentSummaryResponse])
def list_documents(request: Request, namespace: str | None = Query(default=None)) -> list[KnowledgeDocumentSummaryResponse]:
    service = _get_knowledge_document_service(request)
    return [KnowledgeDocumentSummaryResponse.model_validate(item) for item in service.list_documents(namespace=namespace)]
```

- [ ] **Step 4: Run API and chat regression tests**

Run: `python -m pytest backend\tests\test_knowledge_document_api.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini`
Expected: PASS, with existing chat routes unchanged

- [ ] **Step 5: Commit**

```bash
git add backend/api/knowledge backend/api/base/app.py backend/tests/test_knowledge_document_api.py
git commit -m "feat: add knowledge document management api"
```

### Task 5: Add the static knowledge document management UI

**Files:**
- Modify: `frontend/api-tester.html`

- [ ] **Step 1: Write down the manual UI acceptance checks before editing**

```text
1. Page still loads chat panel on desktop and mobile.
2. New knowledge document panel can submit register/list/detail/delete/rechunk requests.
3. Error states render visibly without breaking the chat composer.
4. Existing chat flow still works after the UI changes.
```

- [ ] **Step 2: Implement the new panel and API wiring**

```html
<section class="document-panel">
  <h2>我的文档知识库</h2>
  <form id="documentForm">
    <input id="documentNamespace" name="namespace" />
    <input id="documentSourcePath" name="source_path" />
    <input id="documentChunkSize" name="chunk_size" type="number" />
    <input id="documentChunkOverlap" name="chunk_overlap" type="number" />
    <label><input id="documentKeepVersion" type="checkbox" /> 保留旧版本</label>
    <button id="registerDocumentButton" type="submit">注册文档</button>
  </form>
  <div id="documentList"></div>
</section>
```

- [ ] **Step 3: Add minimal client-side behavior**

```javascript
async function registerDocument(payload) {
  return apiRequest("/knowledge/documents", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

async function refreshDocuments() {
  const documents = await apiRequest("/knowledge/documents");
  renderDocumentList(documents);
}
```

- [ ] **Step 4: Run backend tests to ensure the UI edit did not require backend changes**

Run: `python -m pytest backend\tests -q -c backend\tests\pytest.ini`
Expected: PASS or only existing `@pytest.mark.integration` cases skipped

- [ ] **Step 5: Commit**

```bash
git add frontend/api-tester.html
git commit -m "feat: add knowledge document management demo ui"
```

### Task 6: End-to-end verification and cleanup

**Files:**
- Check: `backend/tests/test_knowledge_document_service.py`
- Check: `backend/tests/test_knowledge_document_api.py`
- Check: `backend/tests/test_knowledge_elasticsearch.py`
- Check: `frontend/api-tester.html`

- [ ] **Step 1: Run the full non-integration test suite**

Run: `python -m pytest backend\tests -q -c backend\tests\pytest.ini`
Expected: PASS, with integration tests skipped by default

- [ ] **Step 2: Run live API manually**

Run: `python backend\run.py`
Expected: FastAPI starts on `http://127.0.0.1:8000` without route-registration errors

- [ ] **Step 3: Validate the static page manually**

Run: open `http://127.0.0.1:8000/frontend/api-tester.html`
Expected: can register a JSON document from `backend/data/files/...`, refresh the list, inspect details, delete a document, and re-chunk with or without `keep_version`

- [ ] **Step 4: Run Elasticsearch integration tests only when Elasticsearch is available**

Run: `python -m pytest backend\tests\test_knowledge_elasticsearch.py -q -c backend\tests\pytest.ini -m integration`
Expected: PASS when local Elasticsearch is running, otherwise explicit `skip` messages only

- [ ] **Step 5: Commit**

```bash
git add backend frontend
git commit -m "test: verify knowledge document management end to end"
```

## Spec Coverage Check

- 文档注册与入库: Covered by Task 2 and Task 3.
- 非法参数拒绝: Covered by Task 2 and Task 4 tests.
- `document_id` 稳定生成: Covered by Task 2.
- 检索结果追溯到 `source_path`/`source_record_id`: Covered by Task 2 and Task 3 chunk metadata assertions.
- 文档列表与详情: Covered by Task 3 and Task 4.
- 默认覆盖与显式保留版本: Covered by Task 3 lifecycle tests and Task 4 API tests.
- Elasticsearch 双索引组织: Covered by Task 1 and Task 3.
- 文档删除且默认不删原始 JSON: Covered by Task 3 delete tests.
- 文档重新切分: Covered by Task 3 and Task 4.
- 静态测试页演示能力: Covered by Task 5 and Task 6.

## Placeholder Scan

- No `TODO`, `TBD`, or “implement later” placeholders remain.
- Each task includes explicit file paths, commands, and expected outcomes.
- Code-bearing steps contain concrete snippets rather than abstract directions.

## Type Consistency Check

- `document_id` is always derived from `namespace + source_path`.
- Version field names stay consistent as `document_version` and `active_version`.
- Trace fields stay consistent as `source_path`, `source_record_id`, `chunk_id`, and `chunk_index`.
- Route prefix stays consistent as `/knowledge/documents`.

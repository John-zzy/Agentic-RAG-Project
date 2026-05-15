from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from backend.application.runtime.api.app import create_app
from backend.platform.knowledge.documents import (
    KnowledgeDocumentDetail,
    KnowledgeDocumentNotFoundError,
    KnowledgeDocumentOperationResult,
    KnowledgeDocumentStoreError,
    KnowledgeDocumentSummary,
    KnowledgeDocumentVersionSummary,
    KnowledgeFileIndexSummary,
)


class FakeKnowledgeDocumentState:
    def __init__(self) -> None:
        self.documents: dict[str, KnowledgeDocumentOperationResult] = {}
        self.file_indexes: list[KnowledgeFileIndexSummary] = []
        self.last_register: dict[str, Any] | None = None
        self.last_rechunk: dict[str, Any] | None = None
        self.fail_store = False
        self.fail_unknown_methods: set[str] = set()
        self.last_list_files_namespace: str | None = None


class FakeKnowledgeDocumentApplicationService:
    """只模拟写路径，测试路由是否正确拿 application service。"""

    def __init__(self, state: FakeKnowledgeDocumentState) -> None:
        self.state = state

    def register_document(
        self,
        namespace: str,
        source_path: str,
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool = False,
    ) -> KnowledgeDocumentOperationResult:
        self.state.last_register = {
            "namespace": namespace,
            "source_path": source_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "keep_version": keep_version,
        }
        if self.state.fail_store:
            raise KnowledgeDocumentStoreError("store unavailable")
        if "register" in self.state.fail_unknown_methods:
            raise RuntimeError("secret backend register failure")
        document_id = f"{namespace}:returns"
        current = self.state.documents.get(document_id)
        document_version = 1 if current is None else current.document_version + 1
        versions = [] if not keep_version else list(current.versions if current else [])
        versions.append(_version(document_version, chunk_size, chunk_overlap))
        result = _operation_result(
            document_id=document_id,
            namespace=namespace,
            source_path=source_path,
            document_version=document_version,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            versions=versions,
        )
        self.state.documents[document_id] = result
        return result

    def delete_document(self, document_id: str) -> KnowledgeDocumentOperationResult:
        if "delete" in self.state.fail_unknown_methods:
            raise RuntimeError("secret backend delete failure")
        document = self.state.documents.pop(document_id, None)
        if document is None:
            raise KnowledgeDocumentNotFoundError(document_id)
        return document.model_copy(update={"status": "deleted"})

    def rechunk_document(
        self,
        document_id: str,
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool = False,
    ) -> KnowledgeDocumentOperationResult:
        self.state.last_rechunk = {
            "document_id": document_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "keep_version": keep_version,
        }
        if "rechunk" in self.state.fail_unknown_methods:
            raise RuntimeError("secret backend rechunk failure")
        current = self.state.documents.get(document_id)
        if current is None:
            raise KnowledgeDocumentNotFoundError(document_id)
        document_version = current.document_version + 1
        versions = [] if not keep_version else list(current.versions)
        versions.append(_version(document_version, chunk_size, chunk_overlap))
        result = current.model_copy(
            update={
                "document_version": document_version,
                "active_version": document_version,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "versions": versions,
            }
        )
        self.state.documents[document_id] = result
        return result


class FakeKnowledgeDocumentQueryService:
    """只模拟读路径，测试路由是否正确拿 query service。"""

    def __init__(self, state: FakeKnowledgeDocumentState) -> None:
        self.state = state

    def list_documents(self, namespace: str | None = None) -> list[KnowledgeDocumentSummary]:
        if "list" in self.state.fail_unknown_methods:
            raise RuntimeError("secret backend list failure")
        documents = list(self.state.documents.values())
        if namespace is not None:
            documents = [document for document in documents if document.namespace == namespace]
        return [
            KnowledgeDocumentSummary(
                document_id=document.document_id,
                namespace=document.namespace,
                source_path=document.source_path,
                status=document.status,
                active_version=document.active_version,
                chunk_count=document.chunk_count,
                updated_at=document.updated_at,
            )
            for document in documents
        ]

    def list_file_indexes(self, namespace: str | None = None) -> list[KnowledgeFileIndexSummary]:
        self.state.last_list_files_namespace = namespace
        if "files" in self.state.fail_unknown_methods:
            raise RuntimeError("secret backend files failure")
        if self.state.fail_store:
            raise KnowledgeDocumentStoreError("store unavailable")
        if namespace is None:
            return list(self.state.file_indexes)
        return [item for item in self.state.file_indexes if item.namespace == namespace]

    def get_document(self, document_id: str) -> KnowledgeDocumentDetail:
        if "detail" in self.state.fail_unknown_methods:
            raise RuntimeError("secret backend detail failure")
        document = self.state.documents.get(document_id)
        if document is None:
            raise KnowledgeDocumentNotFoundError(document_id)
        return KnowledgeDocumentDetail(**document.model_dump(exclude={"document_version"}))


def _version(
    document_version: int,
    chunk_size: int,
    chunk_overlap: int,
) -> KnowledgeDocumentVersionSummary:
    return KnowledgeDocumentVersionSummary(
        document_version=document_version,
        status="active",
        chunk_count=1,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        created_at="2026-05-07T00:00:00+00:00",
    )


def _operation_result(
    *,
    document_id: str,
    namespace: str,
    source_path: str,
    document_version: int,
    chunk_size: int,
    chunk_overlap: int,
    versions: list[KnowledgeDocumentVersionSummary],
) -> KnowledgeDocumentOperationResult:
    return KnowledgeDocumentOperationResult(
        document_id=document_id,
        namespace=namespace,
        source_path=source_path,
        status="active",
        active_version=document_version,
        chunk_count=1,
        updated_at="2026-05-07T00:00:00+00:00",
        source_type="json",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        last_error=None,
        versions=versions,
        document_version=document_version,
    )


def _services() -> tuple[
    FakeKnowledgeDocumentState,
    FakeKnowledgeDocumentApplicationService,
    FakeKnowledgeDocumentQueryService,
]:
    # 读写 fake service 共用同一份状态，这样既能验证拆分注入，也不会改变测试数据流。
    state = FakeKnowledgeDocumentState()
    return (
        state,
        FakeKnowledgeDocumentApplicationService(state),
        FakeKnowledgeDocumentQueryService(state),
    )


def _client(
    application_service: FakeKnowledgeDocumentApplicationService,
    query_service: FakeKnowledgeDocumentQueryService,
) -> TestClient:
    return TestClient(
        create_app(
            knowledge_document_application_service=application_service,
            knowledge_document_query_service=query_service,
        )
    )


class FailingChatOnlyService:
    def chat(self, payload: object) -> object:
        raise AssertionError("chat should not be called")


def test_create_app_with_chat_service_does_not_create_document_service() -> None:
    app = create_app(chat_service=FailingChatOnlyService())  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert not hasattr(app.state, "knowledge_document_application_service")
    assert not hasattr(app.state, "knowledge_document_query_service")


def test_register_knowledge_document_returns_document_payload() -> None:
    state, application_service, query_service = _services()

    with _client(application_service, query_service) as client:
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
    assert state.last_register == {
        "namespace": "faq",
        "source_path": "faq/returns.json",
        "chunk_size": 120,
        "chunk_overlap": 20,
        "keep_version": False,
    }


def test_register_knowledge_document_rejects_invalid_chunk_params() -> None:
    _, application_service, query_service = _services()

    with _client(application_service, query_service) as client:
        response = client.post(
            "/knowledge/documents",
            json={
                "namespace": "faq",
                "source_path": "faq/returns.json",
                "chunk_size": 10,
                "chunk_overlap": 10,
            },
        )

    assert response.status_code == 422


def test_list_knowledge_documents_filters_namespace() -> None:
    _, application_service, query_service = _services()
    application_service.register_document("faq", "faq/returns.json", 120, 20, False)
    application_service.register_document("products", "products/laptop.json", 120, 20, False)

    with _client(application_service, query_service) as client:
        response = client.get("/knowledge/documents", params={"namespace": "faq"})

    assert response.status_code == 200
    payload = response.json()
    assert [document["namespace"] for document in payload["documents"]] == ["faq"]


def test_list_knowledge_files_returns_index_status() -> None:
    state, application_service, query_service = _services()
    state.file_indexes = [
        KnowledgeFileIndexSummary(
            filename="returns.json",
            source_path="faq/returns.json",
            namespace="faq",
            document_id="faq:returns",
            indexed=True,
            status="active",
            active_version=1,
            chunk_count=2,
            updated_at="2026-05-07T00:00:00+00:00",
            can_index=True,
        ),
        KnowledgeFileIndexSummary(
            filename="laptop.json",
            source_path="products/laptop.json",
            namespace="products",
            document_id="products:laptop",
            indexed=True,
            status="active",
            active_version=1,
            chunk_count=1,
            updated_at="2026-05-07T00:00:00+00:00",
            can_index=True,
        ),
    ]

    with _client(application_service, query_service) as client:
        response = client.get("/knowledge/documents/files", params={"namespace": "faq"})

    assert response.status_code == 200
    payload = response.json()
    assert [item["namespace"] for item in payload["items"]] == ["faq"]
    assert state.last_list_files_namespace == "faq"


def test_get_knowledge_document_returns_detail() -> None:
    _, application_service, query_service = _services()
    created = application_service.register_document("faq", "faq/returns.json", 120, 20, False)

    with _client(application_service, query_service) as client:
        response = client.get(f"/knowledge/documents/{created.document_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == created.document_id
    assert payload["versions"][0]["chunk_size"] == 120
    assert payload["last_error"] is None


def test_get_knowledge_document_missing_returns_404() -> None:
    _, application_service, query_service = _services()

    with _client(application_service, query_service) as client:
        response = client.get("/knowledge/documents/missing")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "KNOWLEDGE_DOCUMENT_NOT_FOUND"


def test_delete_knowledge_document_returns_deleted_payload() -> None:
    state, application_service, query_service = _services()
    created = application_service.register_document("faq", "faq/returns.json", 120, 20, False)

    with _client(application_service, query_service) as client:
        response = client.delete(f"/knowledge/documents/{created.document_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    assert query_service.list_documents() == []
    assert state.documents == {}


def test_rechunk_knowledge_document_overwrites_by_default() -> None:
    state, application_service, query_service = _services()
    created = application_service.register_document("faq", "faq/returns.json", 120, 20, False)

    with _client(application_service, query_service) as client:
        response = client.post(
            f"/knowledge/documents/{created.document_id}/rechunk",
            json={"chunk_size": 80, "chunk_overlap": 10},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_version"] == 2
    assert [version["document_version"] for version in payload["versions"]] == [2]
    assert state.last_rechunk == {
        "document_id": created.document_id,
        "chunk_size": 80,
        "chunk_overlap": 10,
        "keep_version": False,
    }


def test_rechunk_knowledge_document_can_keep_previous_version() -> None:
    _, application_service, query_service = _services()
    created = application_service.register_document("faq", "faq/returns.json", 120, 20, False)

    with _client(application_service, query_service) as client:
        response = client.post(
            f"/knowledge/documents/{created.document_id}/rechunk",
            json={"chunk_size": 80, "chunk_overlap": 10, "keep_version": True},
        )

    assert response.status_code == 200
    payload = response.json()
    assert [version["document_version"] for version in payload["versions"]] == [1, 2]


def test_store_errors_return_500_with_structured_detail() -> None:
    state, application_service, query_service = _services()
    state.fail_store = True

    with _client(application_service, query_service) as client:
        response = client.post(
            "/knowledge/documents",
            json={
                "namespace": "faq",
                "source_path": "faq/returns.json",
                "chunk_size": 120,
                "chunk_overlap": 20,
            },
        )

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "KNOWLEDGE_DOCUMENT_STORE_ERROR",
        "message": "Knowledge document backend is unavailable.",
    }


def test_list_knowledge_files_store_error_returns_structured_detail() -> None:
    state, application_service, query_service = _services()
    state.fail_store = True

    with _client(application_service, query_service) as client:
        response = client.get("/knowledge/documents/files")

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "KNOWLEDGE_DOCUMENT_STORE_ERROR",
        "message": "Knowledge document backend is unavailable.",
    }


def test_unknown_register_error_returns_structured_safe_500() -> None:
    state, application_service, query_service = _services()
    state.fail_unknown_methods.add("register")

    with _client(application_service, query_service) as client:
        response = client.post(
            "/knowledge/documents",
            json={
                "namespace": "faq",
                "source_path": "faq/returns.json",
                "chunk_size": 120,
                "chunk_overlap": 20,
            },
        )

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "KNOWLEDGE_DOCUMENT_INTERNAL_ERROR",
        "message": "Knowledge document operation failed.",
    }


def test_unknown_errors_return_structured_safe_500_for_read_write_routes() -> None:
    cases = [
        ("list", "get", "/knowledge/documents", None),
        ("files", "get", "/knowledge/documents/files", None),
        ("detail", "get", "/knowledge/documents/faq:returns", None),
        ("delete", "delete", "/knowledge/documents/faq:returns", None),
        (
            "rechunk",
            "post",
            "/knowledge/documents/faq:returns/rechunk",
            {"chunk_size": 80, "chunk_overlap": 10},
        ),
    ]

    for method_name, http_method, url, json_body in cases:
        state, application_service, query_service = _services()
        application_service.register_document("faq", "faq/returns.json", 120, 20, False)
        state.fail_unknown_methods.add(method_name)
        with _client(application_service, query_service) as client:
            response = client.request(http_method, url, json=json_body)

        assert response.status_code == 500
        assert response.json()["detail"] == {
            "code": "KNOWLEDGE_DOCUMENT_INTERNAL_ERROR",
            "message": "Knowledge document operation failed.",
        }

def test_create_app_accepts_split_document_services() -> None:
    _, application_service, query_service = _services()
    application_service.register_document("faq", "faq/returns.json", 120, 20, False)

    with _client(application_service, query_service) as client:
        response = client.get("/knowledge/documents")

    assert response.status_code == 200
    assert response.json()["documents"][0]["namespace"] == "faq"

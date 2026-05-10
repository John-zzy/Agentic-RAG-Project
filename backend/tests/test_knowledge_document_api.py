from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from backend.application.runtime.api.app import create_app
from backend.platform.knowledge.documents.service import (
    KnowledgeDocumentDetail,
    KnowledgeDocumentNotFoundError,
    KnowledgeDocumentOperationResult,
    KnowledgeDocumentStoreError,
    KnowledgeDocumentSummary,
    KnowledgeDocumentVersionSummary,
)


class FakeKnowledgeDocumentService:
    def __init__(self) -> None:
        self.documents: dict[str, KnowledgeDocumentOperationResult] = {}
        self.last_register: dict[str, Any] | None = None
        self.last_rechunk: dict[str, Any] | None = None
        self.fail_store = False
        self.fail_unknown_methods: set[str] = set()

    def register_document(
        self,
        namespace: str,
        source_path: str,
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool = False,
    ) -> KnowledgeDocumentOperationResult:
        self.last_register = {
            "namespace": namespace,
            "source_path": source_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "keep_version": keep_version,
        }
        if self.fail_store:
            raise KnowledgeDocumentStoreError("store unavailable")
        if "register" in self.fail_unknown_methods:
            raise RuntimeError("secret backend register failure")
        document_id = f"{namespace}:returns"
        current = self.documents.get(document_id)
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
        self.documents[document_id] = result
        return result

    def list_documents(self, namespace: str | None = None) -> list[KnowledgeDocumentSummary]:
        if "list" in self.fail_unknown_methods:
            raise RuntimeError("secret backend list failure")
        documents = list(self.documents.values())
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

    def get_document(self, document_id: str) -> KnowledgeDocumentDetail:
        if "detail" in self.fail_unknown_methods:
            raise RuntimeError("secret backend detail failure")
        document = self.documents.get(document_id)
        if document is None:
            raise KnowledgeDocumentNotFoundError(document_id)
        return KnowledgeDocumentDetail(**document.model_dump(exclude={"document_version"}))

    def delete_document(self, document_id: str) -> KnowledgeDocumentOperationResult:
        if "delete" in self.fail_unknown_methods:
            raise RuntimeError("secret backend delete failure")
        document = self.documents.pop(document_id, None)
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
        self.last_rechunk = {
            "document_id": document_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "keep_version": keep_version,
        }
        if "rechunk" in self.fail_unknown_methods:
            raise RuntimeError("secret backend rechunk failure")
        current = self.documents.get(document_id)
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
        self.documents[document_id] = result
        return result


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


def _client(service: FakeKnowledgeDocumentService) -> TestClient:
    return TestClient(create_app(knowledge_document_service=service))


class FailingChatOnlyService:
    def chat(self, payload: object) -> object:
        raise AssertionError("chat should not be called")


def test_create_app_with_chat_service_does_not_create_document_service() -> None:
    app = create_app(chat_service=FailingChatOnlyService())  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert not hasattr(app.state, "knowledge_document_service")


def test_register_knowledge_document_returns_document_payload() -> None:
    service = FakeKnowledgeDocumentService()

    with _client(service) as client:
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
    assert service.last_register == {
        "namespace": "faq",
        "source_path": "faq/returns.json",
        "chunk_size": 120,
        "chunk_overlap": 20,
        "keep_version": False,
    }


def test_register_knowledge_document_rejects_invalid_chunk_params() -> None:
    service = FakeKnowledgeDocumentService()

    with _client(service) as client:
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
    service = FakeKnowledgeDocumentService()
    service.register_document("faq", "faq/returns.json", 120, 20, False)
    service.register_document("products", "products/laptop.json", 120, 20, False)

    with _client(service) as client:
        response = client.get("/knowledge/documents", params={"namespace": "faq"})

    assert response.status_code == 200
    payload = response.json()
    assert [document["namespace"] for document in payload["documents"]] == ["faq"]


def test_get_knowledge_document_returns_detail() -> None:
    service = FakeKnowledgeDocumentService()
    created = service.register_document("faq", "faq/returns.json", 120, 20, False)

    with _client(service) as client:
        response = client.get(f"/knowledge/documents/{created.document_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == created.document_id
    assert payload["versions"][0]["chunk_size"] == 120
    assert payload["last_error"] is None


def test_get_knowledge_document_missing_returns_404() -> None:
    service = FakeKnowledgeDocumentService()

    with _client(service) as client:
        response = client.get("/knowledge/documents/missing")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "KNOWLEDGE_DOCUMENT_NOT_FOUND"


def test_delete_knowledge_document_returns_deleted_payload() -> None:
    service = FakeKnowledgeDocumentService()
    created = service.register_document("faq", "faq/returns.json", 120, 20, False)

    with _client(service) as client:
        response = client.delete(f"/knowledge/documents/{created.document_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    assert service.list_documents() == []


def test_rechunk_knowledge_document_overwrites_by_default() -> None:
    service = FakeKnowledgeDocumentService()
    created = service.register_document("faq", "faq/returns.json", 120, 20, False)

    with _client(service) as client:
        response = client.post(
            f"/knowledge/documents/{created.document_id}/rechunk",
            json={"chunk_size": 80, "chunk_overlap": 10},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_version"] == 2
    assert [version["document_version"] for version in payload["versions"]] == [2]
    assert service.last_rechunk == {
        "document_id": created.document_id,
        "chunk_size": 80,
        "chunk_overlap": 10,
        "keep_version": False,
    }


def test_rechunk_knowledge_document_can_keep_previous_version() -> None:
    service = FakeKnowledgeDocumentService()
    created = service.register_document("faq", "faq/returns.json", 120, 20, False)

    with _client(service) as client:
        response = client.post(
            f"/knowledge/documents/{created.document_id}/rechunk",
            json={"chunk_size": 80, "chunk_overlap": 10, "keep_version": True},
        )

    assert response.status_code == 200
    payload = response.json()
    assert [version["document_version"] for version in payload["versions"]] == [1, 2]


def test_store_errors_return_500_with_structured_detail() -> None:
    service = FakeKnowledgeDocumentService()
    service.fail_store = True

    with _client(service) as client:
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


def test_unknown_register_error_returns_structured_safe_500() -> None:
    service = FakeKnowledgeDocumentService()
    service.fail_unknown_methods.add("register")

    with _client(service) as client:
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
        service = FakeKnowledgeDocumentService()
        service.register_document("faq", "faq/returns.json", 120, 20, False)
        service.fail_unknown_methods.add(method_name)
        with _client(service) as client:
            response = client.request(http_method, url, json=json_body)

        assert response.status_code == 500
        assert response.json()["detail"] == {
            "code": "KNOWLEDGE_DOCUMENT_INTERNAL_ERROR",
            "message": "Knowledge document operation failed.",
        }

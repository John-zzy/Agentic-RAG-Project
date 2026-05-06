from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.config.settings import AppSettings, VectorStoreConfig
from backend.knowledge.base.store import VectorStore, VectorStoreDocument, VectorStoreHealth
from backend.knowledge.documents.service import (
    KnowledgeDocumentNotFoundError,
    KnowledgeDocumentService,
    KnowledgeDocumentStoreError,
)
from backend.tests.test_support import tmp_path
from backend.knowledge.documents import (
    build_document_chunks,
    build_document_id,
    build_source_record_id,
    load_document_records,
    validate_chunking,
    validate_namespace,
    validate_source_path,
)


class InMemoryDocumentStore(VectorStore):
    def __init__(self, app_settings: AppSettings, *, fail_on_chunks: bool = False) -> None:
        super().__init__(app_settings)
        self.documents: dict[str, dict[str, Any]] = {}
        self.chunks: dict[str, dict[str, Any]] = {}
        self.fail_on_chunks = fail_on_chunks

    def ensure_collections(self) -> None:
        return None

    def upsert_documents(self, namespace: str, documents: list[VectorStoreDocument]) -> None:
        return None

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        return []

    def delete_documents(self, namespace: str, ids: list[str]) -> None:
        return None

    def healthcheck(self) -> VectorStoreHealth:
        return VectorStoreHealth(provider="memory", available=True)

    def ensure_document_indexes(self) -> None:
        return None

    def upsert_document_record(self, record: dict[str, Any]) -> None:
        self.documents[record["document_id"]] = dict(record)

    def get_document_record(self, document_id: str) -> dict[str, Any] | None:
        record = self.documents.get(document_id)
        if record is None or record.get("status") == "deleted":
            return None
        return dict(record)

    def list_document_records(self, namespace: str | None = None) -> list[dict[str, Any]]:
        records = [
            dict(record)
            for record in self.documents.values()
            if record.get("status") != "deleted" and (namespace is None or record.get("namespace") == namespace)
        ]
        return sorted(records, key=lambda record: record["source_path"])

    def upsert_document_chunks(self, chunks: list[VectorStoreDocument]) -> None:
        if self.fail_on_chunks:
            raise RuntimeError("chunk write failed")
        for chunk in chunks:
            self.chunks[chunk.id] = {
                "content": chunk.content,
                "embedding": chunk.embedding or self.build_embedding(chunk.content),
                **chunk.metadata,
            }

    def deactivate_document_chunks(self, document_id: str, document_version: int | None = None) -> None:
        for chunk in self.chunks.values():
            if chunk.get("document_id") != document_id:
                continue
            if document_version is not None and chunk.get("document_version") != document_version:
                continue
            chunk["is_active"] = False

    def delete_document_record(self, document_id: str) -> None:
        if document_id in self.documents:
            self.documents[document_id]["status"] = "deleted"

    def count_chunks(
        self,
        *,
        document_id: str,
        document_version: int,
        is_active: bool,
    ) -> int:
        return len(
            [
                chunk
                for chunk in self.chunks.values()
                if chunk["document_id"] == document_id
                and chunk["document_version"] == document_version
                and chunk["is_active"] is is_active
            ]
        )


class FailsOnceOnChunksStore(InMemoryDocumentStore):
    def __init__(self, app_settings: AppSettings) -> None:
        super().__init__(app_settings)
        self.fail_next_chunk_write = False
        self.deleted_chunk_ids: list[str] = []

    def upsert_document_chunks(self, chunks: list[VectorStoreDocument]) -> None:
        if self.fail_next_chunk_write:
            self.fail_next_chunk_write = False
            raise RuntimeError("chunk write failed")
        super().upsert_document_chunks(chunks)

    def delete_document_chunks(self, chunk_ids: list[str]) -> None:
        self.deleted_chunk_ids.extend(chunk_ids)
        for chunk_id in chunk_ids:
            self.chunks.pop(chunk_id, None)


class NonDocumentCapableStore(InMemoryDocumentStore):
    def get_document_record(self, document_id: str) -> dict[str, Any] | None:
        raise NotImplementedError("document management unavailable")

    def list_document_records(self, namespace: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError("document management unavailable")


class FailsOnceOnRecordWriteStore(InMemoryDocumentStore):
    def __init__(self, app_settings: AppSettings) -> None:
        super().__init__(app_settings)
        self.fail_next_record_write = False

    def upsert_document_record(self, record: dict[str, Any]) -> None:
        if self.fail_next_record_write:
            self.fail_next_record_write = False
            raise RuntimeError("record write failed")
        super().upsert_document_record(record)


@pytest.fixture
def document_app_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        data_dir=tmp_path,
        vector_store=VectorStoreConfig(provider="chroma"),
    )


@pytest.fixture
def files_root(document_app_settings: AppSettings) -> Path:
    root = document_app_settings.data_dir / "files"
    (root / "faq").mkdir(parents=True, exist_ok=True)
    (root / "faq" / "returns.json").write_text(
        '[{"id":"r1","title":"退货政策","content":"7天无理由退货，保持包装完整。"}]',
        encoding="utf-8",
    )
    (root / "products").mkdir(parents=True, exist_ok=True)
    (root / "products" / "laptop.json").write_text(
        '[{"id":"p1","title":"轻薄本","description":"适合办公和会议。"}]',
        encoding="utf-8",
    )
    return root


@pytest.fixture
def store(document_app_settings: AppSettings) -> InMemoryDocumentStore:
    return InMemoryDocumentStore(document_app_settings)


@pytest.fixture
def service(
    document_app_settings: AppSettings,
    files_root: Path,
    store: InMemoryDocumentStore,
) -> KnowledgeDocumentService:
    return KnowledgeDocumentService(
        app_settings=document_app_settings,
        store=store,
        files_root=files_root,
    )


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


@pytest.mark.parametrize("namespace", ["", " faq", "faq/docs", "../faq", "faq-docs"])
def test_validate_namespace_rejects_unsafe_names(namespace: str) -> None:
    with pytest.raises(ValueError, match="namespace"):
        validate_namespace(namespace)


def test_build_document_id_is_stable_for_normalized_source_path() -> None:
    document_id = build_document_id(namespace="faq", source_path="faq\\returns.json")

    assert document_id == build_document_id(namespace="faq", source_path="faq/returns.json")
    assert len(document_id) == 64


def test_build_source_record_id_is_stable_for_canonical_json() -> None:
    first = build_source_record_id("faq/returns.json", {"title": "退货", "content": "7天"}, 0)
    second = build_source_record_id("faq/returns.json", {"content": "7天", "title": "退货"}, 0)

    assert first == second
    assert len(first) == 64


@pytest.mark.parametrize("source_path", ["..\\secret.json", "../secret.json"])
def test_validate_source_path_rejects_path_traversal(tmp_path: Path, source_path: str) -> None:
    with pytest.raises(ValueError, match="source_path"):
        validate_source_path(source_path=source_path, data_root=tmp_path)


@pytest.mark.parametrize("payload", ["[]", '{"id":"r1","content":"x"}'])
def test_load_document_records_rejects_invalid_json_roots(tmp_path: Path, payload: str) -> None:
    source = tmp_path / "files" / "faq.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        load_document_records(namespace="faq", source_path=source, data_root=tmp_path / "files")


def test_load_document_records_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document_records(namespace="faq", source_path="missing.json", data_root=tmp_path)


def test_load_document_records_rejects_non_object_records(tmp_path: Path) -> None:
    source = tmp_path / "files" / "faq.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('["bad"]', encoding="utf-8")

    with pytest.raises(ValueError, match="object records"):
        load_document_records(namespace="faq", source_path=source, data_root=tmp_path / "files")


def test_build_document_chunks_adds_trace_metadata_and_stable_chunk_ids(tmp_path: Path) -> None:
    source = tmp_path / "files" / "faq.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('[{"id":"r1","title":"退货政策","content":"7天无理由退货，保持包装完整。"}]', encoding="utf-8")
    records = load_document_records(namespace="faq", source_path=source, data_root=tmp_path / "files")
    document_id = build_document_id(namespace="faq", source_path=records[0].source_path)

    chunks = build_document_chunks(
        records=records,
        document_id=document_id,
        document_version=1,
        updated_at="2026-05-06T12:00:00Z",
        chunk_size=8,
        chunk_overlap=2,
    )

    assert chunks
    assert chunks[0].chunk_id == chunks[0].metadata["chunk_id"]
    assert chunks[0].chunk_id == build_document_chunks(
        records=records,
        document_id=document_id,
        document_version=1,
        updated_at="2026-05-06T12:00:00Z",
        chunk_size=8,
        chunk_overlap=2,
    )[0].chunk_id
    assert chunks[0].metadata == {
        "document_id": document_id,
        "document_version": 1,
        "namespace": "faq",
        "source_type": "json",
        "source_path": "faq.json",
        "source_record_id": records[0].source_record_id,
        "chunk_id": chunks[0].chunk_id,
        "chunk_index": 0,
        "updated_at": "2026-05-06T12:00:00Z",
    }


def test_register_document_persists_document_record_and_chunks(
    service: KnowledgeDocumentService,
    store: InMemoryDocumentStore,
) -> None:
    result = service.register_document(
        namespace="faq",
        source_path="faq/returns.json",
        chunk_size=12,
        chunk_overlap=2,
        keep_version=False,
    )

    assert result.document_id
    assert result.document_version == 1
    assert result.active_version == 1
    assert result.status == "active"
    assert result.chunk_count > 0
    assert len([chunk for chunk in store.chunks.values() if chunk["is_active"]]) == result.chunk_count

    detail = service.get_document(result.document_id)
    assert detail.active_version == 1
    assert detail.source_path == "faq/returns.json"
    assert detail.chunk_size == 12
    assert detail.chunk_overlap == 2


def test_repeated_register_reuses_document_id_and_overwrites_active_version_by_default(
    service: KnowledgeDocumentService,
    store: InMemoryDocumentStore,
) -> None:
    first = service.register_document("faq", "faq/returns.json", 12, 2, False)
    second = service.register_document("faq", "faq/returns.json", 8, 1, False)

    assert second.document_id == first.document_id
    assert second.document_version == 2
    assert second.active_version == 2
    assert [version.document_version for version in second.versions] == [2]
    assert all(
        chunk["document_version"] != 1 or chunk["is_active"] is False
        for chunk in store.chunks.values()
        if chunk["document_id"] == first.document_id
    )


def test_rechunk_document_keeps_previous_version_when_requested(
    service: KnowledgeDocumentService,
    store: InMemoryDocumentStore,
) -> None:
    first = service.register_document("faq", "faq/returns.json", 12, 2, False)
    second = service.rechunk_document(first.document_id, chunk_size=8, chunk_overlap=1, keep_version=True)

    assert second.document_version == 2
    assert second.active_version == 2
    assert second.chunk_size == 8
    assert second.chunk_overlap == 1
    assert any(version.document_version == 1 for version in second.versions)
    assert store.count_chunks(document_id=first.document_id, document_version=2, is_active=True) > 0
    assert store.count_chunks(document_id=first.document_id, document_version=1, is_active=True) == 0
    assert store.count_chunks(document_id=first.document_id, document_version=1, is_active=False) > 0


def test_register_document_keep_version_keeps_new_chunks_active(
    service: KnowledgeDocumentService,
    store: InMemoryDocumentStore,
) -> None:
    first = service.register_document("faq", "faq/returns.json", 12, 2, False)
    second = service.register_document("faq", "faq/returns.json", 8, 1, True)

    assert second.document_id == first.document_id
    assert second.active_version == 2
    assert any(version.document_version == 1 for version in second.versions)
    assert store.count_chunks(document_id=first.document_id, document_version=2, is_active=True) > 0
    assert store.count_chunks(document_id=first.document_id, document_version=1, is_active=True) == 0
    assert store.count_chunks(document_id=first.document_id, document_version=1, is_active=False) > 0


def test_list_documents_filters_namespace_and_excludes_deleted(
    service: KnowledgeDocumentService,
) -> None:
    faq = service.register_document("faq", "faq/returns.json", 12, 2, False)
    product = service.register_document("products", "products/laptop.json", 12, 2, False)

    service.delete_document(faq.document_id)

    all_documents = service.list_documents()
    product_documents = service.list_documents(namespace="products")

    assert [document.document_id for document in all_documents] == [product.document_id]
    assert [document.namespace for document in product_documents] == ["products"]


def test_get_document_missing_raises_service_error(service: KnowledgeDocumentService) -> None:
    with pytest.raises(KnowledgeDocumentNotFoundError, match="missing"):
        service.get_document("missing")


def test_delete_keeps_source_file_and_removes_document_from_default_paths(
    service: KnowledgeDocumentService,
    files_root: Path,
    store: InMemoryDocumentStore,
) -> None:
    result = service.register_document("faq", "faq/returns.json", 12, 2, False)

    service.delete_document(result.document_id)

    assert (files_root / "faq" / "returns.json").exists()
    assert service.list_documents() == []
    with pytest.raises(KnowledgeDocumentNotFoundError):
        service.get_document(result.document_id)
    assert all(
        chunk["is_active"] is False
        for chunk in store.chunks.values()
        if chunk["document_id"] == result.document_id
    )


def test_rechunk_document_overwrites_old_active_version_by_default(
    service: KnowledgeDocumentService,
) -> None:
    first = service.register_document("faq", "faq/returns.json", 20, 0, False)

    result = service.rechunk_document(first.document_id, chunk_size=8, chunk_overlap=1)

    assert result.document_version == 2
    assert result.active_version == 2
    assert result.chunk_count != first.chunk_count
    assert [version.document_version for version in result.versions] == [2]


def test_register_document_records_failure_status_and_raises_clear_error(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    failing_store = InMemoryDocumentStore(document_app_settings, fail_on_chunks=True)
    service = KnowledgeDocumentService(
        app_settings=document_app_settings,
        store=failing_store,
        files_root=files_root,
    )

    with pytest.raises(KnowledgeDocumentStoreError, match="chunk write failed"):
        service.register_document("faq", "faq/returns.json", 12, 2, False)

    record = next(iter(failing_store.documents.values()))
    assert record["status"] == "failed"
    assert record["last_error"] == "chunk write failed"


def test_rechunk_failure_preserves_previous_active_record_and_chunks(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    store = FailsOnceOnChunksStore(document_app_settings)
    service = KnowledgeDocumentService(
        app_settings=document_app_settings,
        store=store,
        files_root=files_root,
    )
    first = service.register_document("faq", "faq/returns.json", 12, 2, False)
    active_v1_count = len(
        [
            chunk
            for chunk in store.chunks.values()
            if chunk["document_id"] == first.document_id
            and chunk["document_version"] == 1
            and chunk["is_active"] is True
        ]
    )

    store.fail_next_chunk_write = True
    with pytest.raises(KnowledgeDocumentStoreError, match="chunk write failed"):
        service.rechunk_document(first.document_id, chunk_size=8, chunk_overlap=1)

    detail = service.get_document(first.document_id)
    assert detail.status == "active"
    assert detail.active_version == 1
    assert [version.document_version for version in detail.versions] == [1]
    assert len(
        [
            chunk
            for chunk in store.chunks.values()
            if chunk["document_id"] == first.document_id
            and chunk["document_version"] == 1
            and chunk["is_active"] is True
        ]
    ) == active_v1_count


def test_rechunk_publish_failure_preserves_previous_active_record_and_chunks(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    store = FailsOnceOnRecordWriteStore(document_app_settings)
    service = KnowledgeDocumentService(
        app_settings=document_app_settings,
        store=store,
        files_root=files_root,
    )
    first = service.register_document("faq", "faq/returns.json", 12, 2, False)
    active_v1_count = len(
        [
            chunk
            for chunk in store.chunks.values()
            if chunk["document_id"] == first.document_id
            and chunk["document_version"] == 1
            and chunk["is_active"] is True
        ]
    )

    store.fail_next_record_write = True
    with pytest.raises(KnowledgeDocumentStoreError, match="record write failed"):
        service.rechunk_document(first.document_id, chunk_size=8, chunk_overlap=1)

    detail = service.get_document(first.document_id)
    assert detail.status == "active"
    assert detail.active_version == 1
    assert [version.document_version for version in detail.versions] == [1]
    assert len(
        [
            chunk
            for chunk in store.chunks.values()
            if chunk["document_id"] == first.document_id
            and chunk["document_version"] == 1
            and chunk["is_active"] is True
        ]
    ) == active_v1_count


def test_non_document_capable_store_errors_are_wrapped(
    document_app_settings: AppSettings,
) -> None:
    service = KnowledgeDocumentService(
        app_settings=document_app_settings,
        store=NonDocumentCapableStore(document_app_settings),
    )

    with pytest.raises(KnowledgeDocumentStoreError, match="does not support document management"):
        service.list_documents()


def test_register_document_wraps_non_document_capable_store_error(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    service = KnowledgeDocumentService(
        app_settings=document_app_settings,
        store=NonDocumentCapableStore(document_app_settings),
        files_root=files_root,
    )

    with pytest.raises(KnowledgeDocumentStoreError, match="does not support document management"):
        service.register_document("faq", "faq/returns.json", 12, 2, False)

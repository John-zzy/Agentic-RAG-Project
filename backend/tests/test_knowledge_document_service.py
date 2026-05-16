from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest

from backend.platform.config.settings import AppSettings, VectorStoreConfig
from backend.platform.knowledge.base.store import VectorStore, VectorStoreDocument, VectorStoreHealth
from backend.platform.knowledge.documents import (
    KnowledgeDocumentApplicationService,
    KnowledgeDocumentProcessor,
    KnowledgeDocumentNotFoundError,
    KnowledgeDocumentQueryService,
    KnowledgeDocumentStoreError,
)
from backend.tests.test_support import tmp_path
from backend.platform.knowledge.documents import (
    build_document_chunks,
    build_document_id,
    build_preprocess_preview,
    process_document_records,
    build_source_record_id,
    load_document_records,
    validate_chunking,
    validate_namespace,
    validate_source_path,
)


@dataclass(frozen=True)
class SplitKnowledgeDocumentServices:
    """明确区分读写职责，避免测试继续依赖已拆除的聚合服务。"""

    application: KnowledgeDocumentApplicationService
    query: KnowledgeDocumentQueryService


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

    def search_document_chunks(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
    ) -> list[Any]:
        del query, top_k, namespace
        return []

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

    def activate_document_chunks(self, document_id: str, document_version: int) -> None:
        for chunk in self.chunks.values():
            if chunk.get("document_id") != document_id:
                continue
            if chunk.get("document_version") != document_version:
                continue
            chunk["is_active"] = True

    def delete_document_record(self, document_id: str) -> None:
        if document_id in self.documents:
            self.documents[document_id]["status"] = "deleted"

    def delete_document_chunks(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self.chunks.pop(chunk_id, None)

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


class FailsOnceOnDeactivateStore(InMemoryDocumentStore):
    def __init__(self, app_settings: AppSettings) -> None:
        super().__init__(app_settings)
        self.fail_next_deactivate = False
        self.fail_cleanup = False

    def deactivate_document_chunks(self, document_id: str, document_version: int | None = None) -> None:
        if self.fail_next_deactivate:
            self.fail_next_deactivate = False
            super().deactivate_document_chunks(document_id, document_version)
            raise RuntimeError("deactivate failed")
        super().deactivate_document_chunks(document_id, document_version)

    def delete_document_chunks(self, chunk_ids: list[str]) -> None:
        if self.fail_cleanup:
            raise RuntimeError("cleanup failed")
        super().delete_document_chunks(chunk_ids)


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
    (root / "faq" / "manual.pdf").write_text("fake-pdf", encoding="utf-8")
    (root / "faq" / "messy.md").write_text(
        "# Table of Contents\n\nhttps://example.com\n\n<div>退货政策</div>\n\n  7天无理由退货  \n",
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
def split_services(
    document_app_settings: AppSettings,
    files_root: Path,
    store: InMemoryDocumentStore,
) -> SplitKnowledgeDocumentServices:
    # 这里让两个服务共享同一个内存仓储，才能真实覆盖“写后立即读”的协作路径。
    return SplitKnowledgeDocumentServices(
        application=KnowledgeDocumentApplicationService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
        query=KnowledgeDocumentQueryService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
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
    processed = process_document_records(
        namespace="faq",
        source_path=records[0].source_path,
        records=records,
        processing_rules=[],
    )
    document_id = build_document_id(namespace="faq", source_path=records[0].source_path)

    chunks = build_document_chunks(
        records=processed.records,
        document_id=document_id,
        document_version=1,
        updated_at="2026-05-06T12:00:00Z",
        chunk_size=8,
        chunk_overlap=2,
    )

    assert chunks
    assert chunks[0].chunk_id == chunks[0].metadata["chunk_id"]
    assert chunks[0].chunk_id == build_document_chunks(
        records=processed.records,
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
        "source_record_index": 0,
        "chunk_id": chunks[0].chunk_id,
        "chunk_index": 0,
        "applied_rules": [],
        "raw_content_hash": processed.records[0].raw_content_hash,
        "processed_content_hash": processed.records[0].processed_content_hash,
        "updated_at": "2026-05-06T12:00:00Z",
        "is_active": True,
    }


def test_processing_pipeline_applies_rules_and_builds_preview(tmp_path: Path) -> None:
    source = tmp_path / "files" / "faq.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "# Table of Contents\n\nhttps://example.com\n\n<div>退货政策</div>\n\n  7天无理由退货  \n",
        encoding="utf-8",
    )
    records = load_document_records(namespace="faq", source_path=source, data_root=tmp_path / "files")

    preview = build_preprocess_preview(
        namespace="faq",
        source_path=records[0].source_path,
        records=records,
        chunk_size=12,
        chunk_overlap=2,
        processing_rules=[
            "trim_whitespace",
            "strip_html_tags",
            "remove_url_lines",
            "remove_markdown_boilerplate",
        ],
    )

    assert preview.can_index is True
    assert preview.source_type == "md"
    assert [rule.rule_id for rule in preview.selected_rules] == [
        "trim_whitespace",
        "strip_html_tags",
        "remove_url_lines",
        "remove_markdown_boilerplate",
    ]
    assert preview.processing_stats is not None
    assert preview.processing_stats.raw_record_count == 1
    assert preview.processing_stats.processed_record_count == 1
    assert preview.chunk_size == 12
    assert preview.chunk_overlap == 2
    assert "Table of Contents" not in preview.processed_samples[0].content
    assert "https://example.com" not in preview.processed_samples[0].content
    assert "<div>" not in preview.processed_samples[0].content


def test_processing_pipeline_dedupes_and_drops_empty_records(tmp_path: Path) -> None:
    source = tmp_path / "files" / "faq.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(
                [
                    {"title": "退货", "content": " 7天无理由 "},
                    {"title": "退货", "content": "7天无理由"},
                    {"content": "https://example.com"},
                ],
                ensure_ascii=False,
            ),
        encoding="utf-8",
    )
    records = load_document_records(namespace="faq", source_path=source, data_root=tmp_path / "files")

    result = KnowledgeDocumentProcessor().process(
        namespace="faq",
        source_path=records[0].source_path,
        records=records,
        processing_rules=["trim_whitespace", "remove_url_lines", "drop_empty_records", "dedupe_records"],
    )

    assert result.can_index is True
    assert result.processing_stats.raw_record_count == 3
    assert result.processing_stats.processed_record_count == 1
    assert result.processing_stats.removed_record_count == 2
    assert len(result.records) == 1
    assert any(warning.code == "dropped_duplicate_record" for warning in result.warnings)
    assert any(warning.code == "dropped_empty_record" for warning in result.warnings)
    kept_record = result.records[0]
    assert kept_record.raw_content_hash
    assert kept_record.processed_content_hash
    assert kept_record.source_record_id == records[0].source_record_id


def test_processing_pipeline_blocks_unsupported_rule_for_source_type(tmp_path: Path) -> None:
    source = tmp_path / "files" / "faq.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("退货政策", encoding="utf-8")
    records = load_document_records(namespace="faq", source_path=source, data_root=tmp_path / "files")

    with pytest.raises(ValueError, match="remove_markdown_boilerplate"):
        build_preprocess_preview(
            namespace="faq",
            source_path=records[0].source_path,
            records=records,
            chunk_size=12,
            chunk_overlap=2,
            processing_rules=["remove_markdown_boilerplate"],
        )


def test_register_document_persists_document_record_and_chunks(
    split_services: SplitKnowledgeDocumentServices,
    store: InMemoryDocumentStore,
) -> None:
    result = split_services.application.register_document(
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

    detail = split_services.query.get_document(result.document_id)
    assert detail.active_version == 1
    assert detail.source_path == "faq/returns.json"
    assert detail.chunk_size == 12
    assert detail.chunk_overlap == 2
    assert detail.processing_rules == []
    assert detail.processing_stats is not None
    assert detail.provenance_enabled is True


def test_preprocess_preview_returns_supported_rules_and_processed_samples(
    split_services: SplitKnowledgeDocumentServices,
) -> None:
    preview = split_services.application.preprocess_preview(
        namespace="faq",
        source_path="faq/messy.md",
        chunk_size=12,
        chunk_overlap=2,
        processing_rules=[
            "trim_whitespace",
            "strip_html_tags",
            "remove_url_lines",
            "remove_markdown_boilerplate",
        ],
    )

    assert preview.can_index is True
    assert preview.source_type == "md"
    assert preview.chunk_size == 12
    assert preview.chunk_overlap == 2
    assert [rule.rule_id for rule in preview.selected_rules] == [
        "trim_whitespace",
        "strip_html_tags",
        "remove_url_lines",
        "remove_markdown_boilerplate",
    ]
    assert len(preview.processed_samples) > 1
    assert all(len(sample.content) <= 12 for sample in preview.processed_samples)
    assert "Table of Contents" not in "".join(sample.content for sample in preview.processed_samples)


def test_preprocess_preview_for_unsupported_type_returns_warning(
    split_services: SplitKnowledgeDocumentServices,
) -> None:
    preview = split_services.application.preprocess_preview(
        namespace="faq",
        source_path="faq/manual.pdf",
        chunk_size=12,
        chunk_overlap=2,
        processing_rules=[],
    )

    assert preview.can_index is False
    assert preview.source_type == "pdf"
    assert preview.warnings
    assert preview.warnings[0].code == "unsupported_source_type"


def test_reprocess_document_updates_processing_rules_and_stats(
    split_services: SplitKnowledgeDocumentServices,
    store: InMemoryDocumentStore,
) -> None:
    created = split_services.application.register_document("faq", "faq/messy.md", 12, 2, False)

    updated = split_services.application.reprocess_document(
        created.document_id,
        chunk_size=12,
        chunk_overlap=2,
        keep_version=True,
        processing_rules=[
            "trim_whitespace",
            "strip_html_tags",
            "remove_url_lines",
            "remove_markdown_boilerplate",
        ],
    )

    assert updated.document_version == 2
    assert updated.active_version == 2
    assert updated.processing_rules == [
        "trim_whitespace",
        "strip_html_tags",
        "remove_url_lines",
        "remove_markdown_boilerplate",
    ]
    assert updated.processing_stats is not None
    assert any(version.document_version == 1 for version in updated.versions)
    assert store.count_chunks(document_id=created.document_id, document_version=1, is_active=True) == 0
    assert store.count_chunks(document_id=created.document_id, document_version=2, is_active=True) > 0


def test_register_document_publish_failure_records_failed_status_and_cleans_chunks(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    store = FailsOnceOnRecordWriteStore(document_app_settings)
    application_service = KnowledgeDocumentApplicationService(
        app_settings=document_app_settings,
        repository=store,
        files_root=files_root,
    )
    store.fail_next_record_write = True

    with pytest.raises(KnowledgeDocumentStoreError, match="record write failed"):
        application_service.register_document("faq", "faq/returns.json", 12, 2, False)

    record = next(iter(store.documents.values()))
    assert record["status"] == "failed"
    assert record["last_error"] == "record write failed"
    assert store.chunks == {}


def test_reprocess_failure_preserves_previous_active_record_and_chunks(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    store = FailsOnceOnChunksStore(document_app_settings)
    split_services = SplitKnowledgeDocumentServices(
        application=KnowledgeDocumentApplicationService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
        query=KnowledgeDocumentQueryService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
    )
    first = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)
    active_v1_count = store.count_chunks(
        document_id=first.document_id,
        document_version=1,
        is_active=True,
    )

    store.fail_next_chunk_write = True
    with pytest.raises(KnowledgeDocumentStoreError, match="chunk write failed"):
        split_services.application.reprocess_document(
            first.document_id,
            chunk_size=8,
            chunk_overlap=1,
            processing_rules=["trim_whitespace"],
        )

    detail = split_services.query.get_document(first.document_id)
    assert detail.status == "active"
    assert detail.active_version == 1
    assert detail.processing_rules == []
    assert [version.document_version for version in detail.versions] == [1]
    assert store.count_chunks(
        document_id=first.document_id,
        document_version=1,
        is_active=True,
    ) == active_v1_count


def test_reprocess_publish_failure_preserves_previous_active_record_and_chunks(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    store = FailsOnceOnRecordWriteStore(document_app_settings)
    split_services = SplitKnowledgeDocumentServices(
        application=KnowledgeDocumentApplicationService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
        query=KnowledgeDocumentQueryService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
    )
    first = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)
    active_v1_count = store.count_chunks(
        document_id=first.document_id,
        document_version=1,
        is_active=True,
    )

    store.fail_next_record_write = True
    with pytest.raises(KnowledgeDocumentStoreError, match="record write failed"):
        split_services.application.reprocess_document(
            first.document_id,
            chunk_size=8,
            chunk_overlap=1,
            processing_rules=["trim_whitespace"],
        )

    detail = split_services.query.get_document(first.document_id)
    assert detail.status == "active"
    assert detail.active_version == 1
    assert detail.processing_rules == []
    assert [version.document_version for version in detail.versions] == [1]
    assert store.count_chunks(
        document_id=first.document_id,
        document_version=1,
        is_active=True,
    ) == active_v1_count


def test_reprocess_deactivate_failure_restores_previous_record_and_chunks(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    store = FailsOnceOnDeactivateStore(document_app_settings)
    split_services = SplitKnowledgeDocumentServices(
        application=KnowledgeDocumentApplicationService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
        query=KnowledgeDocumentQueryService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
    )
    first = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)
    active_v1_count = store.count_chunks(
        document_id=first.document_id,
        document_version=1,
        is_active=True,
    )

    store.fail_next_deactivate = True
    with pytest.raises(KnowledgeDocumentStoreError, match="deactivate failed"):
        split_services.application.reprocess_document(
            first.document_id,
            chunk_size=8,
            chunk_overlap=1,
            processing_rules=["trim_whitespace"],
        )

    detail = split_services.query.get_document(first.document_id)
    assert detail.status == "active"
    assert detail.active_version == 1
    assert detail.processing_rules == []
    assert [version.document_version for version in detail.versions] == [1]
    assert store.count_chunks(
        document_id=first.document_id,
        document_version=1,
        is_active=True,
    ) == active_v1_count
    assert store.count_chunks(
        document_id=first.document_id,
        document_version=2,
        is_active=True,
    ) == 0


def test_reprocess_cleanup_failure_does_not_mask_original_publish_error(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    store = FailsOnceOnDeactivateStore(document_app_settings)
    split_services = SplitKnowledgeDocumentServices(
        application=KnowledgeDocumentApplicationService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
        query=KnowledgeDocumentQueryService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
    )
    first = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)

    store.fail_next_deactivate = True
    store.fail_cleanup = True
    with pytest.raises(KnowledgeDocumentStoreError, match="deactivate failed") as exc_info:
        split_services.application.reprocess_document(
            first.document_id,
            chunk_size=8,
            chunk_overlap=1,
            processing_rules=["trim_whitespace"],
        )

    assert "cleanup failed" not in str(exc_info.value)
    assert split_services.query.get_document(first.document_id).active_version == 1


def test_default_service_reads_json_files_from_data_dir(document_app_settings: AppSettings) -> None:
    (document_app_settings.data_dir / "orders.json").write_text(
        '[{"order_id":"O1","status":"paid","total_amount":100}]',
        encoding="utf-8",
    )
    store = InMemoryDocumentStore(document_app_settings)
    application_service = KnowledgeDocumentApplicationService(
        app_settings=document_app_settings,
        repository=store,
    )

    result = application_service.register_document("faq", "orders.json", 12, 2, False)

    assert result.source_path == "orders.json"
    assert result.chunk_count > 0


def test_repeated_register_reuses_document_id_and_overwrites_active_version_by_default(
    split_services: SplitKnowledgeDocumentServices,
    store: InMemoryDocumentStore,
) -> None:
    first = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)
    second = split_services.application.register_document("faq", "faq/returns.json", 8, 1, False)

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
    split_services: SplitKnowledgeDocumentServices,
    store: InMemoryDocumentStore,
) -> None:
    first = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)
    second = split_services.application.rechunk_document(
        first.document_id,
        chunk_size=8,
        chunk_overlap=1,
        keep_version=True,
    )

    assert second.document_version == 2
    assert second.active_version == 2
    assert second.chunk_size == 8
    assert second.chunk_overlap == 1
    assert any(version.document_version == 1 for version in second.versions)
    assert store.count_chunks(document_id=first.document_id, document_version=2, is_active=True) > 0
    assert store.count_chunks(document_id=first.document_id, document_version=1, is_active=True) == 0
    assert store.count_chunks(document_id=first.document_id, document_version=1, is_active=False) > 0


def test_register_document_keep_version_keeps_new_chunks_active(
    split_services: SplitKnowledgeDocumentServices,
    store: InMemoryDocumentStore,
) -> None:
    first = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)
    second = split_services.application.register_document("faq", "faq/returns.json", 8, 1, True)

    assert second.document_id == first.document_id
    assert second.active_version == 2
    assert any(version.document_version == 1 for version in second.versions)
    assert store.count_chunks(document_id=first.document_id, document_version=2, is_active=True) > 0
    assert store.count_chunks(document_id=first.document_id, document_version=1, is_active=True) == 0
    assert store.count_chunks(document_id=first.document_id, document_version=1, is_active=False) > 0


def test_list_documents_filters_namespace_and_excludes_deleted(
    split_services: SplitKnowledgeDocumentServices,
) -> None:
    faq = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)
    product = split_services.application.register_document("products", "products/laptop.json", 12, 2, False)

    split_services.application.delete_document(faq.document_id)

    all_documents = split_services.query.list_documents()
    product_documents = split_services.query.list_documents(namespace="products")

    assert [document.document_id for document in all_documents] == [product.document_id]
    assert [document.namespace for document in product_documents] == ["products"]


def test_query_service_list_file_indexes_aggregates_file_status(
    document_app_settings: AppSettings,
    files_root: Path,
    store: InMemoryDocumentStore,
) -> None:
    application_service = KnowledgeDocumentApplicationService(
        app_settings=document_app_settings,
        repository=store,
        files_root=files_root,
    )
    query_service = KnowledgeDocumentQueryService(
        app_settings=document_app_settings,
        repository=store,
        files_root=files_root,
    )
    application_service.register_document("faq", "faq/returns.json", 12, 2, False)

    items = query_service.list_file_indexes()

    assert {item.source_path for item in items} == {
        "products/laptop.json",
        "faq/returns.json",
        "faq/messy.md",
        "faq/manual.pdf",
    }
    indexed_item = next(item for item in items if item.source_path == "faq/returns.json")
    unindexed_item = next(item for item in items if item.source_path == "products/laptop.json")
    unsupported_item = next(item for item in items if item.source_path == "faq/manual.pdf")
    assert indexed_item.indexed is True
    assert indexed_item.namespace == "faq"
    assert unindexed_item.indexed is False
    assert unindexed_item.status == "awaiting_processing"
    assert unsupported_item.status == "unsupported"
    assert unsupported_item.last_error == "当前文件类型 'pdf' 尚未接入处理与索引链路。"


def test_application_service_and_query_service_can_be_used_independently(
    document_app_settings: AppSettings,
    files_root: Path,
    store: InMemoryDocumentStore,
) -> None:
    application_service = KnowledgeDocumentApplicationService(
        app_settings=document_app_settings,
        repository=store,
        files_root=files_root,
    )
    query_service = KnowledgeDocumentQueryService(
        app_settings=document_app_settings,
        repository=store,
        files_root=files_root,
    )

    created = application_service.register_document("faq", "faq/returns.json", 12, 2, False)
    detail = query_service.get_document(created.document_id)
    deleted = application_service.delete_document(created.document_id)

    assert detail.document_id == created.document_id
    assert deleted.status == "deleted"
    with pytest.raises(KnowledgeDocumentNotFoundError):
        query_service.get_document(created.document_id)


def test_get_document_missing_raises_service_error(
    split_services: SplitKnowledgeDocumentServices,
) -> None:
    with pytest.raises(KnowledgeDocumentNotFoundError, match="missing"):
        split_services.query.get_document("missing")


def test_delete_keeps_source_file_and_removes_document_from_default_paths(
    split_services: SplitKnowledgeDocumentServices,
    files_root: Path,
    store: InMemoryDocumentStore,
) -> None:
    result = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)

    split_services.application.delete_document(result.document_id)

    assert (files_root / "faq" / "returns.json").exists()
    assert split_services.query.list_documents() == []
    with pytest.raises(KnowledgeDocumentNotFoundError):
        split_services.query.get_document(result.document_id)
    assert all(
        chunk["is_active"] is False
        for chunk in store.chunks.values()
        if chunk["document_id"] == result.document_id
    )


def test_rechunk_document_overwrites_old_active_version_by_default(
    split_services: SplitKnowledgeDocumentServices,
) -> None:
    first = split_services.application.register_document("faq", "faq/returns.json", 20, 0, False)

    result = split_services.application.rechunk_document(first.document_id, chunk_size=8, chunk_overlap=1)

    assert result.document_version == 2
    assert result.active_version == 2
    assert result.chunk_count != first.chunk_count
    assert [version.document_version for version in result.versions] == [2]
    assert result.processing_rules == []


def test_rechunk_document_reuses_active_processing_rules(
    split_services: SplitKnowledgeDocumentServices,
) -> None:
    first = split_services.application.register_document(
        "faq",
        "faq/messy.md",
        20,
        0,
        False,
        processing_rules=[
            "trim_whitespace",
            "strip_html_tags",
            "remove_url_lines",
            "remove_markdown_boilerplate",
        ],
    )

    result = split_services.application.rechunk_document(first.document_id, chunk_size=8, chunk_overlap=1)

    assert result.document_version == 2
    assert result.processing_rules == [
        "trim_whitespace",
        "strip_html_tags",
        "remove_url_lines",
        "remove_markdown_boilerplate",
    ]


def test_register_document_records_failure_status_and_raises_clear_error(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    failing_store = InMemoryDocumentStore(document_app_settings, fail_on_chunks=True)
    application_service = KnowledgeDocumentApplicationService(
        app_settings=document_app_settings,
        repository=failing_store,
        files_root=files_root,
    )

    with pytest.raises(KnowledgeDocumentStoreError, match="chunk write failed"):
        application_service.register_document("faq", "faq/returns.json", 12, 2, False)

    record = next(iter(failing_store.documents.values()))
    assert record["status"] == "failed"
    assert record["last_error"] == "chunk write failed"


def test_rechunk_failure_preserves_previous_active_record_and_chunks(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    store = FailsOnceOnChunksStore(document_app_settings)
    split_services = SplitKnowledgeDocumentServices(
        application=KnowledgeDocumentApplicationService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
        query=KnowledgeDocumentQueryService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
    )
    first = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)
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
        split_services.application.rechunk_document(first.document_id, chunk_size=8, chunk_overlap=1)

    detail = split_services.query.get_document(first.document_id)
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
    split_services = SplitKnowledgeDocumentServices(
        application=KnowledgeDocumentApplicationService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
        query=KnowledgeDocumentQueryService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
    )
    first = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)
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
        split_services.application.rechunk_document(first.document_id, chunk_size=8, chunk_overlap=1)

    detail = split_services.query.get_document(first.document_id)
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


def test_rechunk_deactivate_failure_restores_previous_record_and_chunks(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    store = FailsOnceOnDeactivateStore(document_app_settings)
    split_services = SplitKnowledgeDocumentServices(
        application=KnowledgeDocumentApplicationService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
        query=KnowledgeDocumentQueryService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
    )
    first = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)
    active_v1_count = store.count_chunks(
        document_id=first.document_id,
        document_version=1,
        is_active=True,
    )

    store.fail_next_deactivate = True
    with pytest.raises(KnowledgeDocumentStoreError, match="deactivate failed"):
        split_services.application.rechunk_document(first.document_id, chunk_size=8, chunk_overlap=1)

    detail = split_services.query.get_document(first.document_id)
    assert detail.status == "active"
    assert detail.active_version == 1
    assert [version.document_version for version in detail.versions] == [1]
    assert store.count_chunks(
        document_id=first.document_id,
        document_version=1,
        is_active=True,
    ) == active_v1_count
    assert store.count_chunks(
        document_id=first.document_id,
        document_version=2,
        is_active=True,
    ) == 0


def test_rechunk_cleanup_failure_does_not_mask_original_publish_error(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    store = FailsOnceOnDeactivateStore(document_app_settings)
    split_services = SplitKnowledgeDocumentServices(
        application=KnowledgeDocumentApplicationService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
        query=KnowledgeDocumentQueryService(
            app_settings=document_app_settings,
            repository=store,
            files_root=files_root,
        ),
    )
    first = split_services.application.register_document("faq", "faq/returns.json", 12, 2, False)

    store.fail_next_deactivate = True
    store.fail_cleanup = True
    with pytest.raises(KnowledgeDocumentStoreError, match="deactivate failed") as exc_info:
        split_services.application.rechunk_document(first.document_id, chunk_size=8, chunk_overlap=1)

    assert "cleanup failed" not in str(exc_info.value)
    assert split_services.query.get_document(first.document_id).active_version == 1


def test_non_document_capable_store_errors_are_wrapped(
    document_app_settings: AppSettings,
) -> None:
    query_service = KnowledgeDocumentQueryService(
        app_settings=document_app_settings,
        repository=NonDocumentCapableStore(document_app_settings),
    )

    with pytest.raises(KnowledgeDocumentStoreError, match="does not support document management"):
        query_service.list_documents()


def test_register_document_wraps_non_document_capable_store_error(
    document_app_settings: AppSettings,
    files_root: Path,
) -> None:
    application_service = KnowledgeDocumentApplicationService(
        app_settings=document_app_settings,
        repository=NonDocumentCapableStore(document_app_settings),
        files_root=files_root,
    )

    with pytest.raises(KnowledgeDocumentStoreError, match="does not support document management"):
        application_service.register_document("faq", "faq/returns.json", 12, 2, False)

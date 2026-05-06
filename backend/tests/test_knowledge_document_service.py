from __future__ import annotations

from pathlib import Path

import pytest

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

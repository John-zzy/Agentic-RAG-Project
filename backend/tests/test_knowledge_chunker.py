from __future__ import annotations

from dataclasses import dataclass

from backend.platform.knowledge.documents.chunker import (
    LangChainTextSplitterAdapter,
    SlidingWindowTextSplitter,
    build_document_chunks,
)
from backend.platform.knowledge.processing.schemas import ProcessedDocumentRecord


@dataclass(frozen=True)
class FakeLangChainSplitter:
    chunks: list[str]

    def split_text(self, text: str) -> list[str]:
        assert text == "abcdef"
        return list(self.chunks)


def test_build_document_chunks_uses_processed_record_metadata() -> None:
    record = ProcessedDocumentRecord(
        namespace="faq",
        source_path="faq/returns.json",
        source_type="json",
        source_record_id="record-1",
        record_index=3,
        raw_content="raw text",
        processed_content="abcdef",
        applied_rules=["trim_whitespace"],
        raw_content_hash="raw-hash",
        processed_content_hash="processed-hash",
    )

    chunks = build_document_chunks(
        records=[record],
        document_id="document-1",
        document_version=2,
        updated_at="2026-05-06T12:00:00Z",
        chunk_size=3,
        chunk_overlap=1,
    )

    assert [chunk.content for chunk in chunks] == ["abc", "cde", "ef"]
    assert chunks[0].chunk_id == build_document_chunks(
        records=[record],
        document_id="document-1",
        document_version=2,
        updated_at="2026-05-06T12:00:00Z",
        chunk_size=3,
        chunk_overlap=1,
    )[0].chunk_id
    assert chunks[0].metadata == {
        "document_id": "document-1",
        "document_version": 2,
        "namespace": "faq",
        "source_type": "json",
        "source_path": "faq/returns.json",
        "source_record_id": "record-1",
        "source_record_index": 3,
        "chunk_id": chunks[0].chunk_id,
        "chunk_index": 0,
        "applied_rules": ["trim_whitespace"],
        "raw_content_hash": "raw-hash",
        "processed_content_hash": "processed-hash",
        "updated_at": "2026-05-06T12:00:00Z",
        "is_active": True,
    }


def test_build_document_chunks_uses_injected_splitter_adapter() -> None:
    record = ProcessedDocumentRecord(
        namespace="faq",
        source_path="faq/returns.json",
        source_type="json",
        source_record_id="record-1",
        record_index=0,
        raw_content="raw text",
        processed_content="abcdef",
        applied_rules=[],
        raw_content_hash="raw-hash",
        processed_content_hash="processed-hash",
    )

    splitter = LangChainTextSplitterAdapter(splitter=FakeLangChainSplitter(["ab", "cd", "ef"]))
    chunks = build_document_chunks(
        records=[record],
        document_id="document-1",
        document_version=1,
        updated_at="2026-05-06T12:00:00Z",
        chunk_size=3,
        chunk_overlap=1,
        chunker=splitter,
    )

    assert [chunk.content for chunk in chunks] == ["ab", "cd", "ef"]


def test_sliding_window_text_splitter_preserves_current_overlap_semantics() -> None:
    splitter = SlidingWindowTextSplitter(chunk_size=4, chunk_overlap=1)

    assert splitter.split_text("abcdefg") == ["abcd", "defg", "g"]

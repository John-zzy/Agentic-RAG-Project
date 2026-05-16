from __future__ import annotations

import hashlib

import pytest

from backend.platform.knowledge.documents.schemas import DocumentRecord
from backend.platform.knowledge.processing import (
    KnowledgeDocumentProcessor,
    build_preprocess_preview,
    process,
    process_document_records,
    preview,
)


def _record(
    *,
    source_record_id: str,
    record_index: int,
    content: str,
    source_path: str = "faq/source.json",
    namespace: str = "faq",
) -> DocumentRecord:
    return DocumentRecord(
        namespace=namespace,
        source_path=source_path,
        source_record_id=source_record_id,
        record_index=record_index,
        content=content,
        record={"content": content},
    )


def test_process_document_records_applies_rules_in_stable_order() -> None:
    records = [
        _record(
            source_record_id="r1",
            record_index=0,
            content="  <div>Alpha</div>\nhttps://example.com  \n",
        )
    ]

    result = process_document_records(
        records,
        processing_rules=["strip_html_tags", "remove_url_lines", "trim_whitespace"],
    )

    assert result.processing_rules == ["strip_html_tags", "remove_url_lines", "trim_whitespace"]
    assert [record.applied_rules for record in result.records] == [
        ["strip_html_tags", "remove_url_lines", "trim_whitespace"]
    ]
    assert result.records[0].processed_content == "Alpha"


def test_process_document_records_drops_empty_records_and_reports_warning() -> None:
    records = [
        _record(source_record_id="r1", record_index=0, content="   \n  "),
        _record(source_record_id="r2", record_index=1, content="  keep  "),
    ]

    result = process_document_records(
        records,
        processing_rules=["trim_whitespace", "drop_empty_records"],
    )

    assert [record.source_record_id for record in result.records] == ["r2"]
    assert result.processing_stats.raw_record_count == 2
    assert result.processing_stats.processed_record_count == 1
    assert result.processing_stats.removed_record_count == 1
    assert result.warnings[0].code == "dropped_empty_record"
    assert result.warnings[0].source_record_id == "r1"


def test_process_document_records_dedupes_after_normalization() -> None:
    records = [
        _record(source_record_id="r1", record_index=0, content=" Alpha "),
        _record(source_record_id="r2", record_index=1, content="Alpha"),
        _record(source_record_id="r3", record_index=2, content="Beta"),
    ]

    result = process_document_records(
        records,
        processing_rules=["trim_whitespace", "dedupe_records"],
    )

    assert [record.source_record_id for record in result.records] == ["r1", "r3"]
    assert result.processing_stats.removed_record_count == 1
    assert [warning.code for warning in result.warnings] == ["dropped_duplicate_record"]


def test_process_document_records_cleans_markdown_html_and_urls() -> None:
    records = [
        _record(
            source_record_id="r1",
            record_index=0,
            source_path="faq/readme.md",
            content=(
                "---\n"
                "title: Demo\n"
                "---\n"
                "[TOC]\n"
                "- [Section](#section)\n"
                "<p>Hello</p>\n"
                "https://example.com\n"
                "## Section\n"
                "Body"
            ),
        )
    ]

    result = process_document_records(
        records,
        processing_rules=[
            "remove_markdown_boilerplate",
            "strip_html_tags",
            "remove_url_lines",
            "trim_whitespace",
        ],
    )

    assert result.records[0].processed_content == "Hello\n## Section\nBody"


def test_preview_returns_original_and_processed_samples_with_hashes() -> None:
    records = [
        _record(source_record_id="r1", record_index=0, content="  Alpha  "),
        _record(source_record_id="r2", record_index=1, content="   "),
    ]

    result = build_preprocess_preview(
        records,
        processing_rules=["trim_whitespace", "drop_empty_records"],
        chunk_size=8,
        chunk_overlap=2,
        sample_size=2,
    )

    assert result.chunk_size == 8
    assert result.chunk_overlap == 2
    assert [sample.content for sample in result.original_samples] == ["  Alpha  ", "   "]
    assert result.processed_samples[0].content == "Alpha"
    assert result.processed_samples[0].applied_rules == ["trim_whitespace", "drop_empty_records"]
    assert result.original_samples[0].content_hash == hashlib.sha256("  Alpha  ".encode("utf-8")).hexdigest()
    assert result.processed_samples[0].content_hash == hashlib.sha256("Alpha".encode("utf-8")).hexdigest()


def test_preview_returns_chunked_processed_samples() -> None:
    records = [
        _record(source_record_id="r1", record_index=0, content="ABCDEFGHIJKL"),
    ]

    result = build_preprocess_preview(
        records,
        chunk_size=5,
        chunk_overlap=1,
        sample_size=4,
    )

    assert [sample.content for sample in result.processed_samples] == ["ABCDE", "EFGHI", "IJKL"]


def test_process_document_records_generates_provenance_hashes_and_preserves_identity() -> None:
    record = _record(source_record_id="r1", record_index=7, content="  Alpha  ")

    result = process_document_records(
        [record],
        processing_rules=["trim_whitespace"],
    )

    processed = result.records[0]
    assert processed.source_path == "faq/source.json"
    assert processed.source_record_id == "r1"
    assert processed.record_index == 7
    assert processed.raw_content == "  Alpha  "
    assert processed.processed_content == "Alpha"
    assert processed.raw_content_hash == hashlib.sha256("  Alpha  ".encode("utf-8")).hexdigest()
    assert processed.processed_content_hash == hashlib.sha256("Alpha".encode("utf-8")).hexdigest()


def test_preview_for_unsupported_type_returns_warning_and_cannot_index() -> None:
    records = [
        _record(
            source_record_id="r1",
            record_index=0,
            content="binary-ish",
            source_path="faq/manual.pdf",
        )
    ]

    result = preview(records, chunk_size=8, chunk_overlap=2)

    assert result.source_type == "pdf"
    assert result.can_index is False
    assert result.processing_stats.raw_record_count == 0
    assert [warning.code for warning in result.warnings] == ["unsupported_source_type"]
    assert result.processed_samples == []


def test_process_document_records_warns_when_all_records_removed() -> None:
    records = [_record(source_record_id="r1", record_index=0, content="   ")]

    result = process([*records], processing_rules=["trim_whitespace", "drop_empty_records"])

    assert result.can_index is False
    assert result.records == []
    assert [warning.code for warning in result.warnings] == [
        "dropped_empty_record",
        "no_records_to_index",
    ]


def test_processor_rejects_unknown_or_unsupported_rules() -> None:
    processor = KnowledgeDocumentProcessor()
    json_records = [_record(source_record_id="r1", record_index=0, content="Alpha")]
    txt_records = [_record(source_record_id="r1", record_index=0, content="Alpha", source_path="faq/a.txt")]

    with pytest.raises(ValueError, match="Unknown processing rule"):
        processor.process(json_records, processing_rules=["missing_rule"])

    with pytest.raises(ValueError, match="not supported for source type 'txt'"):
        processor.process(txt_records, processing_rules=["remove_markdown_boilerplate"])

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

from backend.platform.knowledge.processing.config import DEFAULT_PROCESSING_CHUNK_CONFIG
from backend.platform.knowledge.processing.normalizers import (
    remove_markdown_boilerplate,
    remove_url_lines,
    strip_html_tags,
    trim_whitespace,
)
from backend.platform.knowledge.processing.provenance import (
    UNSUPPORTED_PREVIEW_SOURCE_TYPES,
    build_content_hash,
    normalize_source_type,
    sample_content,
)
from backend.platform.knowledge.processing.rules import get_processing_rule, get_supported_rule_ids, list_processing_rules
from backend.platform.knowledge.processing.schemas import (
    PreprocessPreview,
    ProcessedDocumentRecord,
    ProcessedDocumentResult,
    ProcessingRuleDefinition,
    ProcessingRuleId,
    ProcessingSample,
    ProcessingSourceType,
    ProcessingStats,
    ProcessingWarning,
)

if TYPE_CHECKING:
    from backend.platform.knowledge.documents.schemas import DocumentRecord

DEFAULT_SAMPLE_SIZE = 3


@dataclass(frozen=True, slots=True)
class _PreviewSlidingWindowTextSplitter:
    chunk_size: int
    chunk_overlap: int

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + self.chunk_size])
            start += step
        return chunks


@dataclass
class _WorkingRecord:
    source_record_id: str
    record_index: int
    raw_content: str
    processed_content: str
    record: dict[str, object]
    applied_rules: list[ProcessingRuleId] = field(default_factory=list)
    dropped: bool = False
    warnings: list[ProcessingWarning] = field(default_factory=list)


class KnowledgeDocumentProcessor:
    def __init__(self, *, sample_size: int = DEFAULT_SAMPLE_SIZE) -> None:
        self._sample_size = max(sample_size, 0)

    def preview(
        self,
        records: Sequence[DocumentRecord],
        *,
        source_path: str | None = None,
        namespace: str | None = None,
        processing_rules: Sequence[str] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> PreprocessPreview:
        resolved_chunk_size = _resolve_chunk_size(chunk_size)
        resolved_chunk_overlap = _resolve_chunk_overlap(chunk_overlap)
        result = self.process(
            records,
            source_path=source_path,
            namespace=namespace,
            processing_rules=processing_rules,
        )
        return PreprocessPreview(
            namespace=result.namespace,
            source_path=result.source_path,
            source_type=result.source_type,
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_chunk_overlap,
            supported_rules=list_processing_rules(result.source_type) if result.source_type not in UNSUPPORTED_PREVIEW_SOURCE_TYPES else [],
            selected_rules=[get_processing_rule(rule_id) for rule_id in result.processing_rules],
            processing_stats=result.processing_stats,
            original_samples=self._build_original_samples(records),
            processed_samples=self._build_processed_samples(
                result,
                chunk_size=resolved_chunk_size,
                chunk_overlap=resolved_chunk_overlap,
            ),
            can_index=result.can_index,
            warnings=list(result.warnings),
        )

    def process(
        self,
        records: Sequence[DocumentRecord],
        *,
        source_path: str | None = None,
        namespace: str | None = None,
        processing_rules: Sequence[str] | None = None,
    ) -> ProcessedDocumentResult:
        resolved_source_path = _resolve_source_path(source_path, records)
        resolved_namespace = _resolve_namespace(namespace, records)
        source_type = normalize_source_type(resolved_source_path)
        if source_type is None:
            raise ValueError(f"Unsupported source file type: {resolved_source_path}")

        normalized_rules = _normalize_processing_rules(source_type=source_type, processing_rules=processing_rules or [])
        if source_type in UNSUPPORTED_PREVIEW_SOURCE_TYPES:
            return ProcessedDocumentResult(
                namespace=resolved_namespace,
                source_path=resolved_source_path,
                source_type=source_type,
                processing_rules=list(normalized_rules),
                processing_stats=ProcessingStats(),
                records=[],
                warnings=[
                    ProcessingWarning(
                        code="unsupported_source_type",
                        message=f"Source type '{source_type}' is not supported for processing or indexing yet.",
                    )
                ],
                can_index=False,
            )

        working_records = [
            _WorkingRecord(
                source_record_id=record.source_record_id,
                record_index=record.record_index,
                raw_content=record.content,
                processed_content=record.content,
                record=dict(record.record),
            )
            for record in records
        ]

        for rule_id in normalized_rules:
            _apply_rule(rule_id, working_records)

        processed_records = [
            ProcessedDocumentRecord(
                namespace=resolved_namespace,
                source_path=resolved_source_path,
                source_type=source_type,
                source_record_id=record.source_record_id,
                record_index=record.record_index,
                raw_content=record.raw_content,
                processed_content=record.processed_content,
                applied_rules=list(record.applied_rules),
                raw_content_hash=build_content_hash(record.raw_content),
                processed_content_hash=build_content_hash(record.processed_content),
                dropped=False,
                record=dict(record.record),
                warnings=list(record.warnings),
            )
            for record in working_records
            if not record.dropped
        ]
        warnings = _collect_warnings(working_records)
        can_index = bool(processed_records)
        if not can_index:
            warnings.append(
                ProcessingWarning(
                    code="no_records_to_index",
                    message="No records remain after processing; indexing is disabled.",
                )
            )

        return ProcessedDocumentResult(
            namespace=resolved_namespace,
            source_path=resolved_source_path,
            source_type=source_type,
            processing_rules=list(normalized_rules),
            processing_stats=ProcessingStats(
                raw_record_count=len(records),
                processed_record_count=len(processed_records),
                removed_record_count=len(records) - len(processed_records),
                raw_char_count=sum(len(record.content) for record in records),
                processed_char_count=sum(len(record.processed_content) for record in processed_records),
            ),
            records=processed_records,
            warnings=warnings,
            can_index=can_index,
        )

    def _build_original_samples(self, records: Sequence[DocumentRecord]) -> list[ProcessingSample]:
        return [
            sample_content(
                sample_index=index,
                source_record_id=record.source_record_id,
                record_index=record.record_index,
                content=record.content,
                applied_rules=[],
                dropped=False,
            )
            for index, record in enumerate(records[: self._sample_size])
        ]

    def _build_processed_samples(
        self,
        result: ProcessedDocumentResult,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[ProcessingSample]:
        if result.source_type in UNSUPPORTED_PREVIEW_SOURCE_TYPES:
            return []
        splitter = _PreviewSlidingWindowTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        processed_samples: list[ProcessingSample] = []
        sample_index = 0
        for processed_record in result.records:
            if processed_record.dropped:
                continue
            normalized_content = processed_record.processed_content.strip()
            if not normalized_content:
                continue
            for chunk in splitter.split_text(normalized_content):
                processed_samples.append(
                    sample_content(
                        sample_index=sample_index,
                        source_record_id=processed_record.source_record_id,
                        record_index=processed_record.record_index,
                        content=chunk,
                        applied_rules=list(processed_record.applied_rules),
                        dropped=False,
                    )
                )
                sample_index += 1
                if sample_index >= self._sample_size:
                    return processed_samples
        return processed_samples


def process_document_records(
    records: Sequence[DocumentRecord],
    *,
    source_path: str | None = None,
    namespace: str | None = None,
    processing_rules: Sequence[str] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> ProcessedDocumentResult:
    processor = KnowledgeDocumentProcessor(sample_size=sample_size)
    return processor.process(
        records,
        source_path=source_path,
        namespace=namespace,
        processing_rules=processing_rules,
    )


def build_preprocess_preview(
    records: Sequence[DocumentRecord],
    *,
    source_path: str | None = None,
    namespace: str | None = None,
    processing_rules: Sequence[str] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> PreprocessPreview:
    processor = KnowledgeDocumentProcessor(sample_size=sample_size)
    return processor.preview(
        records,
        source_path=source_path,
        namespace=namespace,
        processing_rules=processing_rules,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def process(
    records: Sequence[DocumentRecord],
    *,
    source_path: str | None = None,
    namespace: str | None = None,
    processing_rules: Sequence[str] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> ProcessedDocumentResult:
    return process_document_records(
        records,
        source_path=source_path,
        namespace=namespace,
        processing_rules=processing_rules,
        sample_size=sample_size,
    )


def preview(
    records: Sequence[DocumentRecord],
    *,
    source_path: str | None = None,
    namespace: str | None = None,
    processing_rules: Sequence[str] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> PreprocessPreview:
    return build_preprocess_preview(
        records,
        source_path=source_path,
        namespace=namespace,
        processing_rules=processing_rules,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        sample_size=sample_size,
    )


def _resolve_chunk_size(chunk_size: int | None) -> int:
    return DEFAULT_PROCESSING_CHUNK_CONFIG.chunk_size if chunk_size is None else chunk_size


def _resolve_chunk_overlap(chunk_overlap: int | None) -> int:
    return DEFAULT_PROCESSING_CHUNK_CONFIG.chunk_overlap if chunk_overlap is None else chunk_overlap


def _resolve_source_path(source_path: str | None, records: Sequence[DocumentRecord]) -> str:
    if source_path:
        return source_path
    if records:
        return records[0].source_path
    raise ValueError("source_path is required when records are empty")


def _resolve_namespace(namespace: str | None, records: Sequence[DocumentRecord]) -> str:
    if namespace:
        return namespace
    if records:
        return records[0].namespace
    raise ValueError("namespace is required when records are empty")


def _normalize_processing_rules(
    *,
    source_type: ProcessingSourceType,
    processing_rules: Sequence[str],
) -> list[ProcessingRuleId]:
    supported_rule_ids = set(get_supported_rule_ids(source_type))
    all_rule_ids = set(get_supported_rule_ids())
    normalized_rules: list[ProcessingRuleId] = []
    for rule_id in processing_rules:
        if rule_id not in all_rule_ids:
            raise ValueError(f"Unknown processing rule: {rule_id}")
        if source_type not in UNSUPPORTED_PREVIEW_SOURCE_TYPES and rule_id not in supported_rule_ids:
            raise ValueError(f"Processing rule '{rule_id}' is not supported for source type '{source_type}'")
        normalized_rules.append(rule_id)
    return normalized_rules


def _apply_rule(rule_id: ProcessingRuleId, records: list[_WorkingRecord]) -> None:
    if rule_id == "trim_whitespace":
        for record in records:
            if record.dropped:
                continue
            record.processed_content = trim_whitespace(record.processed_content)
            record.applied_rules.append(rule_id)
        return

    if rule_id == "strip_html_tags":
        for record in records:
            if record.dropped:
                continue
            record.processed_content = strip_html_tags(record.processed_content)
            record.applied_rules.append(rule_id)
        return

    if rule_id == "remove_url_lines":
        for record in records:
            if record.dropped:
                continue
            record.processed_content = remove_url_lines(record.processed_content)
            record.applied_rules.append(rule_id)
        return

    if rule_id == "remove_markdown_boilerplate":
        for record in records:
            if record.dropped:
                continue
            record.processed_content = remove_markdown_boilerplate(record.processed_content)
            record.applied_rules.append(rule_id)
        return

    if rule_id == "drop_empty_records":
        for record in records:
            if record.dropped:
                continue
            record.applied_rules.append(rule_id)
            if record.processed_content.strip():
                continue
            record.dropped = True
            record.warnings.append(
                ProcessingWarning(
                    code="dropped_empty_record",
                    message="Record was dropped because it is empty after processing.",
                    source_record_id=record.source_record_id,
                    record_index=record.record_index,
                )
            )
        return

    if rule_id == "dedupe_records":
        seen_contents: set[str] = set()
        for record in records:
            if record.dropped:
                continue
            record.applied_rules.append(rule_id)
            if record.processed_content not in seen_contents:
                seen_contents.add(record.processed_content)
                continue
            record.dropped = True
            record.warnings.append(
                ProcessingWarning(
                    code="dropped_duplicate_record",
                    message="Record was dropped because its processed content duplicates an earlier record.",
                    source_record_id=record.source_record_id,
                    record_index=record.record_index,
                )
            )
        return

    raise ValueError(f"Unsupported processing rule: {rule_id}")


def _collect_warnings(records: Iterable[_WorkingRecord]) -> list[ProcessingWarning]:
    warnings: list[ProcessingWarning] = []
    for record in records:
        warnings.extend(record.warnings)
    return warnings

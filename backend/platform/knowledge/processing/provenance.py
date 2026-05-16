from __future__ import annotations

import hashlib
from pathlib import Path

from backend.platform.knowledge.processing.schemas import ProcessingSample, ProcessingSourceType

UNSUPPORTED_PREVIEW_SOURCE_TYPES = {"pdf", "docx", "xlsx"}


def resolve_source_type(source_path: str) -> str:
    return Path(source_path).suffix.lower().lstrip(".")


def build_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sample_content(
    *,
    sample_index: int,
    source_record_id: str,
    record_index: int,
    content: str,
    applied_rules: list[str],
    dropped: bool,
) -> ProcessingSample:
    return ProcessingSample(
        sample_index=sample_index,
        source_record_id=source_record_id,
        record_index=record_index,
        content=content,
        content_hash=build_content_hash(content),
        applied_rules=applied_rules,
        dropped=dropped,
    )


def normalize_source_type(source_path: str) -> ProcessingSourceType | None:
    source_type = resolve_source_type(source_path)
    if source_type in {"json", "csv", "txt", "md", "pdf", "docx", "xlsx"}:
        return source_type
    return None

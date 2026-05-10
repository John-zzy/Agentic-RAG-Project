from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from backend.platform.knowledge.documents.schemas import DocumentChunk, DocumentRecord
from backend.platform.knowledge.documents.validators import validate_chunking


def build_document_chunks(
    records: Iterable[DocumentRecord],
    *,
    document_id: str,
    document_version: int,
    updated_at: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """将文档记录切成带稳定 ID 和追踪元数据的文本块。"""
    validate_chunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[DocumentChunk] = []
    for record in records:
        for content_part in _split_text(record.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunk_index = len(chunks)
            chunk_id = _build_chunk_id(
                document_id=document_id,
                document_version=document_version,
                source_record_id=record.source_record_id,
                chunk_index=chunk_index,
                content=content_part,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    content=content_part,
                    metadata=_build_chunk_metadata(
                        record=record,
                        document_id=document_id,
                        document_version=document_version,
                        updated_at=updated_at,
                        chunk_id=chunk_id,
                        chunk_index=chunk_index,
                    ),
                )
            )
    return chunks


def _split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        chunks.append(normalized[start : start + chunk_size])
        start += step
    return chunks


def _build_chunk_id(
    *,
    document_id: str,
    document_version: int,
    source_record_id: str,
    chunk_index: int,
    content: str,
) -> str:
    payload = f"{document_id}\n{document_version}\n{source_record_id}\n{chunk_index}\n{content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_chunk_metadata(
    *,
    record: DocumentRecord,
    document_id: str,
    document_version: int,
    updated_at: str,
    chunk_id: str,
    chunk_index: int,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "document_version": document_version,
        "namespace": record.namespace,
        "source_type": "json",
        "source_path": record.source_path,
        "source_record_id": record.source_record_id,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "updated_at": updated_at,
    }

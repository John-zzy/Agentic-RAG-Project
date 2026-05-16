from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Iterable
from typing import Any, Protocol

from backend.platform.knowledge.documents.schemas import DocumentChunk
from backend.platform.knowledge.documents.validators import validate_chunking
from backend.platform.knowledge.processing.schemas import ProcessedDocumentRecord

class KnowledgeChunker(Protocol):
    """切块适配边界，便于后续替换为 LangChain splitter。"""

    def split_text(self, text: str) -> list[str]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SlidingWindowTextSplitter:
    """保留当前按字符滑窗切块的默认实现。"""

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


@dataclass(frozen=True, slots=True)
class LangChainTextSplitterAdapter:
    """把现成的 LangChain splitter 包装成平台统一的切块接口。"""

    splitter: Any

    def split_text(self, text: str) -> list[str]:
        return list(self.splitter.split_text(text))


def build_document_chunks(
    records: Iterable[ProcessedDocumentRecord],
    *,
    document_id: str,
    document_version: int,
    updated_at: str,
    chunk_size: int,
    chunk_overlap: int,
    chunker: KnowledgeChunker | None = None,
) -> list[DocumentChunk]:
    """将处理后的标准记录切成带稳定 ID 和追踪元数据的文本块。"""
    validate_chunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    resolved_chunker = chunker or SlidingWindowTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[DocumentChunk] = []
    for record in records:
        normalized_content = record.processed_content.strip()
        if record.dropped or not normalized_content:
            continue
        for content_part in resolved_chunker.split_text(normalized_content):
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
    record: ProcessedDocumentRecord,
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
        "source_type": record.source_type,
        "source_path": record.source_path,
        "source_record_id": record.source_record_id,
        "source_record_index": record.record_index,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "applied_rules": list(record.applied_rules),
        "raw_content_hash": record.raw_content_hash,
        "processed_content_hash": record.processed_content_hash,
        "updated_at": updated_at,
        "is_active": True,
    }

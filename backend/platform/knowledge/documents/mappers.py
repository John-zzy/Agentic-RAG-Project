from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from backend.platform.knowledge.documents.models import (
    KnowledgeDocumentDetail,
    KnowledgeDocumentOperationResult,
    KnowledgeDocumentSummary,
    KnowledgeDocumentVersionSummary,
    KnowledgeFileIndexSummary,
)


class KnowledgeDocumentMapper:
    """把仓储记录和文件信息转换成 API 需要的对象。"""

    def to_summary(self, record: dict[str, object]) -> KnowledgeDocumentSummary:
        return KnowledgeDocumentSummary(
            document_id=str(record["document_id"]),
            namespace=str(record["namespace"]),
            source_path=str(record["source_path"]),
            status=record["status"],  # type: ignore[arg-type]
            active_version=int(record["active_version"]),
            chunk_count=int(record["chunk_count"]),
            updated_at=str(record["updated_at"]),
        )

    def to_detail(self, record: dict[str, object]) -> KnowledgeDocumentDetail:
        versions = [
            KnowledgeDocumentVersionSummary(**version)
            for version in record.get("versions", [])
            if isinstance(version, dict)
        ]
        return KnowledgeDocumentDetail(
            **self.to_summary(record).model_dump(),
            source_type=str(record["source_type"]),
            chunk_size=int(record["chunk_size"]),
            chunk_overlap=int(record["chunk_overlap"]),
            last_error=str(record["last_error"]) if record.get("last_error") else None,
            versions=versions,
        )

    def to_operation_result(
        self,
        record: dict[str, object],
        document_version: int,
    ) -> KnowledgeDocumentOperationResult:
        return KnowledgeDocumentOperationResult(
            **self.to_detail(record).model_dump(),
            document_version=document_version,
        )

    def to_unindexed_file_summary(
        self,
        *,
        file_path: Path,
        files_root: Path,
        can_index: bool,
    ) -> KnowledgeFileIndexSummary:
        stat = file_path.stat()
        return KnowledgeFileIndexSummary(
            filename=file_path.name,
            source_path=file_path.relative_to(files_root).as_posix(),
            file_size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime, UTC).isoformat(),
            indexed=False,
            status="unindexed" if can_index else "unsupported",
            last_error=None if can_index else "当前后端仅支持对 JSON、TXT、MD、CSV 文件构建索引。",
            can_index=can_index,
        )

    def to_indexed_file_summary(
        self,
        *,
        file_path: Path,
        files_root: Path,
        record: dict[str, object],
        can_index: bool,
    ) -> KnowledgeFileIndexSummary:
        stat = file_path.stat()
        return KnowledgeFileIndexSummary(
            filename=file_path.name,
            source_path=file_path.relative_to(files_root).as_posix(),
            file_size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime, UTC).isoformat(),
            namespace=str(record["namespace"]),
            document_id=str(record["document_id"]),
            indexed=True,
            status=str(record["status"]),
            active_version=int(record["active_version"]),
            chunk_count=int(record["chunk_count"]),
            updated_at=str(record["updated_at"]),
            last_error=str(record["last_error"]) if record.get("last_error") else None,
            can_index=can_index,
        )

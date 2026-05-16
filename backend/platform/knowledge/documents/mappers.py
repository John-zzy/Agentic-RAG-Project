from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.platform.knowledge.documents.models import (
    KnowledgeDocumentDetail,
    KnowledgeDocumentOperationResult,
    KnowledgeDocumentSummary,
    KnowledgeDocumentVersionSummary,
    KnowledgeFileIndexSummary,
)
from backend.platform.knowledge.processing.schemas import ProcessingStats


class KnowledgeDocumentMapper:
    """把仓储记录和文件信息转换成 API 需要的对象。"""

    def _resolve_source_type(self, record: dict[str, object]) -> str:
        source_type = record.get("source_type")
        if isinstance(source_type, str) and source_type:
            return source_type

        source_path = record.get("source_path")
        if isinstance(source_path, str):
            suffix = Path(source_path).suffix.lower().lstrip(".")
            if suffix:
                return suffix
        return "json"

    def resolve_processing_rules(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item is not None]

    def _resolve_processing_stats(
        self,
        value: object,
    ) -> ProcessingStats | None:
        if isinstance(value, ProcessingStats):
            return value
        if isinstance(value, dict):
            stats_payload: dict[str, Any] = {}
            for key in (
                "raw_record_count",
                "processed_record_count",
                "removed_record_count",
                "raw_char_count",
                "processed_char_count",
            ):
                stats_value = value.get(key)
                if stats_value is None:
                    stats_payload[key] = None
                else:
                    stats_payload[key] = int(stats_value)
            if "removed_record_count" not in stats_payload:
                stats_payload["removed_record_count"] = 0
            return ProcessingStats(**stats_payload)
        return None

    def _resolve_provenance_enabled(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    def to_summary(self, record: dict[str, object]) -> KnowledgeDocumentSummary:
        return KnowledgeDocumentSummary(
            document_id=str(record["document_id"]),
            namespace=str(record["namespace"]),
            source_path=str(record["source_path"]),
            status=record["status"],  # type: ignore[arg-type]
            source_type=self._resolve_source_type(record),
            processing_rules=self.resolve_processing_rules(record.get("processing_rules")),
            processing_stats=self._resolve_processing_stats(record.get("processing_stats")),
            provenance_enabled=self._resolve_provenance_enabled(record.get("provenance_enabled")),
            active_version=int(record["active_version"]),
            chunk_count=int(record["chunk_count"]),
            updated_at=str(record["updated_at"]),
        )

    def to_detail(self, record: dict[str, object]) -> KnowledgeDocumentDetail:
        versions = [
            KnowledgeDocumentVersionSummary(
                **{
                    **version,
                    "source_type": self._resolve_source_type(version),
                    "processing_rules": self.resolve_processing_rules(version.get("processing_rules")),
                    "processing_stats": self._resolve_processing_stats(version.get("processing_stats")),
                    "provenance_enabled": self._resolve_provenance_enabled(version.get("provenance_enabled")),
                },
            )
            for version in record.get("versions", [])
            if isinstance(version, dict)
        ]
        return KnowledgeDocumentDetail(
            **self.to_summary(record).model_dump(),
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
        warning_message: str | None = None,
    ) -> KnowledgeFileIndexSummary:
        stat = file_path.stat()
        return KnowledgeFileIndexSummary(
            filename=file_path.name,
            source_path=file_path.relative_to(files_root).as_posix(),
            file_size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime, UTC).isoformat(),
            indexed=False,
            status="awaiting_processing" if can_index else "unsupported",
            last_error=warning_message if warning_message is not None else (
                None if can_index else "当前文件类型尚未接入预处理与索引链路。"
            ),
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

from __future__ import annotations

from datetime import UTC, datetime

from backend.platform.knowledge.base.store import VectorStoreDocument
from backend.platform.knowledge.documents.chunker import build_document_chunks
from backend.platform.knowledge.documents.loader import build_document_id
from backend.platform.knowledge.documents.models import DocumentStatus, KnowledgeDocumentStoreError
from backend.platform.knowledge.documents.schemas import DocumentRecord
from backend.platform.knowledge.documents.store_support import KnowledgeDocumentRepositoryGateway
from backend.platform.knowledge.processing import process_document_records


class KnowledgeDocumentPublisher:
    """负责把一个新版本写入仓储，并在失败时把旧状态尽量恢复回来。"""

    def __init__(self, repository_gateway: KnowledgeDocumentRepositoryGateway) -> None:
        self.repository_gateway = repository_gateway

    def publish_new_version(
        self,
        *,
        namespace: str,
        records: list[DocumentRecord],
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool,
        existing_record: dict[str, object] | None,
        processing_rules: list[str],
    ) -> tuple[dict[str, object], int]:
        normalized_source_path = records[0].source_path
        document_id = build_document_id(namespace=namespace, source_path=normalized_source_path)
        current_record = existing_record or self.repository_gateway.get_document_record(document_id)
        document_version = self._next_version(current_record)
        updated_at = self._now()
        processed_result = process_document_records(
            namespace=namespace,
            source_path=normalized_source_path,
            records=records,
            processing_rules=processing_rules,
        )
        if not processed_result.can_index:
            raise ValueError("No records remain after processing; indexing is disabled.")
        chunks = build_document_chunks(
            records=processed_result.records,
            document_id=document_id,
            document_version=document_version,
            updated_at=updated_at,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        version = self._build_version(
            document_version=document_version,
            status="active",
            chunk_count=len(chunks),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            created_at=updated_at,
            source_type=processed_result.source_type,
            processing_rules=processed_result.processing_rules,
            processing_stats=processed_result.processing_stats.model_dump(),
            provenance_enabled=True,
        )
        record = self._build_record(
            current_record=current_record,
            document_id=document_id,
            namespace=records[0].namespace,
            source_path=normalized_source_path,
            active_version=document_version,
            version=version,
            keep_version=keep_version,
            updated_at=updated_at,
            source_type=processed_result.source_type,
            processing_rules=processed_result.processing_rules,
            processing_stats=processed_result.processing_stats.model_dump(),
            provenance_enabled=True,
        )
        vector_chunks = [
            VectorStoreDocument(
                id=chunk.chunk_id,
                content=chunk.content,
                metadata={**chunk.metadata, "is_active": True},
            )
            for chunk in chunks
        ]
        new_chunk_ids = [chunk.id for chunk in vector_chunks]

        try:
            # 第一步先把新分块写进去，这样后面主记录切换时已经有可用内容。
            self.repository_gateway.upsert_document_chunks(vector_chunks)
            # 第二步把主记录切到新版本，列表和详情会先看到最新版本号。
            self.repository_gateway.upsert_document_record(record)
            # 第三步再把旧版本分块停用，避免在写新分块之前把线上数据提前下线。
            if current_record is not None:
                self.repository_gateway.deactivate_document_chunks(
                    document_id,
                    int(current_record["active_version"]),
                )
        except Exception as exc:
            if current_record is not None:
                # 这里先把旧主记录写回去，再把旧分块重新激活，尽量恢复成切换前的状态。
                self._restore_current_record(current_record)
                self._restore_current_chunks(current_record)
            else:
                # 这是第一次建索引时失败的情况，要补一条 failed 记录，方便页面显示错误原因。
                failed_record = self._mark_failed(record, document_version, str(exc))
                try:
                    self.repository_gateway.upsert_document_record(failed_record)
                except KnowledgeDocumentStoreError:
                    pass
            # 最后把这次刚写进去的新分块删掉，避免留下半成品数据。
            self._cleanup_new_chunks(new_chunk_ids)
            raise KnowledgeDocumentStoreError(f"Failed to switch document '{document_id}': {exc}") from exc

        return record, document_version

    def _next_version(self, record: dict[str, object] | None) -> int:
        if record is None:
            return 1
        versions = record.get("versions", [])
        if not isinstance(versions, list) or not versions:
            return int(record.get("active_version", 0)) + 1
        return max(int(version["document_version"]) for version in versions) + 1

    def _build_version(
        self,
        *,
        document_version: int,
        status: DocumentStatus,
        chunk_count: int,
        chunk_size: int,
        chunk_overlap: int,
        created_at: str,
        source_type: str,
        processing_rules: list[str],
        processing_stats: dict[str, object],
        provenance_enabled: bool,
        last_error: str | None = None,
    ) -> dict[str, object]:
        return {
            "document_version": document_version,
            "status": status,
            "chunk_count": chunk_count,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "created_at": created_at,
            "source_type": source_type,
            "processing_rules": processing_rules,
            "processing_stats": processing_stats,
            "provenance_enabled": provenance_enabled,
            "last_error": last_error,
        }

    def _build_record(
        self,
        *,
        current_record: dict[str, object] | None,
        document_id: str,
        namespace: str,
        source_path: str,
        active_version: int,
        version: dict[str, object],
        keep_version: bool,
        updated_at: str,
        source_type: str,
        processing_rules: list[str],
        processing_stats: dict[str, object],
        provenance_enabled: bool,
    ) -> dict[str, object]:
        existing_versions = current_record.get("versions", []) if current_record else []
        versions = list(existing_versions) if keep_version and isinstance(existing_versions, list) else []
        versions.append(version)
        created_at = str(current_record.get("created_at")) if current_record else updated_at
        return {
            "document_id": document_id,
            "namespace": namespace,
            "source_type": source_type,
            "source_path": source_path,
            "status": "active",
            "processing_rules": processing_rules,
            "processing_stats": processing_stats,
            "provenance_enabled": provenance_enabled,
            "active_version": active_version,
            "chunk_count": int(version["chunk_count"]),
            "chunk_size": int(version["chunk_size"]),
            "chunk_overlap": int(version["chunk_overlap"]),
            "created_at": created_at,
            "updated_at": updated_at,
            "last_error": None,
            "versions": versions,
        }

    def _mark_failed(
        self,
        record: dict[str, object],
        document_version: int,
        last_error: str,
    ) -> dict[str, object]:
        failed = dict(record)
        failed["status"] = "failed"
        failed["last_error"] = last_error
        versions = []
        for version in failed.get("versions", []):
            version_copy = dict(version)
            if int(version_copy["document_version"]) == document_version:
                version_copy["status"] = "failed"
                version_copy["last_error"] = last_error
            versions.append(version_copy)
        failed["versions"] = versions
        return failed

    def _cleanup_new_chunks(self, chunk_ids: list[str]) -> None:
        try:
            self.repository_gateway.delete_document_chunks(chunk_ids)
        except Exception:
            return None

    def _restore_current_record(self, current_record: dict[str, object]) -> None:
        try:
            self.repository_gateway.upsert_document_record(current_record)
        except KnowledgeDocumentStoreError:
            return None

    def _restore_current_chunks(self, current_record: dict[str, object]) -> None:
        try:
            self.repository_gateway.activate_document_chunks(
                str(current_record["document_id"]),
                int(current_record["active_version"]),
            )
        except KnowledgeDocumentStoreError:
            return None

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

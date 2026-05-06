from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from backend.config.settings import AppSettings, settings
from backend.knowledge.base.store import VectorStore, VectorStoreDocument, VectorStoreFactory
from backend.knowledge.documents.chunker import build_document_chunks
from backend.knowledge.documents.loader import build_document_id, load_document_records
from backend.knowledge.documents.validators import validate_chunking, validate_namespace


DocumentStatus = Literal["active", "failed", "deleted"]


class KnowledgeDocumentError(RuntimeError):
    """文档管理服务的基础异常，便于 API 层统一转换响应。"""


class KnowledgeDocumentNotFoundError(KnowledgeDocumentError):
    """表示文档不存在或已删除。"""


class KnowledgeDocumentStoreError(KnowledgeDocumentError):
    """表示写入或更新向量存储失败。"""


class KnowledgeDocumentVersionSummary(BaseModel):
    """描述单个文档版本的分块参数和状态。"""

    document_version: int
    status: DocumentStatus
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    created_at: str
    last_error: str | None = None


class KnowledgeDocumentSummary(BaseModel):
    """列表页所需的文档最小摘要。"""

    document_id: str
    namespace: str
    source_path: str
    status: DocumentStatus
    active_version: int
    chunk_count: int
    updated_at: str


class KnowledgeDocumentDetail(KnowledgeDocumentSummary):
    """详情页所需的主记录、当前分块参数和版本摘要。"""

    source_type: str
    chunk_size: int
    chunk_overlap: int
    last_error: str | None = None
    versions: list[KnowledgeDocumentVersionSummary]


class KnowledgeDocumentOperationResult(KnowledgeDocumentDetail):
    """注册、删除和重新分块操作的返回结构。"""

    document_version: int


class KnowledgeDocumentService:
    """编排知识文档注册、查询、删除和重新分块生命周期。"""

    def __init__(
        self,
        app_settings: AppSettings | None = None,
        store: VectorStore | None = None,
        files_root: str | Path | None = None,
    ) -> None:
        """初始化服务依赖；测试可注入 files_root 和 store。"""
        self.app_settings = app_settings or settings
        self.store = store or VectorStoreFactory.create(self.app_settings)
        self.files_root = Path(files_root) if files_root is not None else self.app_settings.data_dir / "files"
        self._call_store("ensure document indexes", self.store.ensure_document_indexes)

    def register_document(
        self,
        namespace: str,
        source_path: str,
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool = False,
    ) -> KnowledgeDocumentOperationResult:
        """加载 JSON 源文件并写入文档记录与当前活跃分块。"""
        return self._write_new_version(
            namespace=namespace,
            source_path=source_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_version=keep_version,
            existing_record=None,
        )

    def list_documents(self, namespace: str | None = None) -> list[KnowledgeDocumentSummary]:
        """列出未删除文档，可按命名空间过滤。"""
        if namespace is not None:
            validate_namespace(namespace)
        records = self._call_store("list document records", self.store.list_document_records, namespace)
        return [self._to_summary(record) for record in records]

    def get_document(self, document_id: str) -> KnowledgeDocumentDetail:
        """读取文档详情；缺失或已删除时抛出服务级 404 异常。"""
        record = self._get_existing_record(document_id)
        return self._to_detail(record)

    def delete_document(self, document_id: str) -> KnowledgeDocumentOperationResult:
        """软删除文档记录并停用该文档所有活跃分块，不删除源 JSON 文件。"""
        record = self._get_existing_record(document_id)
        self._call_store("deactivate document chunks", self.store.deactivate_document_chunks, document_id)
        self._call_store("delete document record", self.store.delete_document_record, document_id)
        record["status"] = "deleted"
        return self._to_operation_result(record, int(record["active_version"]))

    def rechunk_document(
        self,
        document_id: str,
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool = False,
    ) -> KnowledgeDocumentOperationResult:
        """基于原始 JSON 文件重新生成分块并切换活跃版本。"""
        record = self._get_existing_record(document_id)
        return self._write_new_version(
            namespace=str(record["namespace"]),
            source_path=str(record["source_path"]),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_version=keep_version,
            existing_record=record,
        )

    def _write_new_version(
        self,
        *,
        namespace: str,
        source_path: str,
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool,
        existing_record: dict[str, object] | None,
    ) -> KnowledgeDocumentOperationResult:
        validate_chunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        records = load_document_records(namespace=namespace, source_path=source_path, data_root=self.files_root)
        normalized_source_path = records[0].source_path
        document_id = build_document_id(namespace=namespace, source_path=normalized_source_path)
        current_record = existing_record or self._call_store(
            "get document record",
            self.store.get_document_record,
            document_id,
        )
        document_version = self._next_version(current_record)
        updated_at = self._now()
        chunks = build_document_chunks(
            records=records,
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
            self._call_store("upsert document chunks", self.store.upsert_document_chunks, vector_chunks)
            self._call_store("upsert document record", self.store.upsert_document_record, record)
            if current_record is not None:
                self._call_store(
                    "deactivate document chunks",
                    self.store.deactivate_document_chunks,
                    document_id,
                    int(current_record["active_version"]),
                )
        except Exception as exc:
            self._cleanup_new_chunks(new_chunk_ids)
            if current_record is None:
                failed_record = self._mark_failed(record, document_version, str(exc))
                try:
                    self._call_store("upsert failed document record", self.store.upsert_document_record, failed_record)
                except KnowledgeDocumentStoreError:
                    pass
            raise KnowledgeDocumentStoreError(f"Failed to switch document '{document_id}': {exc}") from exc

        return self._to_operation_result(record, document_version)

    def _get_existing_record(self, document_id: str) -> dict[str, object]:
        record = self._call_store("get document record", self.store.get_document_record, document_id)
        if record is None:
            raise KnowledgeDocumentNotFoundError(f"Knowledge document not found: {document_id}")
        return record

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
        last_error: str | None = None,
    ) -> dict[str, object]:
        return {
            "document_version": document_version,
            "status": status,
            "chunk_count": chunk_count,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "created_at": created_at,
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
    ) -> dict[str, object]:
        existing_versions = current_record.get("versions", []) if current_record else []
        versions = list(existing_versions) if keep_version and isinstance(existing_versions, list) else []
        versions.append(version)
        created_at = str(current_record.get("created_at")) if current_record else updated_at
        return {
            "document_id": document_id,
            "namespace": namespace,
            "source_type": "json",
            "source_path": source_path,
            "status": "active",
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

    def _to_summary(self, record: dict[str, object]) -> KnowledgeDocumentSummary:
        return KnowledgeDocumentSummary(
            document_id=str(record["document_id"]),
            namespace=str(record["namespace"]),
            source_path=str(record["source_path"]),
            status=record["status"],  # type: ignore[arg-type]
            active_version=int(record["active_version"]),
            chunk_count=int(record["chunk_count"]),
            updated_at=str(record["updated_at"]),
        )

    def _to_detail(self, record: dict[str, object]) -> KnowledgeDocumentDetail:
        versions = [
            KnowledgeDocumentVersionSummary(**version)
            for version in record.get("versions", [])
            if isinstance(version, dict)
        ]
        return KnowledgeDocumentDetail(
            **self._to_summary(record).model_dump(),
            source_type=str(record["source_type"]),
            chunk_size=int(record["chunk_size"]),
            chunk_overlap=int(record["chunk_overlap"]),
            last_error=str(record["last_error"]) if record.get("last_error") else None,
            versions=versions,
        )

    def _to_operation_result(
        self,
        record: dict[str, object],
        document_version: int,
    ) -> KnowledgeDocumentOperationResult:
        return KnowledgeDocumentOperationResult(
            **self._to_detail(record).model_dump(),
            document_version=document_version,
        )

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def _cleanup_new_chunks(self, chunk_ids: list[str]) -> None:
        """失败回滚时尽力清理新分块，不影响主异常返回。"""
        try:
            self._call_store("delete document chunks", self.store.delete_document_chunks, chunk_ids)
        except KnowledgeDocumentStoreError:
            return None

    def _call_store(self, operation: str, method: object, *args: object) -> object:
        """将后端不支持能力转换为服务级错误，避免泄露原始实现异常。"""
        try:
            return method(*args)  # type: ignore[operator]
        except NotImplementedError as exc:
            raise KnowledgeDocumentStoreError(
                f"Vector store '{self.app_settings.vector_store.provider}' does not support document management: {operation}."
            ) from exc

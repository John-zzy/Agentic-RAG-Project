from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from backend.platform.config.settings import AppSettings, FILES_DIR, settings
from backend.platform.knowledge.base.store import (
    VectorStore,
    VectorStoreDocument,
    VectorStoreFactory,
)
from backend.platform.knowledge.documents.chunker import build_document_chunks
from backend.platform.knowledge.documents.loader import build_document_id, load_document_records
from backend.platform.knowledge.documents.validators import validate_chunking, validate_namespace


DocumentStatus = Literal["active", "failed", "deleted"]
MANAGED_FILE_EXTENSIONS = {".json", ".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx"}
INDEXABLE_FILE_EXTENSIONS = {".json", ".txt", ".md", ".csv"}


class KnowledgeDocumentError(RuntimeError):
    """文档管理服务基础异常。"""


class KnowledgeDocumentNotFoundError(KnowledgeDocumentError):
    """文档不存在或已删除。"""


class KnowledgeDocumentStoreError(KnowledgeDocumentError):
    """向量库读写失败。"""


class KnowledgeDocumentVersionSummary(BaseModel):
    """单个文档版本摘要。"""

    document_version: int
    status: DocumentStatus
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    created_at: str
    last_error: str | None = None


class KnowledgeDocumentSummary(BaseModel):
    """文档列表摘要。"""

    document_id: str
    namespace: str
    source_path: str
    status: DocumentStatus
    active_version: int
    chunk_count: int
    updated_at: str


class KnowledgeDocumentDetail(KnowledgeDocumentSummary):
    """文档详情。"""

    source_type: str
    chunk_size: int
    chunk_overlap: int
    last_error: str | None = None
    versions: list[KnowledgeDocumentVersionSummary]


class KnowledgeDocumentOperationResult(KnowledgeDocumentDetail):
    """文档写操作结果。"""

    document_version: int


class KnowledgeFileIndexSummary(BaseModel):
    """按上传文件聚合的索引状态摘要。"""

    filename: str
    source_path: str
    file_size: int | None = None
    created_at: str | None = None
    namespace: str | None = None
    document_id: str | None = None
    indexed: bool
    status: str
    active_version: int | None = None
    chunk_count: int | None = None
    updated_at: str | None = None
    last_error: str | None = None
    can_index: bool = True


class KnowledgeDocumentService:
    """编排知识文档注册、查询、删除与重建。"""

    def __init__(
        self,
        app_settings: AppSettings | None = None,
        store: VectorStore | None = None,
        files_root: str | Path | None = None,
    ) -> None:
        """初始化依赖；默认从上传文件目录读取源文件。"""
        self.app_settings = app_settings or settings
        self.store = store or VectorStoreFactory.create(self.app_settings)
        self.files_root = Path(files_root) if files_root is not None else FILES_DIR
        self._call_store("ensure document indexes", self.store.ensure_document_indexes)

    def register_document(
        self,
        namespace: str,
        source_path: str,
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool = False,
    ) -> KnowledgeDocumentOperationResult:
        """加载源文件并写入新版本索引。"""
        return self._write_new_version(
            namespace=namespace,
            source_path=source_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_version=keep_version,
            existing_record=None,
        )

    def list_documents(self, namespace: str | None = None) -> list[KnowledgeDocumentSummary]:
        """列出未删除文档。"""
        if namespace is not None:
            validate_namespace(namespace)
        records = self._call_store("list document records", self.store.list_document_records, namespace)
        return [self._to_summary(record) for record in records]

    def list_file_indexes(self, namespace: str | None = None) -> list[KnowledgeFileIndexSummary]:
        """按文件聚合当前索引状态。"""
        if namespace is not None:
            validate_namespace(namespace)

        records = self._call_store("list document records", self.store.list_document_records, namespace)
        document_by_path = {
            str(record["source_path"]): record
            for record in records
            if isinstance(record, dict)
        }

        summaries: list[KnowledgeFileIndexSummary] = []
        if not self.files_root.exists():
            return summaries

        file_paths = sorted(
            (
                path
                for path in self.files_root.iterdir()
                if path.is_file() and path.suffix.lower() in MANAGED_FILE_EXTENSIONS
            ),
            key=lambda path: path.stat().st_ctime,
            reverse=True,
        )
        for file_path in file_paths:
            relative_path = file_path.relative_to(self.files_root).as_posix()
            record = document_by_path.get(relative_path)
            stat = file_path.stat()
            can_index = file_path.suffix.lower() in INDEXABLE_FILE_EXTENSIONS

            if record is None:
                summaries.append(
                    KnowledgeFileIndexSummary(
                        filename=file_path.name,
                        source_path=relative_path,
                        file_size=stat.st_size,
                        created_at=datetime.fromtimestamp(stat.st_ctime, UTC).isoformat(),
                        indexed=False,
                        status="unindexed" if can_index else "unsupported",
                        last_error=None if can_index else "当前后端仅支持对 JSON、TXT、MD、CSV 文件构建索引。",
                        can_index=can_index,
                    )
                )
                continue

            summaries.append(
                KnowledgeFileIndexSummary(
                    filename=file_path.name,
                    source_path=relative_path,
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
            )
        return summaries

    def get_document(self, document_id: str) -> KnowledgeDocumentDetail:
        """读取文档详情。"""
        record = self._get_existing_record(document_id)
        return self._to_detail(record)

    def delete_document(self, document_id: str) -> KnowledgeDocumentOperationResult:
        """软删除文档记录并停用对应分块。"""
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
        """基于源文件重建分块并切换版本。"""
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
        records = self._load_source_records(namespace=namespace, source_path=source_path)
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
            if current_record is not None:
                self._restore_current_record(current_record)
                self._restore_current_chunks(current_record)
            else:
                failed_record = self._mark_failed(record, document_version, str(exc))
                try:
                    self._call_store("upsert failed document record", self.store.upsert_document_record, failed_record)
                except KnowledgeDocumentStoreError:
                    pass
            self._cleanup_new_chunks(new_chunk_ids)
            raise KnowledgeDocumentStoreError(f"Failed to switch document '{document_id}': {exc}") from exc

        return self._to_operation_result(record, document_version)

    def _get_existing_record(self, document_id: str) -> dict[str, object]:
        record = self._call_store("get document record", self.store.get_document_record, document_id)
        if record is None:
            raise KnowledgeDocumentNotFoundError(f"Knowledge document not found: {document_id}")
        return record

    def _load_source_records(self, *, namespace: str, source_path: str) -> list[object]:
        candidate_roots = [self.files_root]
        fallback_root = self.app_settings.data_dir
        if fallback_root not in candidate_roots:
            candidate_roots.append(fallback_root)

        last_error: Exception | None = None
        for root in candidate_roots:
            try:
                return load_document_records(namespace=namespace, source_path=source_path, data_root=root)
            except FileNotFoundError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(f"source_path does not exist: {source_path}")

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
        """失败回滚时尽力清理新分块。"""
        try:
            self._call_store("delete document chunks", self.store.delete_document_chunks, chunk_ids)
        except Exception:
            return None

    def _restore_current_record(self, current_record: dict[str, object]) -> None:
        """切换失败时尽力恢复旧记录。"""
        try:
            self._call_store("restore document record", self.store.upsert_document_record, current_record)
        except KnowledgeDocumentStoreError:
            return None

    def _restore_current_chunks(self, current_record: dict[str, object]) -> None:
        """切换失败时尽力恢复旧分块激活状态。"""
        try:
            self._call_store(
                "restore document chunks",
                self.store.activate_document_chunks,
                str(current_record["document_id"]),
                int(current_record["active_version"]),
            )
        except KnowledgeDocumentStoreError:
            return None

    def _call_store(self, operation: str, method: object, *args: object) -> object:
        """将后端能力异常转换为服务层错误。"""
        try:
            return method(*args)  # type: ignore[operator]
        except NotImplementedError as exc:
            raise KnowledgeDocumentStoreError(
                f"Vector store '{self.app_settings.vector_store.provider}' does not support document management: {operation}."
            ) from exc

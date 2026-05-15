from __future__ import annotations

from pathlib import Path

from backend.platform.config.settings import AppSettings, FILES_DIR, settings
from backend.platform.knowledge.base.store import KnowledgeDocumentRepository, VectorStoreFactory
from backend.platform.knowledge.documents.loader import load_document_records
from backend.platform.knowledge.documents.mappers import KnowledgeDocumentMapper
from backend.platform.knowledge.documents.models import (
    KnowledgeDocumentNotFoundError,
    KnowledgeDocumentOperationResult,
)
from backend.platform.knowledge.documents.publisher import KnowledgeDocumentPublisher
from backend.platform.knowledge.documents.store_support import KnowledgeDocumentRepositoryGateway
from backend.platform.knowledge.documents.validators import validate_chunking


class KnowledgeDocumentApplicationService:
    """负责写路径编排：注册、删除和重切块。"""

    def __init__(
        self,
        app_settings: AppSettings | None = None,
        repository: KnowledgeDocumentRepository | None = None,
        files_root: str | Path | None = None,
        mapper: KnowledgeDocumentMapper | None = None,
        publisher: KnowledgeDocumentPublisher | None = None,
    ) -> None:
        self.app_settings = app_settings or settings
        self.files_root = Path(files_root) if files_root is not None else FILES_DIR
        resolved_repository = repository or VectorStoreFactory.create_document_repository(self.app_settings)
        self.repository_gateway = KnowledgeDocumentRepositoryGateway(self.app_settings, resolved_repository)
        self.repository_gateway.ensure_document_indexes()
        self.mapper = mapper or KnowledgeDocumentMapper()
        self.publisher = publisher or KnowledgeDocumentPublisher(self.repository_gateway)

    def register_document(
        self,
        namespace: str,
        source_path: str,
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool = False,
    ) -> KnowledgeDocumentOperationResult:
        validate_chunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        records = self._load_source_records(namespace=namespace, source_path=source_path)
        record, document_version = self.publisher.publish_new_version(
            namespace=namespace,
            records=records,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_version=keep_version,
            existing_record=None,
        )
        return self.mapper.to_operation_result(record, document_version)

    def delete_document(self, document_id: str) -> KnowledgeDocumentOperationResult:
        record = self._get_existing_record(document_id)
        self.repository_gateway.deactivate_document_chunks(document_id)
        self.repository_gateway.delete_document_record(document_id)
        deleted_record = dict(record)
        deleted_record["status"] = "deleted"
        return self.mapper.to_operation_result(deleted_record, int(record["active_version"]))

    def rechunk_document(
        self,
        document_id: str,
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool = False,
    ) -> KnowledgeDocumentOperationResult:
        validate_chunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        record = self._get_existing_record(document_id)
        records = self._load_source_records(
            namespace=str(record["namespace"]),
            source_path=str(record["source_path"]),
        )
        next_record, document_version = self.publisher.publish_new_version(
            namespace=str(record["namespace"]),
            records=records,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_version=keep_version,
            existing_record=record,
        )
        return self.mapper.to_operation_result(next_record, document_version)

    def _get_existing_record(self, document_id: str) -> dict[str, object]:
        record = self.repository_gateway.get_document_record(document_id)
        if record is None:
            raise KnowledgeDocumentNotFoundError(f"Knowledge document not found: {document_id}")
        return record

    def _load_source_records(self, *, namespace: str, source_path: str) -> list[object]:
        """这里会按顺序尝试两个根目录，先找上传目录，再找 data_dir，避免老数据直接失效。"""
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

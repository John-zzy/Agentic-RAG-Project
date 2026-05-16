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
from backend.platform.knowledge.documents.validators import validate_chunking, validate_source_path
from backend.platform.knowledge.processing import DEFAULT_PROCESSING_CHUNK_CONFIG, build_preprocess_preview
from backend.platform.knowledge.processing.provenance import normalize_source_type
from backend.platform.knowledge.processing.schemas import PreprocessPreview


class KnowledgeDocumentApplicationService:
    """负责知识文档写路径与预处理预览编排。"""

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
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        keep_version: bool = False,
        processing_rules: list[str] | None = None,
    ) -> KnowledgeDocumentOperationResult:
        chunk_size, chunk_overlap = self._resolve_chunking(chunk_size, chunk_overlap)
        validate_chunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        normalized_source_path, resolved_root = self._resolve_source_location(source_path)
        records = self._load_source_records(
            namespace=namespace,
            source_path=normalized_source_path,
            data_root=resolved_root,
        )
        record, document_version = self.publisher.publish_new_version(
            namespace=namespace,
            records=records,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_version=keep_version,
            existing_record=None,
            processing_rules=processing_rules or [],
        )
        return self.mapper.to_operation_result(record, document_version)

    def preprocess_preview(
        self,
        namespace: str,
        source_path: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        processing_rules: list[str] | None = None,
    ) -> PreprocessPreview:
        chunk_size, chunk_overlap = self._resolve_chunking(chunk_size, chunk_overlap)
        validate_chunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        normalized_source_path, resolved_root = self._resolve_source_location(source_path)
        source_type = normalize_source_type(normalized_source_path)
        if source_type is None:
            raise ValueError(f"Unsupported source file type: {normalized_source_path}")
        if source_type in {"pdf", "docx", "xlsx"}:
            return build_preprocess_preview(
                [],
                source_path=normalized_source_path,
                namespace=namespace,
                processing_rules=processing_rules or [],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        records = self._load_source_records(
            namespace=namespace,
            source_path=normalized_source_path,
            data_root=resolved_root,
        )
        return build_preprocess_preview(
            records,
            source_path=normalized_source_path,
            namespace=namespace,
            processing_rules=processing_rules or [],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

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
        normalized_source_path, resolved_root = self._resolve_source_location(str(record["source_path"]))
        records = self._load_source_records(
            namespace=str(record["namespace"]),
            source_path=normalized_source_path,
            data_root=resolved_root,
        )
        next_record, document_version = self.publisher.publish_new_version(
            namespace=str(record["namespace"]),
            records=records,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_version=keep_version,
            existing_record=record,
            processing_rules=self.mapper.resolve_processing_rules(record.get("processing_rules")),
        )
        return self.mapper.to_operation_result(next_record, document_version)

    def reprocess_document(
        self,
        document_id: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        keep_version: bool = False,
        processing_rules: list[str] | None = None,
    ) -> KnowledgeDocumentOperationResult:
        chunk_size, chunk_overlap = self._resolve_chunking(chunk_size, chunk_overlap)
        validate_chunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        record = self._get_existing_record(document_id)
        normalized_source_path, resolved_root = self._resolve_source_location(str(record["source_path"]))
        records = self._load_source_records(
            namespace=str(record["namespace"]),
            source_path=normalized_source_path,
            data_root=resolved_root,
        )
        next_record, document_version = self.publisher.publish_new_version(
            namespace=str(record["namespace"]),
            records=records,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_version=keep_version,
            existing_record=record,
            processing_rules=processing_rules or [],
        )
        return self.mapper.to_operation_result(next_record, document_version)

    def _get_existing_record(self, document_id: str) -> dict[str, object]:
        record = self.repository_gateway.get_document_record(document_id)
        if record is None:
            raise KnowledgeDocumentNotFoundError(f"Knowledge document not found: {document_id}")
        return record

    def _resolve_chunking(self, chunk_size: int | None, chunk_overlap: int | None) -> tuple[int, int]:
        return (
            DEFAULT_PROCESSING_CHUNK_CONFIG.chunk_size if chunk_size is None else chunk_size,
            DEFAULT_PROCESSING_CHUNK_CONFIG.chunk_overlap if chunk_overlap is None else chunk_overlap,
        )

    def _resolve_source_location(self, source_path: str) -> tuple[str, Path]:
        """按优先级解析源文件真实位置，并返回规范化相对路径。"""
        candidate_roots = [self.files_root.resolve()]
        fallback_root = self.app_settings.data_dir.resolve()
        if fallback_root not in candidate_roots:
            candidate_roots.append(fallback_root)

        last_error: FileNotFoundError | None = None
        for root in candidate_roots:
            normalized_path = validate_source_path(source_path=source_path, data_root=root)
            resolved_path = root / normalized_path
            if resolved_path.exists():
                return normalized_path, root
            last_error = FileNotFoundError(f"source_path does not exist: {normalized_path}")
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(f"source_path does not exist: {source_path}")

    def _load_source_records(
        self,
        *,
        namespace: str,
        source_path: str,
        data_root: Path,
    ) -> list[object]:
        return load_document_records(namespace=namespace, source_path=source_path, data_root=data_root)

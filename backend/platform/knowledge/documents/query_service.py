from __future__ import annotations

from pathlib import Path

from backend.platform.config.settings import AppSettings, FILES_DIR, settings
from backend.platform.knowledge.base.store import KnowledgeDocumentRepository, VectorStoreFactory
from backend.platform.knowledge.documents.mappers import KnowledgeDocumentMapper
from backend.platform.knowledge.documents.models import (
    KnowledgeDocumentDetail,
    KnowledgeDocumentNotFoundError,
    KnowledgeDocumentSummary,
    KnowledgeFileIndexSummary,
)
from backend.platform.knowledge.documents.store_support import KnowledgeDocumentRepositoryGateway
from backend.platform.knowledge.documents.validators import validate_namespace

MANAGED_FILE_EXTENSIONS = {".json", ".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx"}
INDEXABLE_FILE_EXTENSIONS = {".json", ".txt", ".md", ".csv"}


class KnowledgeManagedFileScanner:
    """只负责扫描上传目录里的受管文件，不负责拼接口响应。"""

    def __init__(self, files_root: Path) -> None:
        self.files_root = files_root

    def scan_files(self) -> list[Path]:
        if not self.files_root.exists():
            return []
        return sorted(
            (
                path
                for path in self.files_root.rglob("*")
                if path.is_file() and path.suffix.lower() in MANAGED_FILE_EXTENSIONS
            ),
            key=lambda path: path.stat().st_ctime,
            reverse=True,
        )

    def can_index(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in INDEXABLE_FILE_EXTENSIONS


class KnowledgeDocumentQueryService:
    """负责所有读路径：文档列表、文档详情和文件索引状态。"""

    def __init__(
        self,
        app_settings: AppSettings | None = None,
        repository: KnowledgeDocumentRepository | None = None,
        files_root: str | Path | None = None,
        mapper: KnowledgeDocumentMapper | None = None,
        file_scanner: KnowledgeManagedFileScanner | None = None,
    ) -> None:
        self.app_settings = app_settings or settings
        self.files_root = Path(files_root) if files_root is not None else FILES_DIR
        resolved_repository = repository or VectorStoreFactory.create_document_repository(self.app_settings)
        self.repository_gateway = KnowledgeDocumentRepositoryGateway(self.app_settings, resolved_repository)
        self.repository_gateway.ensure_document_indexes()
        self.mapper = mapper or KnowledgeDocumentMapper()
        self.file_scanner = file_scanner or KnowledgeManagedFileScanner(self.files_root)

    def list_documents(self, namespace: str | None = None) -> list[KnowledgeDocumentSummary]:
        if namespace is not None:
            validate_namespace(namespace)
        records = self.repository_gateway.list_document_records(namespace)
        return [self.mapper.to_summary(record) for record in records]

    def get_document(self, document_id: str) -> KnowledgeDocumentDetail:
        record = self.repository_gateway.get_document_record(document_id)
        if record is None:
            raise KnowledgeDocumentNotFoundError(f"Knowledge document not found: {document_id}")
        return self.mapper.to_detail(record)

    def list_file_indexes(self, namespace: str | None = None) -> list[KnowledgeFileIndexSummary]:
        if namespace is not None:
            validate_namespace(namespace)
        records = self.repository_gateway.list_document_records(namespace)
        document_by_path = {
            str(record["source_path"]): record
            for record in records
            if isinstance(record, dict)
        }

        summaries: list[KnowledgeFileIndexSummary] = []
        for file_path in self.file_scanner.scan_files():
            relative_path = file_path.relative_to(self.files_root).as_posix()
            record = document_by_path.get(relative_path)
            can_index = self.file_scanner.can_index(file_path)
            if record is None:
                summaries.append(
                    self.mapper.to_unindexed_file_summary(
                        file_path=file_path,
                        files_root=self.files_root,
                        can_index=can_index,
                    )
                )
                continue
            summaries.append(
                self.mapper.to_indexed_file_summary(
                    file_path=file_path,
                    files_root=self.files_root,
                    record=record,
                    can_index=can_index,
                )
            )
        return summaries

"""文档管理的校验、加载、分块与拆分服务工具包。"""

from backend.platform.knowledge.documents.application_service import KnowledgeDocumentApplicationService
from backend.platform.knowledge.documents.chunker import build_document_chunks
from backend.platform.knowledge.documents.loader import (
    build_document_id,
    build_source_record_id,
    load_document_records,
)
from backend.platform.knowledge.documents.mappers import KnowledgeDocumentMapper
from backend.platform.knowledge.documents.models import (
    KnowledgeDocumentDetail,
    KnowledgeDocumentError,
    KnowledgeDocumentNotFoundError,
    KnowledgeDocumentOperationResult,
    KnowledgeDocumentStoreError,
    KnowledgeDocumentSummary,
    KnowledgeDocumentVersionSummary,
    KnowledgeFileIndexSummary,
)
from backend.platform.knowledge.documents.publisher import KnowledgeDocumentPublisher
from backend.platform.knowledge.documents.query_service import (
    KnowledgeDocumentQueryService,
    KnowledgeManagedFileScanner,
)
from backend.platform.knowledge.documents.schemas import DocumentChunk, DocumentRecord
from backend.platform.knowledge.documents.store_support import KnowledgeDocumentRepositoryGateway
from backend.platform.knowledge.documents.validators import (
    validate_chunking,
    validate_namespace,
    validate_source_path,
)

__all__ = [
    "DocumentChunk",
    "DocumentRecord",
    # 这里明确只导出拆分后的读写服务，避免外部继续依赖已移除的聚合服务。
    "KnowledgeDocumentApplicationService",
    "KnowledgeDocumentDetail",
    "KnowledgeDocumentError",
    "KnowledgeDocumentMapper",
    "KnowledgeDocumentNotFoundError",
    "KnowledgeDocumentOperationResult",
    "KnowledgeDocumentPublisher",
    "KnowledgeDocumentQueryService",
    "KnowledgeDocumentRepositoryGateway",
    "KnowledgeDocumentStoreError",
    "KnowledgeDocumentSummary",
    "KnowledgeDocumentVersionSummary",
    "KnowledgeFileIndexSummary",
    "KnowledgeManagedFileScanner",
    "build_document_chunks",
    "build_document_id",
    "build_source_record_id",
    "load_document_records",
    "validate_chunking",
    "validate_namespace",
    "validate_source_path",
]

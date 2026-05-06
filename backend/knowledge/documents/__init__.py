"""文档管理的校验、加载与分块工具包。"""

from backend.knowledge.documents.chunker import build_document_chunks
from backend.knowledge.documents.loader import build_document_id, build_source_record_id, load_document_records
from backend.knowledge.documents.schemas import DocumentChunk, DocumentRecord
from backend.knowledge.documents.validators import validate_chunking, validate_namespace, validate_source_path

__all__ = [
    "DocumentChunk",
    "DocumentRecord",
    "build_document_chunks",
    "build_document_id",
    "build_source_record_id",
    "load_document_records",
    "validate_chunking",
    "validate_namespace",
    "validate_source_path",
]

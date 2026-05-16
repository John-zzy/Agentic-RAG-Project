from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.platform.knowledge.processing.schemas import ProcessingStats


DocumentStatus = Literal["active", "failed", "deleted"]


class KnowledgeDocumentError(RuntimeError):
    """文档管理服务基础异常。"""


class KnowledgeDocumentNotFoundError(KnowledgeDocumentError):
    """文档不存在或已删除。"""


class KnowledgeDocumentStoreError(KnowledgeDocumentError):
    """文档仓储读写失败。"""


KnowledgeDocumentProcessingStats = ProcessingStats


class KnowledgeDocumentVersionSummary(BaseModel):
    """单个文档版本摘要。"""

    document_version: int
    status: DocumentStatus
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    created_at: str
    source_type: str = "json"
    processing_rules: list[str] = Field(default_factory=list)
    processing_stats: ProcessingStats | None = None
    provenance_enabled: bool = False
    last_error: str | None = None


class KnowledgeDocumentSummary(BaseModel):
    """文档列表摘要。"""

    document_id: str
    namespace: str
    source_path: str
    status: DocumentStatus
    source_type: str = "json"
    processing_rules: list[str] = Field(default_factory=list)
    processing_stats: ProcessingStats | None = None
    provenance_enabled: bool = False
    active_version: int
    chunk_count: int
    updated_at: str


class KnowledgeDocumentDetail(KnowledgeDocumentSummary):
    """文档详情。"""

    source_type: str = "json"
    chunk_size: int
    chunk_overlap: int
    processing_rules: list[str] = Field(default_factory=list)
    processing_stats: ProcessingStats | None = None
    provenance_enabled: bool = False
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

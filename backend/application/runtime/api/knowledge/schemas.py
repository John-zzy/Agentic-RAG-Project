from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from backend.platform.knowledge.processing.config import DEFAULT_PROCESSING_CHUNK_CONFIG
from backend.platform.knowledge.processing.schemas import (
    ProcessingRuleDefinition,
    ProcessingSample,
    ProcessingStats,
    ProcessingWarning,
)


class _KnowledgeChunkingRequest(BaseModel):
    """知识文档切块参数基类。"""

    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "_KnowledgeChunkingRequest":
        """确保 overlap 小于 chunk size。"""
        if self.chunk_overlap is None or self.chunk_size is None:
            return self
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self

    def resolved_chunk_size(self) -> int:
        """返回请求中的分块大小，缺省时使用预处理模块默认值。"""
        return self.chunk_size if self.chunk_size is not None else DEFAULT_PROCESSING_CHUNK_CONFIG.chunk_size

    def resolved_chunk_overlap(self) -> int:
        """返回请求中的重叠长度，缺省时使用预处理模块默认值。"""
        return self.chunk_overlap if self.chunk_overlap is not None else DEFAULT_PROCESSING_CHUNK_CONFIG.chunk_overlap


class _KnowledgeSourceRequest(_KnowledgeChunkingRequest):
    """带源文件信息的知识文档请求基类。"""

    namespace: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    processing_rules: list[str] = Field(default_factory=list)


class KnowledgeDocumentRegisterRequest(_KnowledgeSourceRequest):
    """注册知识文档请求。"""

    keep_version: bool = False


class KnowledgeDocumentPreprocessPreviewRequest(_KnowledgeSourceRequest):
    """预处理预览请求。"""


class KnowledgeDocumentReprocessRequest(_KnowledgeChunkingRequest):
    """重处理请求。"""

    processing_rules: list[str] = Field(default_factory=list)
    keep_version: bool = False


class KnowledgeDocumentRechunkRequest(_KnowledgeChunkingRequest):
    """重建分块请求。"""

    keep_version: bool = False


class KnowledgeDocumentPreprocessPreviewResponse(BaseModel):
    """预处理预览响应。"""

    namespace: str
    source_path: str
    source_type: str = "json"
    chunk_size: int
    chunk_overlap: int
    supported_rules: list[ProcessingRuleDefinition] = Field(default_factory=list)
    selected_rules: list[ProcessingRuleDefinition] = Field(default_factory=list)
    processing_stats: ProcessingStats | None = None
    original_samples: list[ProcessingSample] = Field(default_factory=list)
    processed_samples: list[ProcessingSample] = Field(default_factory=list)
    can_index: bool = True
    warnings: list[ProcessingWarning] = Field(default_factory=list)


class KnowledgeDocumentVersionResponse(BaseModel):
    """文档版本摘要。"""

    document_version: int
    status: str
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    created_at: str
    source_type: str = "json"
    processing_rules: list[str] = Field(default_factory=list)
    processing_stats: ProcessingStats | None = None
    provenance_enabled: bool = False
    last_error: str | None = None


class KnowledgeDocumentSummaryResponse(BaseModel):
    """文档列表项。"""

    document_id: str
    namespace: str
    source_path: str
    status: str
    source_type: str = "json"
    processing_rules: list[str] = Field(default_factory=list)
    processing_stats: ProcessingStats | None = None
    provenance_enabled: bool = False
    active_version: int
    chunk_count: int
    updated_at: str


class KnowledgeDocumentListResponse(BaseModel):
    """文档列表响应。"""

    documents: list[KnowledgeDocumentSummaryResponse]


class KnowledgeFileIndexSummaryResponse(BaseModel):
    """按文件聚合的索引状态。"""

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


class KnowledgeFileIndexListResponse(BaseModel):
    """文件索引状态列表响应。"""

    items: list[KnowledgeFileIndexSummaryResponse]


class KnowledgeDocumentDetailResponse(KnowledgeDocumentSummaryResponse):
    """文档详情响应。"""

    source_type: str = "json"
    chunk_size: int
    chunk_overlap: int
    processing_rules: list[str] = Field(default_factory=list)
    processing_stats: ProcessingStats | None = None
    provenance_enabled: bool = False
    last_error: str | None = None
    versions: list[KnowledgeDocumentVersionResponse]


class KnowledgeDocumentOperationResponse(KnowledgeDocumentDetailResponse):
    """文档写操作响应。"""

    document_version: int


class KnowledgeDocumentDeleteResponse(KnowledgeDocumentOperationResponse):
    """文档删除响应。"""

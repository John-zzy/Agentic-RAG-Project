from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class KnowledgeDocumentRegisterRequest(BaseModel):
    """注册知识文档请求。"""

    namespace: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    keep_version: bool = False

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "KnowledgeDocumentRegisterRequest":
        """确保 overlap 小于 chunk size。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self


class KnowledgeDocumentRechunkRequest(BaseModel):
    """重建分块请求。"""

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    keep_version: bool = False

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "KnowledgeDocumentRechunkRequest":
        """确保 overlap 小于 chunk size。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self


class KnowledgeDocumentVersionResponse(BaseModel):
    """文档版本摘要。"""

    document_version: int
    status: str
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    created_at: str
    last_error: str | None = None


class KnowledgeDocumentSummaryResponse(BaseModel):
    """文档列表项。"""

    document_id: str
    namespace: str
    source_path: str
    status: str
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

    source_type: str
    chunk_size: int
    chunk_overlap: int
    last_error: str | None = None
    versions: list[KnowledgeDocumentVersionResponse]


class KnowledgeDocumentOperationResponse(KnowledgeDocumentDetailResponse):
    """文档写操作响应。"""

    document_version: int


class KnowledgeDocumentDeleteResponse(KnowledgeDocumentOperationResponse):
    """文档删除响应。"""

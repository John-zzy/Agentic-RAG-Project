from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class KnowledgeDocumentRegisterRequest(BaseModel):
    """注册知识文档的请求参数，指定源文件与分块策略。"""

    namespace: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    keep_version: bool = False

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "KnowledgeDocumentRegisterRequest":
        """确保分块重叠小于分块大小，避免切分窗口无法推进。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self


class KnowledgeDocumentRechunkRequest(BaseModel):
    """重新分块知识文档的请求参数。"""

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    keep_version: bool = False

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "KnowledgeDocumentRechunkRequest":
        """确保重新分块参数可以产生有效窗口。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self


class KnowledgeDocumentVersionResponse(BaseModel):
    """单个知识文档版本的状态与分块参数摘要。"""

    document_version: int
    status: str
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    created_at: str
    last_error: str | None = None


class KnowledgeDocumentSummaryResponse(BaseModel):
    """知识文档列表项摘要。"""

    document_id: str
    namespace: str
    source_path: str
    status: str
    active_version: int
    chunk_count: int
    updated_at: str


class KnowledgeDocumentListResponse(BaseModel):
    """知识文档列表响应。"""

    documents: list[KnowledgeDocumentSummaryResponse]


class KnowledgeDocumentDetailResponse(KnowledgeDocumentSummaryResponse):
    """知识文档详情响应，包含当前分块参数和版本列表。"""

    source_type: str
    chunk_size: int
    chunk_overlap: int
    last_error: str | None = None
    versions: list[KnowledgeDocumentVersionResponse]


class KnowledgeDocumentOperationResponse(KnowledgeDocumentDetailResponse):
    """注册、删除、重新分块等写操作响应。"""

    document_version: int


class KnowledgeDocumentDeleteResponse(KnowledgeDocumentOperationResponse):
    """删除知识文档后的结果响应。"""

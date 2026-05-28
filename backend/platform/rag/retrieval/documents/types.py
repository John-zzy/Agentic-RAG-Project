from __future__ import annotations

from pydantic import BaseModel, Field

from backend.platform.search_foundation import VectorStoreDocument


MATCHED_BY_ORDER = ("vector", "keyword")


class DocumentChunkRetrievalResult(BaseModel):
    """统一描述文档分块 Hybrid Search 命中结果。"""

    document: VectorStoreDocument
    score: float | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    matched_by: list[str] = Field(default_factory=list)


class DocumentRetrievalTopChunkTrace(BaseModel):
    """文档检索 trace 中的安全分块摘要，不包含完整正文。"""

    rank: int = Field(ge=1)
    citation_id: str
    document_id: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    source_name: str
    source_path: str | None = None
    score: float | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    matched_by: list[str] = Field(default_factory=list)


class DocumentRetrievalTrace(BaseModel):
    """描述文档检索召回和过滤边界的最小 trace。"""

    raw_candidates_count: int = Field(default=0, ge=0)
    filtered_candidates_count: int = Field(default=0, ge=0)
    top_k_chunks: list[DocumentRetrievalTopChunkTrace] = Field(default_factory=list)


class DocumentRetrievalTraceResult(BaseModel):
    """携带既有检索结果和可观测 trace 的内部返回结构。"""

    results: list[DocumentChunkRetrievalResult] = Field(default_factory=list)
    trace: DocumentRetrievalTrace = Field(default_factory=DocumentRetrievalTrace)


def merge_matched_by(*groups: list[str]) -> list[str]:
    """按固定优先级合并 matched_by，保证输出稳定。"""
    merged: list[str] = []
    for source in MATCHED_BY_ORDER:
        if any(source in group for group in groups):
            merged.append(source)
    for group in groups:
        for source in group:
            if source not in merged:
                merged.append(source)
    return merged

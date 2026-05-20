from __future__ import annotations

from pydantic import BaseModel, Field

from backend.platform.retrieval import VectorStoreDocument


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

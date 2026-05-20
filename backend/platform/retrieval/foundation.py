from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Protocol

from pydantic import BaseModel, Field


VectorMetadata = dict[str, Any]
MetadataValue = str | int | float | bool


class VectorStoreDocument(BaseModel):
    """描述进入向量库的一条标准化文档。"""

    id: str
    content: str
    metadata: VectorMetadata = Field(default_factory=dict)
    embedding: list[float] | None = None


class VectorSearchResult(BaseModel):
    """描述一次向量检索命中的文档和得分。"""

    document: VectorStoreDocument
    score: float | None = None


class VectorStoreHealth(BaseModel):
    """描述向量后端可用性探活结果。"""

    provider: str
    available: bool
    detail: str | None = None


class EmbeddingStrategy(Protocol):
    """定义文本转向量的统一入口。"""

    def embed(self, text: str) -> list[float]:
        """把一段文本转换成向量。"""


def tokenize_text(text: str) -> list[str]:
    """将文本切分为英文 token、中文单字与 n-gram。"""
    ascii_tokens = re.findall(r"[a-z0-9]+", text)
    cjk_sequences = re.findall(r"[\u4e00-\u9fff]+", text)
    cjk_chars = [char for sequence in cjk_sequences for char in sequence]
    ngrams: list[str] = []

    for sequence in cjk_sequences:
        if len(sequence) > 1:
            ngrams.append(sequence)
        for size in (2, 3):
            if len(sequence) < size:
                continue
            for index in range(len(sequence) - size + 1):
                ngrams.append(sequence[index : index + size])

    return ascii_tokens + cjk_chars + ngrams


class LocalHashingEmbedder:
    """提供无外部依赖的本地哈希向量化实现。"""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        normalized = text.strip().lower()
        vector = [0.0] * self.dimensions
        tokens = self.tokenize(normalized)

        if not tokens:
            vector[0] = 1.0
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector

        return [value / magnitude for value in vector]

    def tokenize(self, text: str) -> list[str]:
        return tokenize_text(text)

    def _tokenize(self, text: str) -> list[str]:
        """兼容旧调用面，避免一次性打断测试桩或遗留路径。"""
        return self.tokenize(text)


class SemanticVectorQueryRepository(ABC):
    """定义命名空间级语义检索读侧契约。"""

    @abstractmethod
    def ensure_collections(self) -> None:
        """创建或校验后端所需命名空间。"""

    @abstractmethod
    def search(
        self,
        namespace: str,
        query: str,
        top_k: int | None = None,
        filters: VectorMetadata | None = None,
    ) -> list[VectorSearchResult]:
        """在指定命名空间执行语义检索。"""

    @abstractmethod
    def healthcheck(self) -> VectorStoreHealth:
        """返回向量后端可用性与连通性信息。"""


class SemanticDocumentStoreRepository(SemanticVectorQueryRepository):
    """定义命名空间级语义检索兼写入契约，供兼容场景与预加载流程使用。"""

    @abstractmethod
    def upsert_documents(self, namespace: str, documents: list[VectorStoreDocument]) -> None:
        """在指定命名空间写入或更新文档。"""

    @abstractmethod
    def delete_documents(self, namespace: str, ids: list[str]) -> None:
        """按文档 ID 删除指定命名空间下的数据。"""


class DocumentChunkVectorRepository(ABC):
    """定义文档分块向量查询能力，供 RAG 读侧复用。"""

    @abstractmethod
    def search_document_chunk_vectors(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        namespace: str | None = None,
    ) -> list[VectorSearchResult]:
        """基于 query embedding 搜索活跃文档分块。"""


class ActiveDocumentChunkSource(ABC):
    """定义活跃文档分块文本源枚举能力，供关键词召回使用。"""

    @abstractmethod
    def list_active_document_chunks(
        self,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[VectorStoreDocument]:
        """枚举活跃文档分块的文本与元数据。"""

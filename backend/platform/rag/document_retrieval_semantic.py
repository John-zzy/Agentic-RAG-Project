from __future__ import annotations

from backend.platform.knowledge.base.store import DocumentChunkVectorRepository

from backend.platform.rag.document_retrieval_embedding import DocumentEmbeddingStrategy
from backend.platform.rag.document_retrieval_types import DocumentChunkRetrievalResult


class DocumentSemanticRetriever:
    """基于向量仓储执行文档语义召回。"""

    def __init__(
        self,
        *,
        vector_repository: DocumentChunkVectorRepository,
        embedding_strategy: DocumentEmbeddingStrategy | None = None,
    ) -> None:
        self.vector_repository = vector_repository
        self.embedding_strategy = embedding_strategy or DocumentEmbeddingStrategy()

    def retrieve(
        self,
        *,
        query: str,
        top_k: int,
        namespace: str | None = None,
    ) -> list[DocumentChunkRetrievalResult]:
        query_embedding = self.embedding_strategy.embed(query)
        results = self.vector_repository.search_document_chunk_vectors(
            query_embedding=query_embedding,
            top_k=top_k,
            namespace=namespace,
        )
        return [
            DocumentChunkRetrievalResult(
                document=result.document,
                vector_score=result.score,
                matched_by=["vector"],
            )
            for result in results
        ]

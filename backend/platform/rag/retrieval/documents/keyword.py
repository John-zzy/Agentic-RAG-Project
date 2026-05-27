from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from backend.platform.rag.retrieval.documents.keyword_scoring import DocumentKeywordScoreCalculator
from backend.platform.rag.retrieval.documents.types import DocumentChunkRetrievalResult
from backend.platform.search_foundation import ActiveDocumentChunkSource, LocalHashingEmbedder, VectorStoreDocument


class DocumentKeywordRetriever:
    """基于 BM25 的文档关键词召回器。"""

    def __init__(
        self,
        *,
        chunk_source: ActiveDocumentChunkSource,
        score_calculator: DocumentKeywordScoreCalculator | None = None,
    ) -> None:
        self.chunk_source = chunk_source
        self._score_calculator = score_calculator or DocumentKeywordScoreCalculator(
            embedder=LocalHashingEmbedder()
        )

    def retrieve(
        self,
        *,
        query: str,
        top_k: int,
        namespace: str | None = None,
    ) -> list[DocumentChunkRetrievalResult]:
        candidate_limit = max(top_k * 10, 20)
        chunks = self.chunk_source.list_active_document_chunks(namespace=namespace, limit=candidate_limit)
        if not chunks:
            return []

        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "_chunk_id": chunk.id,
                },
            )
            for chunk in chunks
        ]
        retriever = BM25Retriever.from_documents(documents, k=min(top_k, len(documents)))
        scored_documents = self._score_calculator.score_documents(retriever, query)
        if not scored_documents:
            return []

        ranked_hits = sorted(
            (
                document
                for document in retriever.docs
                if str(document.metadata.get("_chunk_id") or document.metadata.get("chunk_id") or "") in scored_documents
            ),
            key=lambda document: scored_documents[
                str(document.metadata.get("_chunk_id") or document.metadata.get("chunk_id") or "")
            ],
            reverse=True,
        )[:top_k]
        results: list[DocumentChunkRetrievalResult] = []
        for hit in ranked_hits:
            chunk_id = str(hit.metadata.get("_chunk_id") or hit.metadata.get("chunk_id") or "")
            if not chunk_id:
                continue
            results.append(
                DocumentChunkRetrievalResult(
                    document=VectorStoreDocument(
                        id=chunk_id,
                        content=hit.page_content,
                        metadata={key: value for key, value in hit.metadata.items() if key != "_chunk_id"},
                    ),
                    keyword_score=scored_documents.get(chunk_id),
                    matched_by=["keyword"],
                )
            )
        return results

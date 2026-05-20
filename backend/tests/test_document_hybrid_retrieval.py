from __future__ import annotations

from backend.platform.config.settings import AppSettings, VectorStoreConfig
from backend.platform.retrieval import VectorSearchResult, VectorStoreDocument
from backend.platform.rag.document_retrieval import (
    DocumentChunkRetrievalResult,
    DocumentEmbeddingStrategy,
    DocumentHybridRetriever,
    DocumentKeywordRetriever,
    DocumentRetrievalService,
    DocumentSemanticRetriever,
    HybridFusionRanker,
)
from backend.tests.test_support import make_test_runtime_dir


class FakeDocumentChunkVectorRepository:
    def __init__(self, results: list[VectorStoreDocument]) -> None:
        self._results = results

    def search_document_chunk_vectors(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        namespace: str | None = None,
    ) -> list[object]:
        del query_embedding, namespace
        return [
            VectorSearchResult(document=document, score=score)
            for document, score in [
                (document, float(document.metadata.get("_vector_score", 0.0)))
                for document in self._results[: top_k or len(self._results)]
            ]
        ]


class FakeActiveDocumentChunkSource:
    def __init__(self, chunks: list[VectorStoreDocument]) -> None:
        self._chunks = chunks

    def list_active_document_chunks(
        self,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[VectorStoreDocument]:
        filtered = [
            chunk
            for chunk in self._chunks
            if namespace is None or chunk.metadata.get("namespace") == namespace
        ]
        return filtered[: limit or len(filtered)]


def _chunk(
    chunk_id: str,
    content: str,
    *,
    document_id: str,
    namespace: str = "faq",
    chunk_index: int = 0,
    vector_score: float = 0.0,
) -> VectorStoreDocument:
    return VectorStoreDocument(
        id=chunk_id,
        content=content,
        metadata={
            "document_id": document_id,
            "chunk_id": chunk_id,
            "namespace": namespace,
            "chunk_index": chunk_index,
            "source_path": f"{document_id}.md",
            "is_managed_document": True,
            "_vector_score": vector_score,
        },
    )


def test_keyword_retriever_can_recall_unique_keyword_chunk() -> None:
    source = FakeActiveDocumentChunkSource(
        [
            _chunk("chunk-1", "唯一令牌 alphabetaomega 出现在这里", document_id="doc-1"),
            _chunk("chunk-2", "常规帮助文档", document_id="doc-2"),
        ]
    )
    retriever = DocumentKeywordRetriever(chunk_source=source)

    results = retriever.retrieve(query="alphabetaomega", top_k=3, namespace="faq")

    assert results
    assert results[0].document.id == "chunk-1"
    assert results[0].keyword_score is not None
    assert results[0].matched_by == ["keyword"]


def test_hybrid_fusion_ranker_marks_vector_only_keyword_only_and_both() -> None:
    vector_only = DocumentChunkRetrievalResult(
        document=_chunk("vector-only", "向量命中", document_id="doc-v"),
        vector_score=0.8,
        matched_by=["vector"],
    )
    both_vector = DocumentChunkRetrievalResult(
        document=_chunk("both", "双路命中 唯一关键词", document_id="doc-b"),
        vector_score=0.7,
        matched_by=["vector"],
    )
    both_keyword = DocumentChunkRetrievalResult(
        document=_chunk("both", "双路命中 唯一关键词", document_id="doc-b"),
        keyword_score=1.0,
        matched_by=["keyword"],
    )
    keyword_only = DocumentChunkRetrievalResult(
        document=_chunk("keyword-only", "只有关键词命中 specialtoken", document_id="doc-k"),
        keyword_score=0.9,
        matched_by=["keyword"],
    )

    ranked = HybridFusionRanker().rank(
        vector_results=[vector_only, both_vector],
        keyword_results=[both_keyword, keyword_only],
        top_k=3,
    )

    by_id = {result.document.id: result for result in ranked}
    assert by_id["vector-only"].matched_by == ["vector"]
    assert by_id["keyword-only"].matched_by == ["keyword"]
    assert by_id["both"].matched_by == ["vector", "keyword"]
    assert by_id["both"].vector_rank == 2
    assert by_id["both"].keyword_rank == 1


def test_document_retrieval_service_returns_documents_with_hybrid_metadata() -> None:
    runtime_dir = make_test_runtime_dir("document-hybrid-service")
    app_settings = AppSettings(
        data_dir=runtime_dir,
        vector_store=VectorStoreConfig(provider="chroma"),
    )
    chunks = [
        _chunk("chunk-1", "产品手册包含独特词汇 alphabetaomega", document_id="doc-1", vector_score=0.75),
        _chunk("chunk-2", "普通FAQ说明", document_id="doc-2", vector_score=0.65),
    ]
    service = DocumentRetrievalService(
        app_settings=app_settings,
        vector_repository=FakeDocumentChunkVectorRepository(chunks),
        chunk_source=FakeActiveDocumentChunkSource(chunks),
        fusion_ranker=HybridFusionRanker(),
    )

    documents = service.search(query="alphabetaomega", top_k=3, namespace="faq")

    assert documents
    first = documents[0]
    assert first.metadata["citation_id"] == "chunk-1"
    assert first.metadata["document_id"] == "doc-1"
    assert first.metadata["chunk_index"] == 0
    assert first.metadata["score"] is not None
    assert first.metadata["keyword_score"] is not None
    assert "keyword" in first.metadata["matched_by"]


def test_document_hybrid_retriever_delegates_to_service() -> None:
    runtime_dir = make_test_runtime_dir("document-hybrid-retriever")
    app_settings = AppSettings(
        data_dir=runtime_dir,
        vector_store=VectorStoreConfig(provider="chroma"),
    )
    chunk = _chunk("chunk-1", "这里有 unique-hybrid-key", document_id="doc-1", vector_score=0.8)
    service = DocumentRetrievalService(
        app_settings=app_settings,
        vector_repository=FakeDocumentChunkVectorRepository([chunk]),
        chunk_source=FakeActiveDocumentChunkSource([chunk]),
    )
    retriever = service.build_retriever(default_top_k=2, namespace="faq")

    documents = retriever.invoke("unique-hybrid-key")

    assert len(documents) == 1
    assert documents[0].metadata["citation_id"] == "chunk-1"

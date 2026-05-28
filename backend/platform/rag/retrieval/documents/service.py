from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from backend.platform.config.settings import AppSettings, settings
from backend.platform.rag.retrieval.documents.filters import (
    DOCUMENT_MINIMUM_RELEVANCE,
    filter_low_relevance_document_results,
    filter_managed_document_results,
)
from backend.platform.rag.contracts import RecallStrategy
from backend.platform.rag.retrieval.documents.fusion import HybridFusionRanker
from backend.platform.rag.retrieval.documents.keyword import DocumentKeywordRetriever
from backend.platform.rag.retrieval.documents.semantic import DocumentSemanticRetriever
from backend.platform.rag.retrieval.documents.types import (
    DocumentChunkRetrievalResult,
    DocumentRetrievalTopChunkTrace,
    DocumentRetrievalTrace,
    DocumentRetrievalTraceResult,
)
from backend.platform.search_foundation import (
    ActiveDocumentChunkSource,
    DocumentChunkVectorRepository,
    VectorSearchResult,
    VectorStoreDocument,
)


class DocumentHybridRetriever(BaseRetriever):
    """文档 Hybrid Search 统一 LangChain retriever。"""

    retrieval_service: "DocumentRetrievalService" = Field(exclude=True)
    namespace: str | None = None
    default_top_k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        del run_manager
        return self.retrieval_service.search(
            query=query,
            top_k=self.default_top_k,
            namespace=self.namespace,
        )

    def search(
        self,
        *,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        minimum_relevance: float | None = None,
        recall_strategy: RecallStrategy = "hybrid",
    ) -> list[Document]:
        return self.retrieval_service.search(
            query=query,
            top_k=top_k or self.default_top_k,
            namespace=namespace or self.namespace,
            minimum_relevance=minimum_relevance,
            recall_strategy=recall_strategy,
        )


class DocumentRetrievalService:
    """文档知识检索统一入口。"""

    def __init__(
        self,
        *,
        app_settings: AppSettings | None = None,
        vector_repository: DocumentChunkVectorRepository,
        chunk_source: ActiveDocumentChunkSource,
        semantic_retriever: DocumentSemanticRetriever | None = None,
        keyword_retriever: DocumentKeywordRetriever | None = None,
        fusion_ranker: HybridFusionRanker | None = None,
        files_root: str | None = None,
        minimum_relevance: float = DOCUMENT_MINIMUM_RELEVANCE,
    ) -> None:
        self.app_settings = app_settings or settings
        self.files_root = files_root or str(self.app_settings.data_dir / "files")
        self.minimum_relevance = minimum_relevance
        self.vector_repository = vector_repository
        self.chunk_source = chunk_source
        self.semantic_retriever = semantic_retriever or DocumentSemanticRetriever(
            vector_repository=self.vector_repository
        )
        self.keyword_retriever = keyword_retriever or DocumentKeywordRetriever(
            chunk_source=self.chunk_source
        )
        self.fusion_ranker = fusion_ranker or HybridFusionRanker()

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        minimum_relevance: float | None = None,
        recall_strategy: RecallStrategy = "hybrid",
    ) -> list[DocumentChunkRetrievalResult]:
        return self.retrieve_with_trace(
            query=query,
            top_k=top_k,
            namespace=namespace,
            minimum_relevance=minimum_relevance,
            recall_strategy=recall_strategy,
        ).results

    def retrieve_with_trace(
        self,
        *,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        minimum_relevance: float | None = None,
        recall_strategy: RecallStrategy = "hybrid",
    ) -> DocumentRetrievalTraceResult:
        results = self._retrieve_by_strategy(
            query=query,
            top_k=top_k,
            namespace=namespace,
            recall_strategy=recall_strategy,
        )
        raw_candidates_count = len(results)
        filtered_results = filter_low_relevance_document_results(
            [VectorSearchResult(document=result.document, score=result.score) for result in results],
            minimum_relevance=(
                self.minimum_relevance if minimum_relevance is None else minimum_relevance
            ),
        )
        allowed_ids = {result.document.id for result in filtered_results}
        final_results = [result for result in results if result.document.id in allowed_ids]
        final_results = self._filter_managed_documents(final_results)
        return DocumentRetrievalTraceResult(
            results=final_results,
            trace=DocumentRetrievalTrace(
                raw_candidates_count=raw_candidates_count,
                filtered_candidates_count=len(final_results),
                top_k_chunks=self._build_top_chunk_trace(final_results),
            ),
        )

    def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        minimum_relevance: float | None = None,
        recall_strategy: RecallStrategy = "hybrid",
    ) -> list[Document]:
        return [
            self._to_document(result)
            for result in self.retrieve(
                query=query,
                top_k=top_k,
                namespace=namespace,
                minimum_relevance=minimum_relevance,
                recall_strategy=recall_strategy,
            )
        ]

    def build_retriever(
        self,
        *,
        default_top_k: int = 5,
        namespace: str | None = None,
    ) -> DocumentHybridRetriever:
        return DocumentHybridRetriever(
            retrieval_service=self,
            default_top_k=default_top_k,
            namespace=namespace,
        )

    def _filter_managed_documents(
        self,
        results: list[DocumentChunkRetrievalResult],
    ) -> list[DocumentChunkRetrievalResult]:
        filtered = filter_managed_document_results(
            [VectorSearchResult(document=result.document, score=result.score) for result in results],
            files_root=self.files_root,
        )
        allowed_ids = {result.document.id for result in filtered}
        return [result for result in results if result.document.id in allowed_ids]

    def _retrieve_by_strategy(
        self,
        *,
        query: str,
        top_k: int,
        namespace: str | None,
        recall_strategy: RecallStrategy,
    ) -> list[DocumentChunkRetrievalResult]:
        # 召回策略只决定候选来源，相关性过滤和托管文档过滤仍走统一出口。
        if recall_strategy == "semantic":
            return self._semantic_results(query=query, top_k=top_k, namespace=namespace)
        if recall_strategy == "keyword":
            return self._keyword_results(query=query, top_k=top_k, namespace=namespace)
        if recall_strategy == "hybrid":
            vector_results = self.semantic_retriever.retrieve(
                query=query,
                top_k=top_k,
                namespace=namespace,
            )
            keyword_results = self.keyword_retriever.retrieve(
                query=query,
                top_k=top_k,
                namespace=namespace,
            )
            return self.fusion_ranker.rank(
                vector_results=vector_results,
                keyword_results=keyword_results,
                top_k=top_k,
            )
        raise ValueError(f"Unsupported document recall_strategy: {recall_strategy!r}")

    def _semantic_results(
        self,
        *,
        query: str,
        top_k: int,
        namespace: str | None,
    ) -> list[DocumentChunkRetrievalResult]:
        results = self.semantic_retriever.retrieve(query=query, top_k=top_k, namespace=namespace)
        return [
            result.model_copy(
                update={
                    "score": result.vector_score,
                    "vector_rank": rank,
                    "matched_by": result.matched_by or ["vector"],
                }
            )
            for rank, result in enumerate(results, start=1)
        ]

    def _keyword_results(
        self,
        *,
        query: str,
        top_k: int,
        namespace: str | None,
    ) -> list[DocumentChunkRetrievalResult]:
        results = self.keyword_retriever.retrieve(query=query, top_k=top_k, namespace=namespace)
        return [
            result.model_copy(
                update={
                    "score": result.keyword_score,
                    "keyword_rank": rank,
                    "matched_by": result.matched_by or ["keyword"],
                }
            )
            for rank, result in enumerate(results, start=1)
        ]

    def _to_document(self, result: DocumentChunkRetrievalResult) -> Document:
        metadata = {
            **result.document.metadata,
            "namespace": self._resolve_namespace(result.document.metadata),
            "citation_id": self._resolve_citation_id(result.document),
            "score": result.score,
            "vector_score": result.vector_score,
            "keyword_score": result.keyword_score,
            "vector_rank": result.vector_rank,
            "keyword_rank": result.keyword_rank,
            "matched_by": list(result.matched_by),
        }
        return Document(page_content=result.document.content, metadata=metadata)

    def _build_top_chunk_trace(
        self,
        results: list[DocumentChunkRetrievalResult],
    ) -> list[DocumentRetrievalTopChunkTrace]:
        return [
            DocumentRetrievalTopChunkTrace(
                rank=rank,
                citation_id=self._resolve_citation_id(result.document),
                document_id=self._resolve_optional_str(result.document.metadata.get("document_id")),
                chunk_id=self._resolve_optional_str(result.document.metadata.get("chunk_id")),
                chunk_index=self._resolve_int(result.document.metadata.get("chunk_index")),
                source_name=self._resolve_source_name(result.document),
                source_path=self._resolve_optional_str(result.document.metadata.get("source_path")),
                score=float(result.score) if isinstance(result.score, int | float) else None,
                vector_score=(
                    float(result.vector_score)
                    if isinstance(result.vector_score, int | float)
                    else None
                ),
                keyword_score=(
                    float(result.keyword_score)
                    if isinstance(result.keyword_score, int | float)
                    else None
                ),
                vector_rank=result.vector_rank,
                keyword_rank=result.keyword_rank,
                matched_by=list(result.matched_by),
            )
            for rank, result in enumerate(results, start=1)
        ]

    def _resolve_namespace(self, metadata: dict[str, Any]) -> str:
        namespace = metadata.get("namespace")
        if isinstance(namespace, str) and namespace:
            return namespace
        return "documents"

    def _resolve_citation_id(self, document: VectorStoreDocument) -> str:
        metadata = document.metadata
        return str(
            metadata.get("chunk_id")
            or metadata.get("document_id")
            or metadata.get("source_path")
            or metadata.get("id")
            or document.id
        )

    def _resolve_source_name(self, document: VectorStoreDocument) -> str:
        metadata = document.metadata
        source_path = self._resolve_optional_str(metadata.get("source_path"))
        if source_path:
            return Path(source_path).name
        for field_name in ("title", "name", "document_id", "chunk_id", "id"):
            value = self._resolve_optional_str(metadata.get(field_name))
            if value:
                return value
        return document.id

    def _resolve_optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, int | float):
            return str(value)
        return None

    def _resolve_int(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

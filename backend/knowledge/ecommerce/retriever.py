from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from backend.knowledge.base.store import VectorSearchResult
from backend.knowledge.base.text import truncate_snippet


class KnowledgeBaseRetriever(BaseRetriever):
    """将电商知识检索结果适配为 LangChain Retriever。"""

    knowledge_service: Any = Field(exclude=True)
    default_top_k: int = 5
    minimum_relevance: float = 0.18

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        """适配 LangChain 检索器协议，返回相关文档。"""
        return self.search(query=query, top_k=self.default_top_k)

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """聚合商品与评论检索结果，排序去重后返回。"""
        requested_top_k = top_k or self.default_top_k
        product_results = self.knowledge_service.search_products(query=query, top_k=requested_top_k)
        review_results = self.knowledge_service.search_reviews(query=query, top_k=requested_top_k)

        combined = self._to_documents("products", product_results) + self._to_documents("reviews", review_results)
        combined.sort(key=self._doc_score, reverse=True)

        deduped: list[Document] = []
        seen: set[tuple[str, str]] = set()
        for doc in combined:
            namespace = str(doc.metadata.get("namespace", "knowledge"))
            citation_id = str(doc.metadata.get("citation_id", "unknown"))
            key = (namespace, citation_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
            if len(deduped) >= requested_top_k:
                break
        return deduped

    def _to_documents(self, namespace: str, results: list[VectorSearchResult]) -> list[Document]:
        """将向量检索结果转换为统一的 LangChain Document。"""
        documents: list[Document] = []
        for result in results:
            score = float(result.score) if result.score is not None else None
            if score is not None and score < self.minimum_relevance:
                continue
            metadata = result.document.metadata
            citation_id = str(
                metadata.get("review_id")
                or metadata.get("product_id")
                or metadata.get("id")
                or result.document.id
            )
            snippet = truncate_snippet(result.document.content)
            if not snippet:
                continue
            documents.append(
                Document(
                    page_content=snippet,
                    metadata={"namespace": namespace, "citation_id": citation_id, "score": score},
                )
            )
        return documents

    def _doc_score(self, doc: Document) -> float:
        """提取文档分数用于排序；缺失时返回较低默认值。"""
        score = doc.metadata.get("score")
        if isinstance(score, int | float):
            return float(score)
        return -1.0

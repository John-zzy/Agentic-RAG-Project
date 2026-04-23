from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from backend.config.settings import AppSettings, settings
from backend.knowledge.extractor import build_product_document, build_review_document
from backend.knowledge.store import (
    VectorSearchResult,
    VectorStore,
    VectorStoreDocument,
    VectorStoreFactory,
)

SUPPORTED_KNOWLEDGE_NAMESPACES = ("products", "reviews")


class KnowledgeUpsertSummary(BaseModel):
    namespace: str
    upserted: int


class KnowledgeService:
    def __init__(
        self,
        app_settings: AppSettings | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = app_settings or settings
        self.store = store or VectorStoreFactory.create(self.settings)
        # 统一在服务初始化时准备命名空间，避免调用方关心底层 collection/index 生命周期。
        self.store.ensure_collections()

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        self._validate_namespace(namespace)
        return self.store.search(namespace=namespace, query=query, top_k=top_k, filters=filters)

    def search_products(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        return self.search(namespace="products", query=query, top_k=top_k, filters=filters)

    def search_reviews(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        return self.search(namespace="reviews", query=query, top_k=top_k, filters=filters)

    def upsert_products(self, products: list[dict[str, Any]]) -> KnowledgeUpsertSummary:
        documents = [build_product_document(product) for product in products]
        return self._upsert_documents("products", documents)

    def upsert_reviews(self, reviews: list[dict[str, Any]]) -> KnowledgeUpsertSummary:
        documents = [build_review_document(review) for review in reviews]
        return self._upsert_documents("reviews", documents)

    def delete_documents(self, namespace: str, ids: list[str]) -> None:
        self._validate_namespace(namespace)
        self.store.delete_documents(namespace=namespace, ids=ids)

    def _upsert_documents(
        self,
        namespace: str,
        documents: list[VectorStoreDocument],
    ) -> KnowledgeUpsertSummary:
        self._validate_namespace(namespace)
        # 增量更新对上层暴露统一语义，具体写入细节由不同后端实现自行处理。
        self.store.upsert_documents(namespace=namespace, documents=documents)
        return KnowledgeUpsertSummary(namespace=namespace, upserted=len(documents))

    def _validate_namespace(self, namespace: str) -> None:
        if namespace not in SUPPORTED_KNOWLEDGE_NAMESPACES:
            raise ValueError(
                f"Unsupported namespace '{namespace}'. "
                f"Expected one of: {', '.join(SUPPORTED_KNOWLEDGE_NAMESPACES)}."
            )


def create_knowledge_service(
    app_settings: AppSettings | None = None,
    store: VectorStore | None = None,
) -> KnowledgeService:
    return KnowledgeService(app_settings=app_settings, store=store)

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from backend.platform.config.settings import AppSettings, settings
from backend.platform.knowledge.base.store import (
    KnowledgeRetriever,
    VectorSearchResult,
    VectorStore,
    VectorStoreDocument,
    VectorStoreFactory,
)
from backend.scenes.ecommerce.extractor import build_order_document, build_product_document, build_review_document


class KnowledgeUpsertSummary(BaseModel):
    """描述单个命名空间的数据写入结果。"""

    namespace: str
    upserted: int


class KnowledgeService:
    """封装电商知识库与扩展知识文档的业务入口。"""

    def __init__(
        self,
        app_settings: AppSettings | None = None,
        store: VectorStore | KnowledgeRetriever | None = None,
    ) -> None:
        """初始化知识服务并确保向量库命名空间可用。"""
        self.settings = app_settings or settings
        self.store = store or VectorStoreFactory.create_retriever(self.settings)
        self.store.ensure_collections()

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """按命名空间执行语义检索。"""
        self._validate_namespace(namespace)
        return self.store.search(namespace=namespace, query=query, top_k=top_k, filters=filters)

    def search_products(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """在商品知识库中执行检索。"""
        return self.search(namespace="products", query=query, top_k=top_k, filters=filters)

    def search_reviews(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """在评价知识库中执行检索。"""
        return self.search(namespace="reviews", query=query, top_k=top_k, filters=filters)

    def search_orders(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """在订单知识库中执行检索。"""
        return self.search(namespace="orders", query=query, top_k=top_k, filters=filters)

    def search_document_chunks(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
    ) -> list[VectorSearchResult]:
        """在用户上传知识文档的分块索引中执行检索。"""
        if namespace is not None:
            self._validate_document_namespace(namespace)
        return self.store.search_document_chunks(query=query, top_k=top_k, namespace=namespace)

    def upsert_products(self, products: list[dict[str, Any]]) -> KnowledgeUpsertSummary:
        """批量写入或更新商品数据。"""
        documents = [build_product_document(product) for product in products]
        return self._upsert_documents("products", documents)

    def upsert_reviews(self, reviews: list[dict[str, Any]]) -> KnowledgeUpsertSummary:
        """批量写入或更新评价数据。"""
        documents = [build_review_document(review) for review in reviews]
        return self._upsert_documents("reviews", documents)

    def upsert_orders(self, orders: list[dict[str, Any]]) -> KnowledgeUpsertSummary:
        """批量写入或更新订单数据。"""
        documents = [build_order_document(order) for order in orders]
        return self._upsert_documents("orders", documents)

    def delete_documents(self, namespace: str, ids: list[str]) -> None:
        """按命名空间与文档 ID 删除向量文档。"""
        self._validate_namespace(namespace)
        self.store.delete_documents(namespace=namespace, ids=ids)

    def _upsert_documents(
        self,
        namespace: str,
        documents: list[VectorStoreDocument],
    ) -> KnowledgeUpsertSummary:
        """执行统一的 upsert 流程并返回写入数量。"""
        self._validate_namespace(namespace)
        self.store.upsert_documents(namespace=namespace, documents=documents)
        return KnowledgeUpsertSummary(namespace=namespace, upserted=len(documents))

    def _validate_namespace(self, namespace: str) -> None:
        """校验内置命名空间是否合法。"""
        supported_namespaces = tuple(self.settings.vector_store.knowledge_sources.keys())
        if namespace not in supported_namespaces:
            raise ValueError(
                f"Unsupported namespace '{namespace}'. "
                f"Expected one of: {', '.join(supported_namespaces)}."
            )

    def _validate_document_namespace(self, namespace: str) -> None:
        """校验知识文档命名空间格式。"""
        if not namespace or namespace.strip() != namespace:
            raise ValueError("namespace must be a non-empty slug.")


def create_knowledge_service(
    app_settings: AppSettings | None = None,
    store: VectorStore | None = None,
) -> KnowledgeService:
    """知识服务工厂函数。"""
    return KnowledgeService(app_settings=app_settings, store=store)

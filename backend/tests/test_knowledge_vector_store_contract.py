from __future__ import annotations

from uuid import uuid4

import pytest

from backend.platform.config.settings import AppSettings, VectorStoreConfig
from backend.platform.knowledge.base.store import ChromaVectorStore, ElasticsearchVectorStore
from backend.platform.knowledge.base.store import VectorStoreDocument
from backend.tests.test_support import DATA_DIR, make_test_runtime_dir


@pytest.fixture(params=["chroma", "elasticsearch"])
def knowledge_service(request: pytest.FixtureRequest):
    provider = request.param
    if provider == "chroma":
        tmp_path = make_test_runtime_dir("knowledge-contract-chroma")
        app_settings = AppSettings(
            data_dir=DATA_DIR,
            vector_store=VectorStoreConfig(
                provider="chroma",
                chroma={"persist_directory": tmp_path / ".chroma"},
            ),
        )
        yield _StoreBackedKnowledgeService(ChromaVectorStore(app_settings))
        return

    index_prefix = f"ai-rag-contract-{uuid4().hex[:8]}"
    app_settings = AppSettings(
        data_dir=DATA_DIR,
        vector_store=VectorStoreConfig(
            provider="elasticsearch",
            elasticsearch={
                "url": "http://localhost:9200",
                "index_prefix": index_prefix,
                "verify_certs": False,
            },
        ),
    )
    try:
        service = _StoreBackedKnowledgeService(ElasticsearchVectorStore(app_settings))
    except Exception as exc:
        pytest.skip(
            "Elasticsearch is unavailable at http://localhost:9200. "
            f"Start local ES to run contract tests. Details: {exc}"
        )
    elasticsearch_store = service.store
    if not isinstance(elasticsearch_store, ElasticsearchVectorStore):
        pytest.fail("Expected ElasticsearchVectorStore when provider=elasticsearch.")

    health = elasticsearch_store.healthcheck()
    if not health.available:
        pytest.skip(
            "Elasticsearch is unavailable at http://localhost:9200. "
            "Start local ES to run contract tests."
        )

    try:
        yield service
    finally:
        product_index = elasticsearch_store.resolve_index_name("products")
        review_index = elasticsearch_store.resolve_index_name("reviews")
        elasticsearch_store._client.indices.delete(index=product_index, ignore_unavailable=True)
        elasticsearch_store._client.indices.delete(index=review_index, ignore_unavailable=True)


def test_contract_search_result_shape_is_consistent(knowledge_service) -> None:
    knowledge_service.upsert_products(
        [
            {
                "product_id": "P001",
                "name": "轻薄办公本",
                "category": "笔记本电脑",
                "description": "面向办公场景的轻薄本，续航能力突出。",
                "price": 5999,
                "currency": "CNY",
                "specs": {"cpu": "Intel Core Ultra 5", "memory": "16GB"},
                "inventory": {"status": "in_stock", "quantity": 23, "warehouse": "HZ-1"},
            },
            {
                "product_id": "P007",
                "name": "头戴式降噪耳机",
                "category": "耳机",
                "description": "通勤降噪明显，支持多设备连接。",
                "price": 1299,
                "currency": "CNY",
                "specs": {"battery": "40h"},
                "inventory": {"status": "in_stock", "quantity": 18, "warehouse": "HZ-2"},
            },
        ]
    )

    results = knowledge_service.search_products("办公 轻薄 续航 笔记本", top_k=2)
    assert results
    assert all(result.document.id for result in results)
    assert all(isinstance(result.document.content, str) for result in results)
    assert all(isinstance(result.document.metadata, dict) for result in results)
    assert all(result.score is None or isinstance(result.score, float) for result in results)
    assert any(result.document.id == "P001" for result in results)


def test_contract_filter_and_access_pattern_are_consistent(knowledge_service) -> None:
    knowledge_service.upsert_reviews(
        [
            {
                "review_id": "R001",
                "product_id": "P007",
                "rating": 5,
                "title": "降噪明显",
                "content": "地铁里听播客很清晰，降噪体验很好。",
                "user_name": "Alice",
                "created_at": "2026-04-20T10:00:00+08:00",
            },
            {
                "review_id": "R002",
                "product_id": "P008",
                "rating": 4,
                "title": "佩戴舒适",
                "content": "长时间开会佩戴压力不大。",
                "user_name": "Bob",
                "created_at": "2026-04-20T12:00:00+08:00",
            },
        ]
    )

    filtered = knowledge_service.search_reviews(
        "降噪 地铁",
        top_k=5,
        filters={"product_id": "P007"},
    )
    assert len(filtered) == 1
    assert filtered[0].document.id == "R001"
    assert filtered[0].document.metadata["product_id"] == "P007"


def test_contract_delete_access_is_consistent(knowledge_service) -> None:
    knowledge_service.upsert_products(
        [
            {
                "product_id": "P009",
                "name": "机械键盘",
                "category": "键盘",
                "description": "支持热插拔，回弹清晰。",
                "price": 499,
                "currency": "CNY",
                "specs": {"layout": "87"},
                "inventory": {"status": "in_stock", "quantity": 9, "warehouse": "HZ-3"},
            }
        ]
    )

    knowledge_service.delete_documents("products", ["P009"])
    remaining = knowledge_service.search_products("机械 键盘 回弹", top_k=5)
    assert all(result.document.id != "P009" for result in remaining)


def test_contract_chroma_preserves_system_document_indexes_with_dynamic_knowledge_sources() -> None:
    tmp_path = make_test_runtime_dir("knowledge-contract-dynamic-chroma")
    app_settings = AppSettings(
        data_dir=DATA_DIR,
        vector_store=VectorStoreConfig(
            provider="chroma",
            knowledge_sources={
                "catalog": {
                    "collection_name": "scene_catalog",
                    "index_name": "scene-catalog",
                }
            },
            documents={"index_name": "documents"},
            chunks={"index_name": "chunks"},
            chroma={"persist_directory": tmp_path / ".chroma"},
        ),
    )
    store = ChromaVectorStore(app_settings)

    store.ensure_collections()
    store.ensure_document_indexes()

    assert store.resolve_namespace_config("catalog").collection_name == "scene_catalog"
    assert store.resolve_document_collection_name("documents") == "knowledge_documents"
    assert store.resolve_document_collection_name("chunks") == "knowledge_chunks"


class _StoreBackedKnowledgeService:
    def __init__(self, store) -> None:
        self.store = store
        self.store.ensure_collections()

    def upsert_products(self, products: list[dict[str, object]]) -> None:
        self.store.upsert_documents(
            "products",
            [
                VectorStoreDocument(
                    id=str(product["product_id"]),
                    content=f'{product["name"]} {product["description"]}',
                    metadata={"product_id": product["product_id"]},
                )
                for product in products
            ],
        )

    def search_products(self, query: str, top_k: int | None = None):
        return self.store.search("products", query, top_k=top_k)

    def upsert_reviews(self, reviews: list[dict[str, object]]) -> None:
        self.store.upsert_documents(
            "reviews",
            [
                VectorStoreDocument(
                    id=str(review["review_id"]),
                    content=f'{review["title"]} {review["content"]}',
                    metadata={"product_id": review["product_id"], "review_id": review["review_id"]},
                )
                for review in reviews
            ],
        )

    def search_reviews(self, query: str, top_k: int | None = None, filters: dict[str, object] | None = None):
        return self.store.search("reviews", query, top_k=top_k, filters=filters)

    def delete_documents(self, namespace: str, ids: list[str]) -> None:
        self.store.delete_documents(namespace, ids)

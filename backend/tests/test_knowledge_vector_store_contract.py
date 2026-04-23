from __future__ import annotations

from uuid import uuid4

import pytest

from backend.config.settings import AppSettings, VectorStoreConfig
from backend.knowledge.service import KnowledgeService
from backend.knowledge.store import ElasticsearchVectorStore
from backend.tests.test_support import DATA_DIR, make_test_runtime_dir


@pytest.fixture(params=["chroma", "elasticsearch"])
def knowledge_service(request: pytest.FixtureRequest) -> KnowledgeService:
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
        yield KnowledgeService(app_settings=app_settings)
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
    service = KnowledgeService(app_settings=app_settings)
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


def test_contract_search_result_shape_is_consistent(knowledge_service: KnowledgeService) -> None:
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


def test_contract_filter_and_access_pattern_are_consistent(knowledge_service: KnowledgeService) -> None:
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


def test_contract_delete_access_is_consistent(knowledge_service: KnowledgeService) -> None:
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

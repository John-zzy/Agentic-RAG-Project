from __future__ import annotations

import math
from typing import Any
from uuid import uuid4

import pytest

from backend.config.settings import AppSettings, VectorStoreConfig
from backend.knowledge.base.store import ElasticsearchVectorStore, VectorStoreDocument, VectorStoreFactory
from backend.knowledge.ecommerce.loader import preload_knowledge_base
from backend.tests.test_support import DATA_DIR
import backend.knowledge.base.store as store_module


class FakeElasticsearchIndicesClient:
    def __init__(self, owner: "FakeElasticsearchClient") -> None:
        self.owner = owner
        self.mappings: dict[str, dict[str, Any]] = {}
        self.settings: dict[str, dict[str, Any]] = {}

    def exists(self, index: str) -> bool:
        return index in self.owner.documents

    def create(self, index: str, mappings: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
        self.owner.documents.setdefault(index, {})
        self.mappings[index] = mappings
        self.settings[index] = settings
        return {"acknowledged": True}


class FakeElasticsearchClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.indices = FakeElasticsearchIndicesClient(self)
        self.documents: dict[str, dict[str, dict[str, Any]]] = {}

    def ping(self) -> bool:
        return True

    def bulk(self, operations: list[dict[str, Any]], refresh: bool = True) -> dict[str, Any]:
        assert refresh is True
        items: list[dict[str, Any]] = []
        has_errors = False
        idx = 0
        while idx < len(operations):
            op = operations[idx]
            if "index" in op:
                action = op["index"]
                source = operations[idx + 1] if idx + 1 < len(operations) else {}
                idx += 2
                index_name = action["_index"]
                document_id = action["_id"]
                self.documents.setdefault(index_name, {})[document_id] = source
                items.append({"index": {"_id": document_id, "status": 201}})
            elif "delete" in op:
                action = op["delete"]
                idx += 1
                index_name = action["_index"]
                document_id = action["_id"]
                if index_name in self.documents and document_id in self.documents[index_name]:
                    del self.documents[index_name][document_id]
                    items.append({"delete": {"_id": document_id, "status": 200}})
                else:
                    items.append({"delete": {"_id": document_id, "status": 404, "error": "not_found"}})
                    has_errors = True
            else:
                idx += 1
        return {"errors": has_errors, "items": items}

    def search(
        self,
        *,
        index: str,
        query: dict[str, Any],
        size: int,
        source: list[str] | None = None,
    ) -> dict[str, Any]:
        del source
        script_score = query["script_score"]
        query_vector = script_score["script"]["params"]["query_vector"]
        filtered_documents = self._filter_documents(index, script_score["query"])

        hits: list[dict[str, Any]] = []
        for document_id, document in filtered_documents:
            score = 1.0 + self._cosine_similarity(query_vector, document["embedding"])
            hits.append(
                {
                    "_id": document_id,
                    "_score": score,
                    "_source": {
                        "content": document["content"],
                        "metadata": document["metadata"],
                        "namespace": document["namespace"],
                    },
                }
            )

        hits.sort(key=lambda hit: hit["_score"], reverse=True)
        return {"hits": {"hits": hits[:size]}}

    def delete(self, *, index: str, id: str, refresh: bool = True) -> dict[str, Any]:
        assert refresh is True
        self.documents.setdefault(index, {}).pop(id, None)
        return {"result": "deleted"}

    def _filter_documents(
        self,
        index: str,
        base_query: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        documents = list(self.documents.get(index, {}).items())
        if "match_all" in base_query:
            return documents

        filters = base_query.get("bool", {}).get("filter", [])
        results: list[tuple[str, dict[str, Any]]] = []
        for document_id, document in documents:
            metadata = document.get("metadata", {})
            if all(self._matches_filter(metadata, filter_clause) for filter_clause in filters):
                results.append((document_id, document))
        return results

    def _matches_filter(self, metadata: dict[str, Any], filter_clause: dict[str, Any]) -> bool:
        field, expected = next(iter(filter_clause["term"].items()))
        if not field.startswith("metadata."):
            return False
        metadata_key = field.removeprefix("metadata.")
        return metadata.get(metadata_key) == expected

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)


class FakeElasticsearchFactory:
    def __init__(self) -> None:
        self.instances: list[FakeElasticsearchClient] = []

    def __call__(self, **kwargs: Any) -> FakeElasticsearchClient:
        client = FakeElasticsearchClient(**kwargs)
        self.instances.append(client)
        return client


def build_elasticsearch_settings() -> AppSettings:
    return AppSettings(
        data_dir=DATA_DIR,
        vector_store=VectorStoreConfig(
            provider="elasticsearch",
            elasticsearch={
                "url": "http://localhost:9200",
                "index_prefix": "ai-rag",
            },
        ),
    )


def build_live_elasticsearch_settings(index_prefix: str) -> AppSettings:
    return AppSettings(
        data_dir=DATA_DIR,
        vector_store=VectorStoreConfig(
            provider="elasticsearch",
            products={
                "collection_name": "products",
                "index_name": "products",
            },
            reviews={
                "collection_name": "reviews",
                "index_name": "reviews",
            },
            elasticsearch={
                "url": "http://localhost:9200",
                "index_prefix": index_prefix,
                "verify_certs": False,
            },
        ),
    )


def test_elasticsearch_store_initializes_indexes_and_healthcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_factory = FakeElasticsearchFactory()
    monkeypatch.setattr(store_module, "Elasticsearch", fake_factory)

    app_settings = build_elasticsearch_settings()
    store = ElasticsearchVectorStore(app_settings)
    health = store.healthcheck()

    assert health.available is True

    store.ensure_collections()
    fake_client = fake_factory.instances[-1]

    assert fake_client.indices.exists(index="ai-rag-products")
    assert fake_client.indices.exists(index="ai-rag-reviews")
    assert (
        fake_client.indices.mappings["ai-rag-products"]["properties"]["embedding"]["type"]
        == "dense_vector"
    )


def test_elasticsearch_store_supports_upsert_search_filter_and_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_factory = FakeElasticsearchFactory()
    monkeypatch.setattr(store_module, "Elasticsearch", fake_factory)

    app_settings = build_elasticsearch_settings()
    store = ElasticsearchVectorStore(app_settings)
    store.ensure_collections()

    store.upsert_documents(
        "products",
        [
            VectorStoreDocument(
                id="P001",
                content="轻薄办公笔记本，续航出色，适合会议和文档处理。",
                metadata={"product_id": "P001", "category": "笔记本电脑"},
            ),
            VectorStoreDocument(
                id="P007",
                content="头戴式无线耳机，主动降噪明显，适合通勤和地铁。",
                metadata={"product_id": "P007", "category": "耳机"},
            ),
        ],
    )

    product_results = store.search("products", "办公 续航 轻薄 笔记本", top_k=1)
    assert product_results
    assert product_results[0].document.id == "P001"
    assert product_results[0].score is not None

    store.upsert_documents(
        "reviews",
        [
            VectorStoreDocument(
                id="R001",
                content="降噪效果很好，地铁里很安静。",
                metadata={"product_id": "P007", "rating": 5},
            ),
            VectorStoreDocument(
                id="R002",
                content="佩戴舒适，适合长时间开会。",
                metadata={"product_id": "P008", "rating": 4},
            ),
        ],
    )

    review_results = store.search("reviews", "降噪 地铁", top_k=5, filters={"product_id": "P007"})
    assert len(review_results) == 1
    assert review_results[0].document.id == "R001"
    assert review_results[0].document.metadata["product_id"] == "P007"

    store.delete_documents("products", ["P001"])
    remaining = store.search("products", "办公 续航 轻薄 笔记本", top_k=2)
    assert all(result.document.id != "P001" for result in remaining)


def test_preload_knowledge_base_supports_elasticsearch_provider_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_factory = FakeElasticsearchFactory()
    monkeypatch.setattr(store_module, "Elasticsearch", fake_factory)

    app_settings = build_elasticsearch_settings()
    summary = preload_knowledge_base(app_settings=app_settings)

    assert summary.products_loaded == 20
    assert summary.reviews_loaded == 21

    fake_client = fake_factory.instances[-1]
    store = ElasticsearchVectorStore(app_settings, client=fake_client)
    product_results = store.search("products", "轻薄办公 笔记本 14英寸", top_k=3)
    review_results = store.search(
        "reviews",
        "降噪 地铁 效果明显",
        top_k=3,
        filters={"product_id": "P007"},
    )

    assert any(result.document.id == "P001" for result in product_results)
    assert any(result.document.metadata["product_id"] == "P007" for result in review_results)


@pytest.mark.integration
def test_factory_switch_to_elasticsearch_initializes_indexes() -> None:
    app_settings = build_live_elasticsearch_settings(index_prefix=f"ai-rag-switch-{uuid4().hex[:8]}")
    store = VectorStoreFactory.create(app_settings)
    assert isinstance(store, ElasticsearchVectorStore)

    product_index = store.resolve_index_name("products")
    review_index = store.resolve_index_name("reviews")

    try:
        health = store.healthcheck()
        if not health.available:
            pytest.skip("Elasticsearch is unavailable at http://localhost:9200.")

        store.ensure_collections()
        assert store._client.indices.exists(index=product_index)
        assert store._client.indices.exists(index=review_index)
    finally:
        store._client.indices.delete(index=product_index, ignore_unavailable=True)
        store._client.indices.delete(index=review_index, ignore_unavailable=True)


@pytest.mark.integration
def test_elasticsearch_store_live_roundtrip() -> None:
    app_settings = build_live_elasticsearch_settings(index_prefix=f"ai-rag-test-{uuid4().hex[:8]}")
    store = ElasticsearchVectorStore(app_settings)
    product_index = store.resolve_index_name("products")
    review_index = store.resolve_index_name("reviews")

    try:
        health = store.healthcheck()
        if not health.available:
            pytest.skip("Elasticsearch is unavailable at http://localhost:9200.")

        store.ensure_collections()
        assert store._client.indices.exists(index=product_index)
        assert store._client.indices.exists(index=review_index)

        store.upsert_documents(
            "products",
            [
                VectorStoreDocument(
                    id="P001",
                    content="lightweight office ultrabook with long battery life",
                    metadata={"product_id": "P001", "category": "laptop"},
                ),
                VectorStoreDocument(
                    id="P007",
                    content="wireless headphones with strong noise cancellation for subway commuting",
                    metadata={"product_id": "P007", "category": "headphones"},
                ),
            ],
        )

        product_results = store.search("products", "office ultrabook battery", top_k=2)
        assert product_results
        assert product_results[0].document.id == "P001"
        assert product_results[0].score is not None

        store.upsert_documents(
            "reviews",
            [
                VectorStoreDocument(
                    id="R001",
                    content="noise cancellation works great on the subway commute",
                    metadata={"product_id": "P007", "rating": 5},
                ),
                VectorStoreDocument(
                    id="R002",
                    content="comfortable to wear during long meetings",
                    metadata={"product_id": "P008", "rating": 4},
                ),
            ],
        )

        review_results = store.search(
            "reviews",
            "subway noise cancellation",
            top_k=5,
            filters={"product_id": "P007"},
        )
        assert len(review_results) == 1
        assert review_results[0].document.id == "R001"
        assert review_results[0].document.metadata["product_id"] == "P007"

        store.delete_documents("products", ["P001"])
        remaining = store.search("products", "office ultrabook battery", top_k=5)
        assert all(result.document.id != "P001" for result in remaining)
    finally:
        store._client.indices.delete(index=product_index, ignore_unavailable=True)
        store._client.indices.delete(index=review_index, ignore_unavailable=True)

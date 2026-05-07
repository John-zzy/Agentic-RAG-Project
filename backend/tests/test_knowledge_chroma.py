from backend.config.settings import AppSettings, VectorStoreConfig
from backend.knowledge.base.store import ChromaVectorStore, VectorStoreDocument, VectorStoreFactory
from backend.knowledge.ecommerce.loader import preload_knowledge_base
from backend.tests.test_support import DATA_DIR, make_test_runtime_dir


def build_test_settings(tmp_path) -> AppSettings:
    return AppSettings(
        data_dir=DATA_DIR,
        vector_store=VectorStoreConfig(
            provider="chroma",
            chroma={"persist_directory": tmp_path / ".chroma"},
        ),
    )


def test_factory_defaults_to_chroma_on_startup() -> None:
    tmp_path = make_test_runtime_dir("knowledge-default-chroma-startup")
    app_settings = AppSettings(
        data_dir=DATA_DIR,
        vector_store=VectorStoreConfig(
            chroma={"persist_directory": tmp_path / ".chroma"},
        ),
    )

    store = VectorStoreFactory.create(app_settings)
    assert isinstance(store, ChromaVectorStore)
    store.ensure_collections()

    health = store.healthcheck()
    assert health.provider == "chroma"
    assert health.available is True


def test_chroma_store_supports_upsert_search_and_delete() -> None:
    tmp_path = make_test_runtime_dir("knowledge-upsert-search-delete")
    app_settings = build_test_settings(tmp_path)
    store = ChromaVectorStore(app_settings)
    store.ensure_collections()

    documents = [
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
    ]
    store.upsert_documents("products", documents)

    results = store.search("products", "办公 续航 轻薄 笔记本", top_k=1)
    assert results
    assert results[0].document.id == "P001"
    assert "product_id" in results[0].document.metadata
    assert results[0].score is not None

    store.delete_documents("products", ["P001"])
    remaining = store.search("products", "办公 续航 轻薄 笔记本", top_k=2)
    assert all(result.document.id != "P001" for result in remaining)


def test_chroma_store_supports_filters_and_consistent_result_shape() -> None:
    tmp_path = make_test_runtime_dir("knowledge-filter-shape")
    app_settings = build_test_settings(tmp_path)
    store = ChromaVectorStore(app_settings)
    store.ensure_collections()

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

    results = store.search("reviews", "降噪 地铁", top_k=5, filters={"product_id": "P007"})
    assert len(results) == 1
    assert results[0].document.id == "R001"
    assert results[0].document.metadata["product_id"] == "P007"
    assert isinstance(results[0].document.content, str)


def test_chroma_store_supports_document_management_operations() -> None:
    tmp_path = make_test_runtime_dir("knowledge-chroma-document-management")
    app_settings = build_test_settings(tmp_path)
    store = ChromaVectorStore(app_settings)
    store.ensure_document_indexes()

    record = {
        "document_id": "doc-1",
        "namespace": "faq",
        "source_type": "json",
        "source_path": "orders.json",
        "status": "active",
        "active_version": 1,
        "chunk_count": 1,
        "chunk_size": 120,
        "chunk_overlap": 20,
        "created_at": "2026-05-06T12:00:00Z",
        "updated_at": "2026-05-06T12:00:00Z",
        "last_error": None,
        "versions": [],
    }
    store.upsert_document_record(record)
    store.upsert_document_chunks(
        [
            VectorStoreDocument(
                id="chunk-1",
                content="订单状态：已付款",
                metadata={
                    "document_id": "doc-1",
                    "document_version": 1,
                    "namespace": "faq",
                    "source_type": "json",
                    "source_path": "orders.json",
                    "chunk_id": "chunk-1",
                    "chunk_index": 0,
                    "updated_at": "2026-05-06T12:00:00Z",
                    "is_active": True,
                },
            )
        ]
    )

    assert store.get_document_record("doc-1")["source_path"] == "orders.json"
    assert [document["document_id"] for document in store.list_document_records(namespace="faq")] == ["doc-1"]

    store.deactivate_document_chunks("doc-1", document_version=1)
    chunks = store._get_document_collection("chunks").get(ids=["chunk-1"])
    assert chunks["metadatas"][0]["is_active"] is False

    store.activate_document_chunks("doc-1", document_version=1)
    chunks = store._get_document_collection("chunks").get(ids=["chunk-1"])
    assert chunks["metadatas"][0]["is_active"] is True

    store.delete_document_record("doc-1")
    assert store.get_document_record("doc-1") is None
    assert store.list_document_records() == []


def test_chroma_document_chunk_search_supports_hybrid_keyword_recall() -> None:
    tmp_path = make_test_runtime_dir("knowledge-chroma-hybrid-document-search")
    app_settings = build_test_settings(tmp_path)
    store = ChromaVectorStore(app_settings)
    store.ensure_document_indexes()
    store.upsert_document_chunks(
        [
            VectorStoreDocument(
                id="chunk-doc-1",
                content="我是Ai Agent的文档：我叫zzy",
                metadata={
                    "document_id": "doc-1",
                    "document_version": 1,
                    "namespace": "faq",
                    "source_type": "txt",
                    "source_path": "doc.txt",
                    "chunk_id": "chunk-doc-1",
                    "chunk_index": 0,
                    "updated_at": "2026-05-07T12:00:00Z",
                    "is_active": True,
                },
            )
        ]
    )

    results = store.search_document_chunks("你叫什么", top_k=3, namespace="faq")
    assert results
    assert results[0].document.id == "chunk-doc-1"
    assert "zzy" in results[0].document.content


def test_preload_knowledge_base_uses_factory_and_loads_json_data() -> None:
    tmp_path = make_test_runtime_dir("knowledge-preload")
    app_settings = build_test_settings(tmp_path)
    summary = preload_knowledge_base(app_settings=app_settings)

    assert summary.products_loaded == 20
    assert summary.reviews_loaded == 21

    store = VectorStoreFactory.create(app_settings)
    product_results = store.search("products", "轻薄办公 笔记本 14英寸", top_k=3)
    review_results = store.search("reviews", "降噪 地铁 效果明显", top_k=3)

    assert product_results
    assert any(result.document.id == "P001" for result in product_results)
    assert any(result.document.metadata["product_id"] == "P007" for result in review_results)

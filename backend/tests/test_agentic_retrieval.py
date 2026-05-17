from backend.platform.config.settings import AppSettings
from backend.scenes.ecommerce.definition import create_agentic_knowledge_retriever
from backend.scenes.generic_assistant.definition import build_generic_assistant_scene_definition
from backend.scenes.ecommerce.definition import build_ecommerce_scene_definition
from backend.platform.knowledge.base.store import VectorSearchResult, VectorStoreDocument
from backend.tests.test_support import DATA_DIR, make_test_runtime_dir


def _build_settings(test_name: str) -> AppSettings:
    runtime_dir = make_test_runtime_dir(test_name)
    return AppSettings(
        data_dir=DATA_DIR,
        vector_store={
            "provider": "chroma",
            "chroma": {"persist_directory": runtime_dir / ".chroma"},
        },
    )


class FakeKnowledgeService:
    def __init__(self) -> None:
        self._products: list[VectorSearchResult] = []
        self._reviews: list[VectorSearchResult] = []
        self._orders: list[VectorSearchResult] = []
        self._documents: list[VectorSearchResult] = []

    def upsert_products(self, products: list[dict[str, object]]) -> None:
        self._products = [
            VectorSearchResult(
                document=VectorStoreDocument(
                    id=str(product["product_id"]),
                    content=f'{product["name"]} {product["description"]}',
                    metadata={"product_id": product["product_id"]},
                ),
                score=0.95,
            )
            for product in products
        ]

    def upsert_reviews(self, reviews: list[dict[str, object]]) -> None:
        self._reviews = [
            VectorSearchResult(
                document=VectorStoreDocument(
                    id=str(review["review_id"]),
                    content=str(review["content"]),
                    metadata={"product_id": review["product_id"], "review_id": review["review_id"]},
                ),
                score=0.88,
            )
            for review in reviews
        ]

    def upsert_documents(self, documents: list[dict[str, object]]) -> None:
        self._documents = [
            VectorSearchResult(
                document=VectorStoreDocument(
                    id=str(document["document_id"]),
                    content=str(document["content"]),
                    metadata={
                        "document_id": document["document_id"],
                        "source_path": document.get("source_path", f'{document["document_id"]}.md'),
                        "namespace": "documents",
                    },
                ),
                score=float(document.get("score", 0.9)),
            )
            for document in documents
        ]

    def search_products(self, query: str, top_k: int | None = None):
        del top_k
        query_lower = query.lower()
        return [item for item in self._products if str(item.document.id).lower() in query_lower or "aerophone x" in query_lower]

    def search_reviews(self, query: str, top_k: int | None = None):
        del query, top_k
        return list(self._reviews)

    def search_orders(self, query: str, top_k: int | None = None):
        del query, top_k
        return list(self._orders)

    def search_document_chunks(self, query: str, top_k: int | None = None, namespace: str | None = None):
        del query, top_k, namespace
        return list(self._documents)


def _build_knowledge_service(test_name: str) -> tuple[AppSettings, FakeKnowledgeService]:
    app_settings = _build_settings(test_name)
    knowledge_service = FakeKnowledgeService()
    knowledge_service.upsert_products(
        [
            {
                "product_id": "P005",
                "name": "AeroPhone X",
                "category": "智能手机",
                "description": "旗舰 5G 手机，主打影像和高刷屏，电池容量 5000mAh。",
                "price": 4599,
                "currency": "CNY",
                "specs": {"battery": "5000mAh", "camera": "50MP", "display": "120Hz"},
                "inventory": {"status": "in_stock", "quantity": 12, "warehouse": "SH-1"},
            }
        ]
    )
    knowledge_service.upsert_reviews(
        [
            {
                "review_id": "R005",
                "product_id": "P005",
                "rating": 5,
                "title": "续航稳定",
                "content": "重度使用一天也够用，拍照效果也很好。",
                "user_name": "Alice",
                "created_at": "2026-04-20T10:00:00+08:00",
            }
        ]
    )
    knowledge_service.upsert_documents(
        [
            {
                "document_id": "DOC-001",
                "content": "AeroPhone X 产品手册：电池 5000mAh，屏幕 120Hz，价格 4599 元。",
                "source_path": "manuals/aerophone-x.md",
                "score": 0.93,
            },
            {
                "document_id": "DOC-002",
                "content": "售后 FAQ：订单查询需要提供订单号，库存问题以系统实时状态为准。",
                "source_path": "faq/after-sale.md",
                "score": 0.88,
            },
        ]
    )
    return app_settings, knowledge_service


def test_agentic_retriever_switches_to_inventory_tool_for_stock_query() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-inventory")
    retriever = create_agentic_knowledge_retriever(
        app_settings,
        knowledge_service=knowledge_service,
    )

    outcome = retriever.retrieve_with_trace("AeroPhone X 现在有货吗")

    assert outcome.documents
    assert outcome.exit_reason == "sufficient"
    assert [entry.tool_name for entry in outcome.decision_log] == [
        "knowledge_document_search",
        "product_semantic_search",
        "inventory_lookup",
    ]
    assert outcome.decision_log[2].query == "P005"


def test_agentic_retriever_returns_detail_lookup_for_spec_question() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-detail")
    retriever = create_agentic_knowledge_retriever(
        app_settings,
        knowledge_service=knowledge_service,
    )

    outcome = retriever.retrieve_with_trace("AeroPhone X 的参数和价格是什么")

    assert outcome.documents
    assert any(doc.metadata.get("namespace") == "product_detail" for doc in outcome.documents)
    assert outcome.decision_log[-1].tool_name == "product_detail_lookup"
    assert outcome.decision_log[0].tool_name == "knowledge_document_search"


def test_ecommerce_scene_definition_builds_agentic_retriever_and_scene_metadata() -> None:
    app_settings, knowledge_service = _build_knowledge_service("scene-ecommerce-retriever")
    definition = build_ecommerce_scene_definition(
        app_settings=app_settings,
        knowledge_service=knowledge_service,
    )

    retriever = definition.build_retriever()
    outcome = retriever.retrieve_with_trace("AeroPhone X 现在有货吗")

    assert definition.scene == "ecommerce"
    assert definition.metadata["supports_agentic_retrieval"] is True
    assert outcome.decision_log[0].tool_name == "knowledge_document_search"
    assert outcome.decision_log[1].tool_name == "product_semantic_search"
    assert outcome.decision_log[2].tool_name == "inventory_lookup"


def test_generic_scene_definition_only_uses_document_knowledge_and_generic_fallback() -> None:
    app_settings = _build_settings("scene-generic-retriever")
    definition = build_generic_assistant_scene_definition(app_settings=app_settings)

    tool_names = {tool.name for tool in definition.build_tools()}

    assert definition.scene == "generic_assistant"
    assert "product_semantic_search" not in tool_names
    assert "inventory_lookup" not in tool_names
    assert "商品" not in definition.fallback_policy.no_hit_message


def test_agentic_retriever_stays_on_documents_for_document_question() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-documents-first")
    retriever = create_agentic_knowledge_retriever(
        app_settings,
        knowledge_service=knowledge_service,
    )

    outcome = retriever.retrieve_with_trace("请根据产品手册说明 AeroPhone X 的价格和电池参数")

    assert outcome.documents
    assert [entry.tool_name for entry in outcome.decision_log] == ["knowledge_document_search"]
    assert all(
        str(document.metadata.get("namespace")) == "documents"
        for document in outcome.documents
    )


def test_agentic_retriever_restricts_to_documents_only_candidate_tools() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-documents-only")
    retriever = create_agentic_knowledge_retriever(
        app_settings,
        knowledge_service=knowledge_service,
    )

    outcome = retriever.retrieve_with_trace(
        "AeroPhone X 现在有货吗",
        candidate_tools=("knowledge_document_search",),
    )

    assert outcome.documents
    assert [entry.tool_name for entry in outcome.decision_log] == ["knowledge_document_search"]

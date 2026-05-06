from backend.config.settings import AppSettings
from backend.knowledge.ecommerce.retriever import create_agentic_knowledge_retriever
from backend.knowledge.ecommerce.service import KnowledgeService
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


def _build_knowledge_service(test_name: str) -> tuple[AppSettings, KnowledgeService]:
    app_settings = _build_settings(test_name)
    knowledge_service = KnowledgeService(app_settings=app_settings)
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
        "product_semantic_search",
        "inventory_lookup",
    ]
    assert outcome.decision_log[1].query == "P005"


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

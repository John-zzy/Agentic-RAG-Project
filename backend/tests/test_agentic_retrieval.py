import json
from pathlib import Path
import warnings

from backend.application.runtime.service import build_default_scene_registry
from backend.platform.config.settings import AppSettings
from backend.platform.rag.document_retrieval_rules import DOCUMENT_MINIMUM_RELEVANCE
from backend.platform.rag.core import RetrievalContext, SufficiencyDecision
from backend.platform.retrieval import VectorSearchResult, VectorStoreDocument
from backend.platform.rag.document_retrieval import DocumentChunkRetrievalResult
from backend.scenes.ecommerce.definition import create_agentic_knowledge_retriever
from backend.scenes.ecommerce.retrieval_tools import build_semantic_retrieval_tool
from backend.scenes.generic_assistant.definition import (
    GENERIC_ASSISTANT_RETRIEVAL_POLICY,
    GenericAssistantBusinessExtension,
    build_generic_assistant_scene_definition,
)
from backend.scenes.ecommerce.definition import ECOMMERCE_RETRIEVAL_POLICY, build_ecommerce_scene_definition
from backend.tests.test_support import DATA_DIR, make_test_runtime_dir


def _build_settings(test_name: str) -> AppSettings:
    runtime_dir = make_test_runtime_dir(test_name)
    files_root = runtime_dir / "files"
    (files_root / "manuals").mkdir(parents=True, exist_ok=True)
    (files_root / "faq").mkdir(parents=True, exist_ok=True)
    (files_root / "manuals" / "aerophone-x.md").write_text("AeroPhone X 产品手册", encoding="utf-8")
    (files_root / "faq" / "after-sale.md").write_text("售后 FAQ", encoding="utf-8")
    return AppSettings(
        data_dir=runtime_dir,
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
                        "is_managed_document": True,
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


class FakeDocumentRetrievalService:
    def __init__(self, knowledge_service: FakeKnowledgeService) -> None:
        self._knowledge_service = knowledge_service
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        minimum_relevance: float | None = None,
    ) -> list[DocumentChunkRetrievalResult]:
        del query, namespace
        self.calls.append({"top_k": top_k, "minimum_relevance": minimum_relevance})
        results = self._knowledge_service._documents
        if minimum_relevance is not None:
            results = [
                result
                for result in results
                if result.score is None or float(result.score) >= minimum_relevance
            ]
        return [
            DocumentChunkRetrievalResult(
                document=result.document,
                score=result.score,
                vector_score=result.score,
                vector_rank=index,
                matched_by=["vector"],
            )
            for index, result in enumerate(results[:top_k], start=1)
        ]


class FakeOrderRoutingExtension(GenericAssistantBusinessExtension):
    knowledge_source = "ecommerce"

    def __init__(self, knowledge_service: FakeKnowledgeService) -> None:
        self._knowledge_service = knowledge_service

    @property
    def retrieval_tool_names(self) -> tuple[str, ...]:
        return ("order_semantic_search",)

    def build_retrieval_tools(self):
        return (
            build_semantic_retrieval_tool(
                self._knowledge_service,
                namespace="orders",
                tool_name="order_semantic_search",
                description="Search order information semantically for tracking queries.",
            ),
        )

    def should_handoff(self, context: RetrievalContext) -> SufficiencyDecision | None:
        query = context.plan.user_query.lower()
        if context.results[-1].records:
            return None
        if "订单" in query or "tracking" in query or "物流" in query or "追踪号" in query:
            return SufficiencyDecision(
                is_sufficient=False,
                next_action="switch_tool",
                reason="问题需要切换到订单知识源。",
                suggested_tool="order_semantic_search",
            )
        return None

    def resolve_followup(self, context: RetrievalContext) -> SufficiencyDecision | None:
        result = context.results[-1]
        if result.records:
            return SufficiencyDecision(
                is_sufficient=True,
                next_action="finish",
                reason="订单证据已足够支持回答。",
                confidence=result.confidence,
            )
        return None


def _build_knowledge_service(test_name: str) -> tuple[AppSettings, FakeKnowledgeService]:
    app_settings = _build_settings(test_name)
    knowledge_service = FakeKnowledgeService()
    products = [
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
    (app_settings.data_dir / "products.json").write_text(
        json.dumps(products, ensure_ascii=False),
        encoding="utf-8",
    )
    knowledge_service.upsert_products(products)
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


def _build_ecommerce_retriever(
    app_settings: AppSettings,
    knowledge_service: FakeKnowledgeService,
):
    return build_ecommerce_scene_definition(
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
    ).build_retriever()


def test_agentic_retriever_switches_to_inventory_tool_for_stock_query() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-inventory")
    retriever = _build_ecommerce_retriever(app_settings, knowledge_service)

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
    retriever = _build_ecommerce_retriever(app_settings, knowledge_service)

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
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
    )

    retriever = definition.build_retriever()
    outcome = retriever.retrieve_with_trace("AeroPhone X 现在有货吗")

    assert definition.scene == "ecommerce"
    assert definition.metadata["supports_agentic_retrieval"] is True
    assert outcome.decision_log[0].tool_name == "knowledge_document_search"
    assert outcome.decision_log[1].tool_name == "product_semantic_search"
    assert outcome.decision_log[2].tool_name == "inventory_lookup"


def test_scene_definitions_declare_explicit_retrieval_policies() -> None:
    app_settings, knowledge_service = _build_knowledge_service("scene-retrieval-policy-defaults")
    generic_definition = build_generic_assistant_scene_definition(
        app_settings=app_settings,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
    )
    ecommerce_definition = build_ecommerce_scene_definition(
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
    )

    assert generic_definition.retrieval_policy == GENERIC_ASSISTANT_RETRIEVAL_POLICY
    assert ecommerce_definition.retrieval_policy == ECOMMERCE_RETRIEVAL_POLICY
    for policy in (generic_definition.retrieval_policy, ecommerce_definition.retrieval_policy):
        assert policy.top_k == 5
        assert policy.min_relevance_score == DOCUMENT_MINIMUM_RELEVANCE
        assert policy.recall_strategy == "hybrid"
        assert policy.no_hit_strategy == "ask_user"
        assert policy.rerank_enabled is False
        assert policy.rerank_top_n is None


def test_ecommerce_scene_definition_resolves_candidate_tools_from_generic_docs_first_chain() -> None:
    app_settings, knowledge_service = _build_knowledge_service("scene-ecommerce-candidate-tools")
    definition = build_ecommerce_scene_definition(
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
    )

    assert definition.resolve_candidate_retrieval_tools(("documents",)) == (
        "knowledge_document_search",
    )
    assert definition.resolve_candidate_retrieval_tools(("documents", "ecommerce")) == (
        "knowledge_document_search",
        "product_semantic_search",
        "review_semantic_search",
        "order_semantic_search",
        "inventory_lookup",
        "product_detail_lookup",
    )


def test_generic_scene_definition_only_uses_document_knowledge_and_generic_fallback() -> None:
    app_settings = _build_settings("scene-generic-retriever")
    definition = build_generic_assistant_scene_definition(app_settings=app_settings)

    tool_names = {tool.name for tool in definition.build_tools()}

    assert definition.scene == "generic_assistant"
    assert "product_semantic_search" not in tool_names
    assert "inventory_lookup" not in tool_names
    assert "商品" not in definition.fallback_policy.no_hit_message


def test_generic_scene_definition_resolves_docs_only_candidate_tools() -> None:
    app_settings = _build_settings("scene-generic-candidate-tools")
    definition = build_generic_assistant_scene_definition(app_settings=app_settings)

    assert definition.resolve_candidate_retrieval_tools(("documents",)) == (
        "knowledge_document_search",
    )


def test_generic_scene_definition_includes_extension_tools_only_when_mounted() -> None:
    app_settings, knowledge_service = _build_knowledge_service("scene-generic-extension-candidate-tools")
    definition = build_generic_assistant_scene_definition(
        app_settings=app_settings,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
        business_extensions=(FakeOrderRoutingExtension(knowledge_service),),
    )

    assert definition.resolve_candidate_retrieval_tools(("documents",)) == (
        "knowledge_document_search",
    )
    assert definition.resolve_candidate_retrieval_tools(("documents", "ecommerce")) == (
        "knowledge_document_search",
        "order_semantic_search",
    )


def test_generic_definition_does_not_import_ecommerce_default_routing_symbols() -> None:
    source = Path("backend/scenes/generic_assistant/definition.py").read_text(encoding="utf-8")

    assert "EcommerceSufficiencyJudge" not in source
    assert "EcommerceQueryRewriter" not in source
    assert "build_agentic_retrieval_tools" not in source


def test_default_scene_registry_injects_ecommerce_extension_into_generic_scene() -> None:
    app_settings, knowledge_service = _build_knowledge_service("scene-registry-generic-extension")
    registry = build_default_scene_registry(
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
    )

    generic_definition = registry.get_definition("generic_assistant")

    assert generic_definition.resolve_candidate_retrieval_tools(("documents",)) == (
        "knowledge_document_search",
    )
    assert generic_definition.resolve_candidate_retrieval_tools(("documents", "ecommerce")) == (
        "knowledge_document_search",
        "product_semantic_search",
        "review_semantic_search",
        "order_semantic_search",
        "inventory_lookup",
        "product_detail_lookup",
    )


def test_default_scene_registry_can_disable_default_business_extensions() -> None:
    app_settings, knowledge_service = _build_knowledge_service("scene-registry-without-default-extension")
    registry = build_default_scene_registry(
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
        include_default_business_extensions=False,
    )

    generic_definition = registry.get_definition("generic_assistant")

    assert generic_definition.resolve_candidate_retrieval_tools(("documents",)) == (
        "knowledge_document_search",
    )


def test_create_agentic_knowledge_retriever_emits_deprecation_warning() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-retriever-deprecation")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        retriever = create_agentic_knowledge_retriever(
            app_settings=app_settings,
            knowledge_service=knowledge_service,
            document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
        )

    assert retriever is not None
    assert any(item.category is DeprecationWarning for item in captured)


def test_agentic_retriever_stays_on_documents_for_document_question() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-documents-first")
    retriever = _build_ecommerce_retriever(app_settings, knowledge_service)

    outcome = retriever.retrieve_with_trace("请根据产品手册说明 AeroPhone X 的价格和电池参数")

    assert outcome.documents
    assert [entry.tool_name for entry in outcome.decision_log] == ["knowledge_document_search"]
    assert all(
        str(document.metadata.get("namespace")) == "documents"
        for document in outcome.documents
    )


def test_generic_docs_sufficient_does_not_handoff_to_mounted_extension() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-docs-sufficient-no-handoff")
    definition = build_generic_assistant_scene_definition(
        app_settings=app_settings,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
        business_extensions=(FakeOrderRoutingExtension(knowledge_service),),
    )
    retriever = definition.build_retriever()

    outcome = retriever.retrieve_with_trace(
        "请根据产品手册说明 AeroPhone X 的价格和电池参数",
        candidate_tools=("knowledge_document_search", "order_semantic_search"),
    )

    assert outcome.documents
    assert [entry.tool_name for entry in outcome.decision_log] == ["knowledge_document_search"]


def test_generic_docs_only_empty_result_falls_back_without_business_switch() -> None:
    app_settings = _build_settings("agentic-docs-only-empty-result")
    definition = build_generic_assistant_scene_definition(app_settings=app_settings)
    retriever = definition.build_retriever()

    outcome = retriever.retrieve_with_trace(
        "订单 O202604210010 的物流状态是什么",
        candidate_tools=("knowledge_document_search",),
    )

    assert outcome.documents == []
    assert {entry.tool_name for entry in outcome.decision_log} == {"knowledge_document_search"}
    assert outcome.exit_reason in {"ask_user", "max_rounds_reached"}


def test_generic_docs_only_greeting_does_not_rewrite_to_document_terms() -> None:
    app_settings = _build_settings("agentic-docs-only-greeting-no-rewrite")
    definition = build_generic_assistant_scene_definition(app_settings=app_settings)
    retriever = definition.build_retriever()

    outcome = retriever.retrieve_with_trace(
        "你好",
        candidate_tools=("knowledge_document_search",),
    )

    assert outcome.documents == []
    assert outcome.exit_reason == "ask_user"
    assert len(outcome.decision_log) == 1
    assert outcome.decision_log[0].query == "你好"
    assert outcome.decision_log[0].rewritten_query is None
    assert outcome.decision_log[0].decision == "ask_user"


def test_agentic_retriever_restricts_to_documents_only_candidate_tools() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-documents-only")
    retriever = _build_ecommerce_retriever(app_settings, knowledge_service)

    outcome = retriever.retrieve_with_trace(
        "AeroPhone X 现在有货吗",
        candidate_tools=("knowledge_document_search",),
    )

    assert outcome.documents
    assert [entry.tool_name for entry in outcome.decision_log] == ["knowledge_document_search"]


def test_agentic_retriever_prioritizes_exact_order_match_for_tracking_query() -> None:
    app_settings = _build_settings("agentic-order-priority")
    knowledge_service = FakeKnowledgeService()
    knowledge_service._orders = [
        VectorSearchResult(
            document=VectorStoreDocument(
                id="O202604210010",
                content="订单 O202604210010，承运商申通快递，运单 ST0011223344CN，状态已签收。",
                metadata={
                    "order_id": "O202604210010",
                    "tracking_no": "ST0011223344CN",
                    "carrier": "申通快递",
                    "namespace": "orders",
                },
            ),
            score=0.96,
        )
    ]
    definition = build_generic_assistant_scene_definition(
        app_settings=app_settings,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
        business_extensions=(FakeOrderRoutingExtension(knowledge_service),),
    )
    retriever = definition.build_retriever()

    outcome = retriever.retrieve_with_trace(
        "订单 O202604210010 的物流状态是什么？追踪号 ST0011223344CN",
        candidate_tools=("knowledge_document_search", "order_semantic_search"),
    )

    assert outcome.documents
    assert [entry.tool_name for entry in outcome.decision_log] == [
        "knowledge_document_search",
        "order_semantic_search",
    ]
    assert outcome.documents[0].metadata.get("namespace") == "orders"
    assert outcome.documents[0].metadata.get("order_id") == "O202604210010"
    assert outcome.documents[0].metadata.get("tracking_no") == "ST0011223344CN"

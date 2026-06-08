import json
from pathlib import Path
import warnings

import pytest

from backend.application.runtime.service import build_default_scene_registry
from backend.platform.config.settings import AppSettings
from backend.platform.rag.retrieval.documents.filters import DOCUMENT_MINIMUM_RELEVANCE
from backend.platform.rag.contracts import RetrievalContext, RetrievalPlan, RetrievalResult
from backend.platform.rag.orchestration.decisions import SufficiencyDecision
from backend.platform.rag.post_retrieval import DashScopeRetrievalReranker, IdentityRetrievalReranker
from backend.platform.models.llm.guards import JsonSchemaGuard
from backend.platform.search_foundation import VectorSearchResult, VectorStoreDocument
from backend.platform.rag.retrieval.documents import DocumentChunkRetrievalResult
from backend.platform.tools import build_retrieval_tool
from backend.scenes.ecommerce.definition import create_agentic_knowledge_retriever
from backend.scenes.ecommerce.tools import OrderSemanticSearchTool, ProductCatalogStore, ProductSemanticSearchTool
from backend.scenes.generic_assistant.definition import (
    GENERIC_ASSISTANT_RETRIEVAL_POLICY,
    GenericAssistantBusinessExtension,
    GenericAssistantSufficiencyJudge,
    GenericAssistantQueryRewriter,
    build_generic_assistant_scene_definition,
)
from backend.scenes.base import SceneRetrievalPolicy
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
                        "namespace": document.get("namespace", "documents"),
                        "is_managed_document": True,
                    },
                ),
                score=float(document.get("score", 0.9)),
            )
            for document in documents
        ]

    def search_products(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, object] | None = None,
    ):
        del top_k, filters
        query_lower = query.lower()
        return [item for item in self._products if str(item.document.id).lower() in query_lower or "aerophone x" in query_lower]

    def search_reviews(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, object] | None = None,
    ):
        del query, top_k, filters
        return list(self._reviews)

    def search_orders(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, object] | None = None,
    ):
        del query, top_k, filters
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
        recall_strategy: str = "hybrid",
    ) -> list[DocumentChunkRetrievalResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "namespace": namespace,
                "minimum_relevance": minimum_relevance,
                "recall_strategy": recall_strategy,
            }
        )
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


class QueryMatchingFakeDocumentRetrievalService(FakeDocumentRetrievalService):
    def __init__(self, knowledge_service: FakeKnowledgeService, *, matching_query: str) -> None:
        super().__init__(knowledge_service)
        self._matching_query = matching_query

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        minimum_relevance: float | None = None,
        recall_strategy: str = "hybrid",
    ) -> list[DocumentChunkRetrievalResult]:
        if query != self._matching_query:
            self.calls.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "namespace": namespace,
                    "minimum_relevance": minimum_relevance,
                    "recall_strategy": recall_strategy,
                }
            )
            return []
        return super().retrieve(
            query=query,
            top_k=top_k,
            namespace=namespace,
            minimum_relevance=minimum_relevance,
            recall_strategy=recall_strategy,
        )


class FakeQueryRewriteModelClient:
    def __init__(self, output: str) -> None:
        self.output = output
        self.get_runnable_calls: list[str] = []
        self.invoke_runnable_calls: list[dict[str, object]] = []

    def get_runnable(
        self,
        complexity: str = "simple",
        prompt_template: object | None = None,
        *,
        output_parser: object | None = None,
    ) -> object:
        del prompt_template, output_parser
        self.get_runnable_calls.append(complexity)
        return object()

    def invoke_runnable(
        self,
        runnable: object,
        input: object,
        *,
        config: object | None = None,
    ) -> str:
        self.invoke_runnable_calls.append(
            {"runnable": runnable, "input": input, "config": config}
        )
        return self.output

    def invoke_json_schema(
        self,
        runnable: object,
        input: object,
        *,
        schema_model: type[object],
        schema_source: str,
        config: object | None = None,
        complexity: str = "unknown",
        metadata: dict[str, object] | None = None,
    ) -> object:
        del complexity, metadata
        raw_output = self.invoke_runnable(runnable, input, config=config)
        return JsonSchemaGuard().validate(
            raw_output,
            schema_model=schema_model,
            source=schema_source,
        )


class FailingQueryRewriteModelClient(FakeQueryRewriteModelClient):
    def __init__(self) -> None:
        super().__init__("")

    def invoke_runnable(
        self,
        runnable: object,
        input: object,
        *,
        config: object | None = None,
    ) -> str:
        self.invoke_runnable_calls.append(
            {"runnable": runnable, "input": input, "config": config}
        )
        raise RuntimeError("rewrite model unavailable")


class FakeOrderRoutingExtension(GenericAssistantBusinessExtension):
    knowledge_source = "ecommerce"

    def __init__(self, knowledge_service: FakeKnowledgeService) -> None:
        self._knowledge_service = knowledge_service

    @property
    def retrieval_tool_names(self) -> tuple[str, ...]:
        return ("order_semantic_search",)

    def build_retrieval_tools(self):
        return (
            build_retrieval_tool(OrderSemanticSearchTool(knowledge_service=self._knowledge_service)),
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


def test_generic_assistant_document_tool_searches_all_managed_document_namespaces() -> None:
    app_settings, knowledge_service = _build_knowledge_service("generic-document-source-namespaces")
    knowledge_service.upsert_documents(
        [
            {
                "document_id": "DOC-FAQ",
                "content": "sessions 表字段包括 session_id、scene、mounted_knowledge_sources、status、created_at。",
                "source_path": "data-model.md",
                "namespace": "faq",
                "score": 0.96,
            }
        ]
    )
    document_service = FakeDocumentRetrievalService(knowledge_service)
    definition = build_generic_assistant_scene_definition(
        app_settings=app_settings,
        document_retrieval_service=document_service,
    )

    outcome = definition.build_retriever().retrieve_with_trace(
        "当前 session 表有哪些字段？",
        candidate_tools=("knowledge_document_search",),
    )

    assert outcome.documents
    assert document_service.calls[0]["namespace"] is None
    assert outcome.documents[0].metadata["namespace"] == "faq"


def test_generic_query_rewriter_uses_llm_json_query() -> None:
    model = FakeQueryRewriteModelClient(
        '{"query":"session 表字段","reason":"保留 session 实体并聚焦字段检索"}'
    )
    rewriter = GenericAssistantQueryRewriter(model_client=model)
    rewrite = rewriter.invoke(
        RetrievalContext(
            plan=RetrievalPlan(
                user_query="当前有哪些表",
                active_query="当前有哪些表",
                selected_tool="knowledge_document_search",
            ),
            results=[
                RetrievalResult.ok(
                    tool_name="knowledge_document_search",
                    query="当前有哪些表",
                    records=[],
                    documents=[],
                )
            ],
        )
    )

    assert rewrite.query == "session 表字段"
    assert rewrite.reason == "保留 session 实体并聚焦字段检索"
    assert rewrite.metadata == {
        "original_query": "当前有哪些表",
        "strategy": "llm_json",
        "fallback": False,
        "fallback_reason": None,
        "preserved_tokens": [],
    }
    assert model.get_runnable_calls == ["simple"]
    assert model.invoke_runnable_calls[0]["input"] == {
        "original_query": "当前有哪些表",
        "active_query": "当前有哪些表",
        "retrieval_summary": (
            "tool=knowledge_document_search; query=当前有哪些表; record_count=0; "
            "document_count=0; success=True; error=none"
        ),
    }


def test_agentic_retriever_uses_accepted_llm_rewritten_query_for_next_round() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-llm-rewrite-next-query")
    knowledge_service.upsert_documents(
        [
            {
                "document_id": "DOC-SESSION",
                "content": "session 表字段包括 session_id、scene、mounted_knowledge_sources。",
                "source_path": "data-model.md",
                "namespace": "documents",
                "score": 0.94,
            }
        ]
    )
    document_service = QueryMatchingFakeDocumentRetrievalService(
        knowledge_service,
        matching_query="session 表字段",
    )
    definition = build_generic_assistant_scene_definition(
        app_settings=app_settings,
        document_retrieval_service=document_service,
    )
    retriever = definition.build_retriever()
    retriever.query_rewriter = GenericAssistantQueryRewriter(
        model_client=FakeQueryRewriteModelClient(
            '{"query":"session 表字段","reason":"聚焦 session 表字段"}'
        )
    )

    outcome = retriever.retrieve_with_trace(
        "当前有哪些表",
        candidate_tools=("knowledge_document_search",),
    )

    assert outcome.documents
    assert outcome.exit_reason == "sufficient"
    assert [call["query"] for call in document_service.calls] == [
        "当前有哪些表",
        "session 表字段",
    ]
    assert outcome.decision_log[0].rewritten_query == "session 表字段"
    assert outcome.rounds[0].rewrite is not None
    assert outcome.rounds[0].rewrite.metadata["fallback"] is False


def test_generic_query_rewriter_falls_back_for_invalid_json_empty_query_and_model_error() -> None:
    context = RetrievalContext(
        plan=RetrievalPlan(
            user_query="  当前   有哪些表  ",
            active_query="  当前   有哪些表  ",
            selected_tool="knowledge_document_search",
        ),
        results=[
            RetrievalResult.ok(
                tool_name="knowledge_document_search",
                query="当前有哪些表",
                records=[],
                documents=[],
            )
        ],
    )

    for model in (
        FakeQueryRewriteModelClient("这不是 JSON"),
        FakeQueryRewriteModelClient('{"query":"   ","reason":"empty"}'),
        FailingQueryRewriteModelClient(),
    ):
        rewrite = GenericAssistantQueryRewriter(model_client=model).invoke(context)

        assert rewrite.query == "当前 有哪些表"
        assert rewrite.metadata["fallback"] is True
        assert rewrite.metadata["original_query"] == "当前 有哪些表"
        assert rewrite.metadata["strategy"] == "llm_json"
        assert rewrite.metadata["preserved_tokens"] == []
        assert rewrite.metadata["fallback_reason"] in {
            "model_schema_error",
            "RuntimeError",
        }
        if isinstance(model, FailingQueryRewriteModelClient):
            assert rewrite.metadata["fallback_reason"] == "RuntimeError"
        else:
            assert rewrite.metadata["fallback_reason"] == "model_schema_error"
            assert rewrite.metadata["failure"]["category"] == "model_schema_error"
        assert model.get_runnable_calls == ["simple"]
        assert len(model.invoke_runnable_calls) == 1


def test_generic_query_rewriter_falls_back_when_llm_drops_preserved_tokens() -> None:
    cases = [
        (
            "VOID-ALPHA-7788 secret handshake?",
            '{"query":"secret handshake 排查","reason":"删除了错误码"}',
            "missing_preserved_token:VOID-ALPHA-7788",
        ),
        (
            "Python 3.11 安装要求是什么？",
            '{"query":"Python 安装要求","reason":"删除了版本号"}',
            "missing_preserved_token:3.11",
        ),
        (
            "MFA 策略要求是什么？",
            '{"query":"策略要求","reason":"删除了缩写"}',
            "missing_preserved_token:MFA",
        ),
    ]

    for original_query, model_output, fallback_reason in cases:
        context = RetrievalContext(
            plan=RetrievalPlan(
                user_query=original_query,
                active_query=original_query,
                selected_tool="knowledge_document_search",
            ),
            results=[
                RetrievalResult.ok(
                    tool_name="knowledge_document_search",
                    query=original_query,
                    records=[],
                    documents=[],
                )
            ],
        )

        rewrite = GenericAssistantQueryRewriter(
            model_client=FakeQueryRewriteModelClient(model_output)
        ).invoke(context)

        assert rewrite.query == original_query
        assert rewrite.metadata["fallback"] is True
        assert rewrite.metadata["fallback_reason"] == fallback_reason


def test_generic_query_rewriter_falls_back_for_unsupported_generic_expansion() -> None:
    cases = [
        '{"query":"火星基地 快递 数据模型","reason":"添加数据模型"}',
        '{"query":"火星基地 快递 表结构","reason":"添加表结构"}',
        '{"query":"火星基地 快递 常见问题","reason":"添加常见问题"}',
    ]

    for model_output in cases:
        original_query = "火星基地快递多久到"
        context = RetrievalContext(
            plan=RetrievalPlan(
                user_query=original_query,
                active_query=original_query,
                selected_tool="knowledge_document_search",
            ),
            results=[
                RetrievalResult.ok(
                    tool_name="knowledge_document_search",
                    query=original_query,
                    records=[],
                    documents=[],
                )
            ],
        )

        rewrite = GenericAssistantQueryRewriter(
            model_client=FakeQueryRewriteModelClient(model_output)
        ).invoke(context)

        assert rewrite.query == original_query
        assert rewrite.metadata["fallback"] is True
        assert str(rewrite.metadata["fallback_reason"]).startswith(
            "unsupported_generic_expansion:"
        )


def test_generic_sufficiency_judge_retries_arbitrary_no_hit_queries() -> None:
    judge = GenericAssistantSufficiencyJudge()
    decision = judge.invoke(
        RetrievalContext(
            plan=RetrievalPlan(
                user_query="当前有哪些表",
                active_query="当前有哪些表",
                selected_tool="knowledge_document_search",
                max_rounds=3,
            ),
            results=[
                RetrievalResult.ok(
                    tool_name="knowledge_document_search",
                    query="当前有哪些表",
                    records=[],
                    documents=[],
                )
            ],
        )
    )

    assert decision.next_action == "rewrite"


def test_generic_sufficiency_judge_does_not_rewrite_greetings() -> None:
    judge = GenericAssistantSufficiencyJudge()
    decision = judge.invoke(
        RetrievalContext(
            plan=RetrievalPlan(
                user_query="你好",
                active_query="你好",
                selected_tool="knowledge_document_search",
                max_rounds=3,
            ),
            results=[
                RetrievalResult.ok(
                    tool_name="knowledge_document_search",
                    query="你好",
                    records=[],
                    documents=[],
                )
            ],
        )
    )

    assert decision.next_action == "ask_user"


def test_scene_retrieval_policy_validates_allowed_values_and_rerank_top_n() -> None:
    assert SceneRetrievalPolicy(recall_strategy="semantic").recall_strategy == "semantic"
    assert SceneRetrievalPolicy(recall_strategy="keyword").recall_strategy == "keyword"
    assert SceneRetrievalPolicy(no_hit_strategy="fallback_answer").no_hit_strategy == "fallback_answer"
    assert SceneRetrievalPolicy(rerank_top_n=1).rerank_top_n == 1


def test_scene_retrieval_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="recall_strategy"):
        SceneRetrievalPolicy(recall_strategy="vector")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no_hit_strategy"):
        SceneRetrievalPolicy(no_hit_strategy="guess")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rerank_top_n"):
        SceneRetrievalPolicy(rerank_top_n=0)


def test_ecommerce_semantic_tool_filters_low_relevance_results() -> None:
    knowledge_service = FakeKnowledgeService()
    knowledge_service._products = [
        VectorSearchResult(
            document=VectorStoreDocument(
                id="P-low",
                content="AeroPhone X 低相关商品片段",
                metadata={"product_id": "P-low", "name": "AeroPhone X"},
            ),
            score=0.5,
        )
    ]
    product_store = ProductCatalogStore(
        data_dir=make_test_runtime_dir("agentic-low-relevance-product-store")
    )
    tool = ProductSemanticSearchTool(
        knowledge_service=knowledge_service,
        product_store=product_store,
    )

    result = tool.retrieve("aerophone x", min_relevance_score=DOCUMENT_MINIMUM_RELEVANCE)

    assert result.records == []
    assert result.documents == []
    assert result.citations == []


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


def test_agentic_retriever_passes_recall_strategy_to_document_tool() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-recall-strategy")
    document_service = FakeDocumentRetrievalService(knowledge_service)
    definition = build_generic_assistant_scene_definition(
        app_settings=app_settings,
        document_retrieval_service=document_service,
        retrieval_policy=SceneRetrievalPolicy(recall_strategy="keyword"),
    )
    retriever = definition.build_retriever()

    outcome = retriever.retrieve_with_trace(
        "请根据产品手册说明 AeroPhone X 的价格和电池参数",
        recall_strategy=definition.retrieval_policy.recall_strategy,
    )

    assert outcome.documents
    assert document_service.calls[-1]["recall_strategy"] == "keyword"


class _FakeDashScopeRerankWrapper:
    model = "gte-rerank-v2"

    def __init__(
        self,
        results: list[dict[str, object]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[dict[str, object]] = []

    def rerank(self, documents: list[object], query: str, **kwargs: object) -> list[dict[str, object]]:
        self.calls.append({"documents": documents, "query": query, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        return self.results


class _UnexpectedReranker:
    provider = "unexpected"

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, **kwargs: object):
        del kwargs
        self.calls += 1
        raise AssertionError("disabled rerank must not call reranker")


def _build_generic_retriever_with_reranker(test_name: str, reranker: object):
    app_settings, knowledge_service = _build_knowledge_service(test_name)
    definition = build_generic_assistant_scene_definition(
        app_settings=app_settings,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
    )
    retriever = definition.build_retriever()
    retriever.reranker = reranker
    return retriever, knowledge_service


def test_agentic_dashscope_rerank_sorts_records_documents_and_citations() -> None:
    wrapper = _FakeDashScopeRerankWrapper(
        [
            {"index": 1, "relevance_score": 0.97},
            {"index": 0, "relevance_score": 0.12},
        ]
    )
    reranker = DashScopeRetrievalReranker(wrapper_factory=lambda: wrapper)
    retriever, _knowledge_service = _build_generic_retriever_with_reranker(
        "agentic-dashscope-rerank-success",
        reranker,
    )

    outcome = retriever.retrieve_with_trace(
        "请根据产品手册说明 AeroPhone X 的价格和电池参数",
        rerank_enabled=True,
    )
    result = outcome.results[0]

    assert [record["citation_id"] for record in result.records] == ["DOC-002", "DOC-001"]
    assert [document.metadata["citation_id"] for document in result.documents] == ["DOC-002", "DOC-001"]
    assert [citation.citation_id for citation in result.citations] == ["DOC-002", "DOC-001"]
    assert [record["rerank_score"] for record in result.records] == [0.97, 0.12]
    assert [document.metadata["rerank_score"] for document in result.documents] == [0.97, 0.12]
    assert [citation.metadata["rerank_score"] for citation in result.citations] == [0.97, 0.12]
    assert result.metadata["rerank"] == {
        "enabled": True,
        "provider": "dashscope",
        "model": "gte-rerank-v2",
        "applied": True,
        "input_count": 2,
        "output_count": 2,
        "top_n": None,
        "fallback_reason": None,
        "error": None,
    }


def test_agentic_rerank_disabled_does_not_call_model_and_keeps_no_hit_empty() -> None:
    reranker = _UnexpectedReranker()
    retriever, knowledge_service = _build_generic_retriever_with_reranker(
        "agentic-rerank-disabled-no-call",
        reranker,
    )

    hit_outcome = retriever.retrieve_with_trace("请根据产品手册说明 AeroPhone X 的价格和电池参数")

    assert reranker.calls == 0
    assert [record["citation_id"] for record in hit_outcome.results[0].records] == ["DOC-001", "DOC-002"]
    assert [citation.citation_id for citation in hit_outcome.results[0].citations] == ["DOC-001", "DOC-002"]
    assert all("rerank_score" not in record for record in hit_outcome.results[0].records)
    assert all("rerank_score" not in document.metadata for document in hit_outcome.documents)
    assert hit_outcome.results[0].metadata["rerank"]["enabled"] is False

    knowledge_service.upsert_documents([])
    no_hit_outcome = retriever.retrieve_with_trace("没有任何文档能命中")

    assert reranker.calls == 0
    assert no_hit_outcome.documents == []
    assert no_hit_outcome.results[0].citations == []
    assert no_hit_outcome.results[0].metadata["rerank"]["enabled"] is False


@pytest.mark.parametrize(
    ("wrapper", "fallback_reason", "error", "output_count"),
    [
        (
            _FakeDashScopeRerankWrapper(error=RuntimeError("rerank failed")),
            "RuntimeError",
            "rerank failed",
            2,
        ),
        (
            _FakeDashScopeRerankWrapper(error=TimeoutError("rerank timeout")),
            "TimeoutError",
            "rerank timeout",
            2,
        ),
        (
            _FakeDashScopeRerankWrapper([]),
            "empty_rerank_result",
            None,
            0,
        ),
    ],
)
def test_agentic_rerank_fallback_preserves_original_order(
    wrapper: _FakeDashScopeRerankWrapper,
    fallback_reason: str,
    error: str | None,
    output_count: int,
) -> None:
    reranker = DashScopeRetrievalReranker(wrapper_factory=lambda: wrapper)
    retriever, _knowledge_service = _build_generic_retriever_with_reranker(
        f"agentic-rerank-fallback-{fallback_reason}",
        reranker,
    )

    outcome = retriever.retrieve_with_trace(
        "请根据产品手册说明 AeroPhone X 的价格和电池参数",
        rerank_enabled=True,
    )
    result = outcome.results[0]

    assert [record["citation_id"] for record in result.records] == ["DOC-001", "DOC-002"]
    assert [document.metadata["citation_id"] for document in result.documents] == ["DOC-001", "DOC-002"]
    assert [citation.citation_id for citation in result.citations] == ["DOC-001", "DOC-002"]
    assert all("rerank_score" not in record for record in result.records)
    assert all("rerank_score" not in document.metadata for document in result.documents)
    assert all("rerank_score" not in citation.metadata for citation in result.citations)
    assert result.metadata["rerank"] == {
        "enabled": True,
        "provider": "dashscope",
        "model": "gte-rerank-v2",
        "applied": False,
        "input_count": 2,
        "output_count": output_count,
        "top_n": None,
        "fallback_reason": fallback_reason,
        "error": error,
    }


def test_agentic_dashscope_rerank_top_n_limits_final_evidence_and_citations() -> None:
    wrapper = _FakeDashScopeRerankWrapper(
        [
            {"index": 1, "relevance_score": 0.97},
            {"index": 0, "relevance_score": 0.12},
        ]
    )
    reranker = DashScopeRetrievalReranker(wrapper_factory=lambda: wrapper)
    retriever, _knowledge_service = _build_generic_retriever_with_reranker(
        "agentic-dashscope-rerank-top-n",
        reranker,
    )

    outcome = retriever.retrieve_with_trace(
        "请根据产品手册说明 AeroPhone X 的价格和电池参数",
        rerank_enabled=True,
        rerank_top_n=1,
    )
    result = outcome.results[0]

    assert wrapper.calls[0]["kwargs"] == {"top_n": 1}
    assert [record["citation_id"] for record in result.records] == ["DOC-002"]
    assert [document.metadata["citation_id"] for document in result.documents] == ["DOC-002"]
    assert [citation.citation_id for citation in result.citations] == ["DOC-002"]
    assert [document.metadata["citation_id"] for document in outcome.documents] == ["DOC-002"]
    assert result.metadata["rerank"]["output_count"] == 1
    assert result.metadata["rerank"]["top_n"] == 1


def test_agentic_identity_rerank_truncates_records_documents_and_citations() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-identity-rerank")
    definition = build_generic_assistant_scene_definition(
        app_settings=app_settings,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
    )
    retriever = definition.build_retriever()
    retriever.reranker = IdentityRetrievalReranker()

    outcome = retriever.retrieve_with_trace(
        "请根据产品手册说明 AeroPhone X 的价格和电池参数",
        rerank_enabled=True,
        rerank_top_n=1,
    )

    assert len(outcome.results[0].records) == 1
    assert len(outcome.results[0].documents) == 1
    assert len(outcome.results[0].citations) == 1
    assert outcome.results[0].metadata["rerank"] == {
        "enabled": True,
        "provider": "identity",
        "model": None,
        "applied": False,
        "input_count": 2,
        "output_count": 1,
        "top_n": 1,
        "fallback_reason": None,
        "error": None,
    }


def test_agentic_rerank_disabled_preserves_result_order() -> None:
    app_settings, knowledge_service = _build_knowledge_service("agentic-rerank-disabled")
    definition = build_generic_assistant_scene_definition(
        app_settings=app_settings,
        document_retrieval_service=FakeDocumentRetrievalService(knowledge_service),
    )
    retriever = definition.build_retriever()

    outcome = retriever.retrieve_with_trace("请根据产品手册说明 AeroPhone X 的价格和电池参数")

    assert [record["citation_id"] for record in outcome.results[0].records] == ["DOC-001", "DOC-002"]
    assert outcome.results[0].metadata["rerank"]["enabled"] is False


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

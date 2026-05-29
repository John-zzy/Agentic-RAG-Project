from __future__ import annotations

import logging
from typing import Any
import warnings

from langchain_core.runnables import RunnableConfig

from backend.platform.config.settings import AppSettings, settings
from backend.platform.knowledge.repositories import VectorStoreFactory
from backend.platform.rag.retrieval.documents import DocumentRetrievalService
from backend.platform.rag.retrieval.documents.filters import DOCUMENT_MINIMUM_RELEVANCE
from backend.scenes.base import (
    SceneBootstrapResult,
    SceneDefinition,
    SceneFallbackPolicy,
    SceneRetrievalPolicy,
)
from backend.platform.rag.contracts import RetrievalContext
from backend.platform.rag.orchestration.decisions import SufficiencyDecision
from backend.platform.models.base.router import TaskComplexity
from backend.scenes.ecommerce.knowledge_service import KnowledgeService, create_knowledge_service
from backend.scenes.ecommerce.loader import preload_knowledge_base
from backend.scenes.ecommerce.tools import (
    ProductCatalogStore,
    build_ecommerce_action_tools,
    build_ecommerce_agentic_retrieval_tools,
    build_ecommerce_structured_retrieval_tools,
)
from backend.scenes.generic_assistant.definition import (
    GenericAssistantBusinessExtension,
    build_generic_assistant_scene_definition,
)


ECOMMERCE_SYSTEM_PROMPT = (
    "You are an ecommerce customer service assistant. "
    "Answer with retrieved product, review, order, and document evidence first. "
    "Do not fabricate inventory, price, or order status details. "
    "If evidence is missing, say what is missing and ask the user for a product name, order id, or keyword."
)

ECOMMERCE_RETRIEVAL_POLICY = SceneRetrievalPolicy(
    top_k=5,
    min_relevance_score=DOCUMENT_MINIMUM_RELEVANCE,
    recall_strategy="hybrid",
    no_hit_strategy="ask_user",
    rerank_enabled=False,
    rerank_top_n=None,
)

logger = logging.getLogger(__name__)

ECOMMERCE_INVENTORY_KEYWORDS: tuple[str, ...] = (
    "inventory",
    "stock",
    "available",
    "availability",
    "in stock",
    "有货",
    "库存",
    "现货",
    "补货",
)
ECOMMERCE_DETAIL_KEYWORDS: tuple[str, ...] = (
    "spec",
    "specs",
    "configuration",
    "price",
    "cost",
    "brand",
    "camera",
    "参数",
    "配置",
    "规格",
    "价格",
    "多少钱",
    "什么",
)
ECOMMERCE_REVIEW_KEYWORDS: tuple[str, ...] = (
    "review",
    "reviews",
    "rating",
    "feedback",
    "worth buying",
    "pros",
    "cons",
    "评价",
    "评论",
    "口碑",
    "值得买",
    "优点",
    "缺点",
)
ECOMMERCE_ORDER_KEYWORDS: tuple[str, ...] = (
    "order",
    "shipping",
    "logistics",
    "tracking",
    "delivery",
    "package",
    "订单",
    "发货",
    "物流",
    "快递",
    "配送",
    "包裹",
)
ECOMMERCE_DOCUMENT_KEYWORDS: tuple[str, ...] = (
    "文档",
    "说明",
    "制度",
    "规则",
    "手册",
    "流程",
    "faq",
    "知识库",
    "文件",
    "条款",
)


class EcommerceIntentRouter:
    """识别电商问题意图，并给出首跳与后续切换建议。"""

    inventory_keywords: tuple[str, ...] = ECOMMERCE_INVENTORY_KEYWORDS
    detail_keywords: tuple[str, ...] = ECOMMERCE_DETAIL_KEYWORDS
    review_keywords: tuple[str, ...] = ECOMMERCE_REVIEW_KEYWORDS
    order_keywords: tuple[str, ...] = ECOMMERCE_ORDER_KEYWORDS
    document_keywords: tuple[str, ...] = ECOMMERCE_DOCUMENT_KEYWORDS

    def has_document_intent(self, query: str) -> bool:
        return self._contains_any(query, self.document_keywords)

    def has_ecommerce_intent(self, query: str) -> bool:
        return self._contains_any(
            query,
            self.inventory_keywords
            + self.detail_keywords
            + self.review_keywords
            + self.order_keywords,
        )

    def resolve_entry_tool(
        self,
        query: str,
        candidate_tools: tuple[str, ...],
    ) -> str | None:
        if self._contains_any(query, self.order_keywords) and "order_semantic_search" in candidate_tools:
            return "order_semantic_search"
        if self._contains_any(query, self.review_keywords) and "review_semantic_search" in candidate_tools:
            return "review_semantic_search"
        if "product_semantic_search" in candidate_tools:
            return "product_semantic_search"
        return None

    def resolve_followup_tool(
        self,
        query: str,
        candidate_tools: tuple[str, ...],
        attempted_tools: tuple[str, ...],
    ) -> str | None:
        attempted = set(attempted_tools)
        if self._contains_any(query, self.order_keywords):
            if "order_semantic_search" in candidate_tools and "order_semantic_search" not in attempted:
                return "order_semantic_search"
        if self._contains_any(query, self.review_keywords):
            if "review_semantic_search" in candidate_tools and "review_semantic_search" not in attempted:
                return "review_semantic_search"
        return None

    def should_lookup_inventory(self, query: str) -> bool:
        return self._contains_any(query, self.inventory_keywords)

    def should_lookup_product_detail(self, query: str) -> bool:
        return self._contains_any(query, self.detail_keywords)

    def should_lookup_reviews(
        self,
        query: str,
        *,
        candidate_tools: tuple[str, ...],
        attempted_tools: tuple[str, ...],
    ) -> bool:
        return (
            self._contains_any(query, self.review_keywords)
            and "review_semantic_search" in candidate_tools
            and "review_semantic_search" not in attempted_tools
        )

    def should_lookup_orders(
        self,
        query: str,
        *,
        candidate_tools: tuple[str, ...],
        attempted_tools: tuple[str, ...],
    ) -> bool:
        return (
            self._contains_any(query, self.order_keywords)
            and "order_semantic_search" in candidate_tools
            and "order_semantic_search" not in attempted_tools
        )

    def _contains_any(self, query: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in query for keyword in keywords)


class EcommerceRetrievalToolProvider:
    """集中封装电商 retrieval tools 的构建。"""

    def __init__(
        self,
        *,
        app_settings: AppSettings,
        knowledge_service: KnowledgeService,
        product_store: ProductCatalogStore,
    ) -> None:
        self._app_settings = app_settings
        self._knowledge_service = knowledge_service
        self._product_store = product_store

    def build_tools(self) -> tuple[Any, ...]:
        return build_ecommerce_agentic_retrieval_tools(
            app_settings=self._app_settings,
            knowledge_service=self._knowledge_service,
            product_store=self._product_store,
        )


class EcommerceExtensionBootstrapper:
    """负责预加载电商 demo 数据，避免 builder 混入环境准备职责。"""

    def __init__(
        self,
        *,
        app_settings: AppSettings,
        knowledge_service: KnowledgeService,
    ) -> None:
        self._app_settings = app_settings
        self._knowledge_service = knowledge_service

    def bootstrap(self) -> SceneBootstrapResult:
        summary = preload_knowledge_base(
            app_settings=self._app_settings,
            store=self._knowledge_service.store,
        )
        return SceneBootstrapResult(
            metrics={
                "products_loaded": summary.products_loaded,
                "reviews_loaded": summary.reviews_loaded,
                "orders_loaded": summary.orders_loaded,
            }
        )


class EcommerceBusinessExtension(GenericAssistantBusinessExtension):
    """通过 generic docs-first 主链接入电商知识源。"""

    knowledge_source = "ecommerce"

    def __init__(
        self,
        *,
        tool_provider: EcommerceRetrievalToolProvider,
        intent_router: EcommerceIntentRouter,
        bootstrapper: EcommerceExtensionBootstrapper | None = None,
    ) -> None:
        self._tool_provider = tool_provider
        self._intent_router = intent_router
        self._bootstrapper = bootstrapper

    @property
    def retrieval_tool_names(self) -> tuple[str, ...]:
        return (
            "product_semantic_search",
            "review_semantic_search",
            "order_semantic_search",
            "inventory_lookup",
            "product_detail_lookup",
        )

    def build_retrieval_tools(self) -> tuple[Any, ...]:
        return self._tool_provider.build_tools()

    def bootstrap(self) -> SceneBootstrapResult:
        if self._bootstrapper is None:
            return SceneBootstrapResult()
        return self._bootstrapper.bootstrap()

    def should_handoff(self, context: RetrievalContext) -> SufficiencyDecision | None:
        result = context.results[-1]
        query = context.plan.user_query.lower()
        candidate_tools = context.plan.candidate_tools
        preferred_tool = self._intent_router.resolve_entry_tool(query, candidate_tools)
        if preferred_tool is None:
            return None
        is_document_question = self._intent_router.has_document_intent(query)
        if result.records and is_document_question:
            return None
        return SufficiencyDecision(
            is_sufficient=False,
            next_action="switch_tool",
            reason=(
                "当前问题带有明显电商意图，继续切换到电商知识源补充证据。"
                if result.records
                else "文档知识不足，继续切换到电商知识源补充证据。"
            ),
            suggested_tool=preferred_tool,
        )

    def resolve_followup(self, context: RetrievalContext) -> SufficiencyDecision | None:
        plan = context.plan
        result = context.results[-1]
        query = plan.user_query.lower()
        current_tool = plan.selected_tool
        top_product_id = self._resolve_top_product_id(result.records)

        if not result.records and current_tool == "product_semantic_search":
            fallback_tool = self._intent_router.resolve_followup_tool(
                query,
                plan.candidate_tools,
                plan.attempted_tools,
            )
            if fallback_tool is not None:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="商品检索没有命中，改查更贴近当前问题类型的电商知识源。",
                    suggested_tool=fallback_tool,
                )
            return None

        if current_tool == "product_semantic_search":
            if self._intent_router.should_lookup_inventory(query) and top_product_id:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="已定位到候选商品，下一步补查库存状态。",
                    suggested_tool="inventory_lookup",
                    metadata={"resolved_query": top_product_id},
                )
            if self._intent_router.should_lookup_product_detail(query) and top_product_id:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="已定位到候选商品，下一步补充精确参数与价格。",
                    suggested_tool="product_detail_lookup",
                    metadata={"resolved_query": top_product_id},
                )
            if self._intent_router.should_lookup_reviews(
                query,
                candidate_tools=plan.candidate_tools,
                attempted_tools=plan.attempted_tools,
            ):
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="需要评价证据补充推荐理由或口碑信息。",
                    suggested_tool="review_semantic_search",
                )
            if self._intent_router.should_lookup_orders(
                query,
                candidate_tools=plan.candidate_tools,
                attempted_tools=plan.attempted_tools,
            ):
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="需要订单证据补充物流或订单状态信息。",
                    suggested_tool="order_semantic_search",
                )
        return None

    def _resolve_top_product_id(self, records: list[dict[str, Any]]) -> str | None:
        for record in records:
            product_id = record.get("product_id") or record.get("citation_id")
            if isinstance(product_id, str) and product_id:
                return product_id
        return None


def build_ecommerce_business_extension(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: KnowledgeService | None = None,
    product_store: ProductCatalogStore | None = None,
) -> EcommerceBusinessExtension:
    """构建 generic scene 可注入的电商业务扩展。"""
    current_settings = app_settings or settings
    resolved_knowledge_service = knowledge_service or create_knowledge_service(current_settings)
    resolved_product_store = product_store or ProductCatalogStore(data_dir=current_settings.data_dir)
    tool_provider = EcommerceRetrievalToolProvider(
        app_settings=current_settings,
        knowledge_service=resolved_knowledge_service,
        product_store=resolved_product_store,
    )
    bootstrapper = (
        EcommerceExtensionBootstrapper(
            app_settings=current_settings,
            knowledge_service=resolved_knowledge_service,
        )
        if hasattr(resolved_knowledge_service, "store")
        else None
    )
    return EcommerceBusinessExtension(
        tool_provider=tool_provider,
        intent_router=EcommerceIntentRouter(),
        bootstrapper=bootstrapper,
    )


def build_ecommerce_scene_definition(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: object | None = None,
    document_retrieval_service: DocumentRetrievalService | None = None,
    product_store: ProductCatalogStore | None = None,
    max_rounds: int = 3,
) -> SceneDefinition:
    """构建电商场景定义。"""
    current_settings = app_settings or settings
    resolved_knowledge_service = _resolve_knowledge_service(current_settings, knowledge_service)
    resolved_document_retrieval_service = document_retrieval_service or DocumentRetrievalService(
        app_settings=current_settings,
        vector_repository=VectorStoreFactory.create_document_chunk_vector_repository(current_settings),
        chunk_source=VectorStoreFactory.create_active_document_chunk_source(current_settings),
    )
    resolved_product_store = product_store or ProductCatalogStore(data_dir=current_settings.data_dir)
    ecommerce_extension = build_ecommerce_business_extension(
        app_settings=current_settings,
        knowledge_service=resolved_knowledge_service,
        product_store=resolved_product_store,
    )
    generic_definition = build_generic_assistant_scene_definition(
        app_settings=current_settings,
        business_extensions=(ecommerce_extension,),
        document_retrieval_service=resolved_document_retrieval_service,
        retrieval_policy=ECOMMERCE_RETRIEVAL_POLICY,
        max_rounds=max_rounds,
    )
    return SceneDefinition(
        scene="ecommerce",
        name="Ecommerce Customer Service",
        description="Scene with product, review, order, and document retrieval for ecommerce support.",
        build_retriever=generic_definition.build_retriever,
        build_tools=lambda: (
            *build_ecommerce_structured_retrieval_tools(
                app_settings=current_settings,
                knowledge_service=resolved_knowledge_service,
                product_store=resolved_product_store,
            ),
            *build_ecommerce_action_tools(app_settings=current_settings),
        ),
        candidate_retrieval_tools_resolver=generic_definition.candidate_retrieval_tools_resolver,
        system_prompt=ECOMMERCE_SYSTEM_PROMPT,
        fallback_policy=SceneFallbackPolicy(
            no_hit_message="No relevant ecommerce knowledge was found. Please provide a more specific product name, order detail, or document keyword."
        ),
        infer_complexity=infer_ecommerce_complexity,
        retrieval_policy=ECOMMERCE_RETRIEVAL_POLICY,
        bootstrap=lambda: _bootstrap_scene(current_settings, resolved_knowledge_service),
        metadata={
            **generic_definition.metadata,
            "supports_agentic_retrieval": True,
            "knowledge_sources": ("documents", "ecommerce"),
            "default_agent": "shopping_agent",
            "prompt_style": "ecommerce_customer_service",
        },
    )


def create_agentic_knowledge_retriever(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: KnowledgeService | None = None,
    document_retrieval_service: DocumentRetrievalService | None = None,
    product_store: ProductCatalogStore | None = None,
    max_rounds: int = 3,
):
    """兼容入口：返回基于 generic docs-first 主链装配的电商 retriever。"""
    warnings.warn(
        "create_agentic_knowledge_retriever() is deprecated; use build_ecommerce_scene_definition(...).build_retriever() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_ecommerce_scene_definition(
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        document_retrieval_service=document_retrieval_service,
        product_store=product_store,
        max_rounds=max_rounds,
    ).build_retriever()


def infer_ecommerce_complexity(message: str) -> TaskComplexity:
    """Estimate model complexity for ecommerce support prompts."""
    normalized = message.lower()
    complex_keywords = ("refund", "return", "complaint", "ticket", "dissatisfied", "escalate", "human")
    moderate_keywords = ("recommend", "compare", "order", "shipping", "inventory", "spec", "price", "review")

    if any(keyword in normalized for keyword in complex_keywords):
        return "complex"
    if any(keyword in normalized for keyword in moderate_keywords) or len(normalized) > 40:
        return "moderate"
    return "simple"


def _bootstrap_scene(
    app_settings: AppSettings,
    knowledge_service: KnowledgeService,
) -> SceneBootstrapResult:
    """Preload demo ecommerce knowledge when the scene is activated."""
    return EcommerceExtensionBootstrapper(
        app_settings=app_settings,
        knowledge_service=knowledge_service,
    ).bootstrap()


def _resolve_knowledge_service(
    current_settings: AppSettings,
    knowledge_service: object | None,
) -> KnowledgeService:
    if knowledge_service is not None:
        return knowledge_service  # type: ignore[return-value]
    return create_knowledge_service(current_settings)

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from backend.platform.config.settings import AppSettings, settings
from backend.scenes.base import (
    SceneBootstrapResult,
    SceneDefinition,
    SceneFallbackPolicy,
)
from backend.platform.rag.agentic import AgenticRetriever
from backend.platform.rag.core import (
    QueryRewrite,
    QueryRewriter,
    RetrievalContext,
    SufficiencyDecision,
    SufficiencyJudge,
)
from backend.platform.models.base.router import TaskComplexity
from backend.scenes.ecommerce.commerce_tools import build_commerce_tools
from backend.scenes.ecommerce.knowledge_service import KnowledgeService, create_knowledge_service
from backend.scenes.ecommerce.loader import preload_knowledge_base
from backend.scenes.ecommerce.retrieval_tools import (
    ProductCatalogStore,
    build_agentic_retrieval_tools,
    build_retrieval_tools,
)


ECOMMERCE_SYSTEM_PROMPT = (
    "You are an ecommerce customer service assistant. "
    "Answer with retrieved product, review, order, and document evidence first. "
    "Do not fabricate inventory, price, or order status details. "
    "If evidence is missing, say what is missing and ask the user for a product name, order id, or keyword."
)

logger = logging.getLogger(__name__)


class EcommerceSufficiencyJudge(SufficiencyJudge):
    """驱动电商 Agentic Retrieval 的“文档优先、按需切换”决策。"""

    inventory_keywords: tuple[str, ...] = (
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
    detail_keywords: tuple[str, ...] = (
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
    review_keywords: tuple[str, ...] = (
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
    order_keywords: tuple[str, ...] = (
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
    document_keywords: tuple[str, ...] = (
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

    def invoke(
        self,
        input: RetrievalContext,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> SufficiencyDecision:
        """按当前证据判断是否结束、改写，或切换到电商工具。"""
        del config, kwargs
        plan = input.plan
        result = input.results[-1]
        normalized_query = plan.user_query.lower()
        current_tool = plan.selected_tool
        result_count = len(result.records)
        top_product_id = self._resolve_top_product_id(result.records)
        is_document_round = current_tool == "knowledge_document_search"
        can_use_ecommerce = self._has_any_ecommerce_tool(plan.candidate_tools)
        has_ecommerce_intent = self._is_ecommerce_intent(normalized_query)
        is_document_question = self._contains_any(normalized_query, self.document_keywords)
        has_document_evidence = self._has_document_evidence(input)
        preferred_ecommerce_tool = self._resolve_preferred_ecommerce_tool(
            normalized_query,
            plan.candidate_tools,
        )
        logger.info(
            "Sufficiency judge evaluating: round=%s, current_tool=%s, result_count=%s, candidate_tools=%s, has_ecommerce_intent=%s, is_document_question=%s, has_document_evidence=%s",
            plan.round_index,
            current_tool,
            result_count,
            plan.candidate_tools,
            has_ecommerce_intent,
            is_document_question,
            has_document_evidence,
        )

        if result_count == 0:
            if (
                is_document_round
                and can_use_ecommerce
                and has_ecommerce_intent
                and preferred_ecommerce_tool is not None
            ):
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="文档知识没有命中，且问题带有明显电商意图，继续尝试电商知识源。",
                    suggested_tool=preferred_ecommerce_tool,
                )
            if current_tool == "product_semantic_search":
                fallback_tool = self._resolve_followup_ecommerce_tool(
                    normalized_query,
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
            if plan.round_index >= plan.max_rounds:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="ask_user",
                    reason="在允许的检索轮次内没有找到足够相关的证据。",
                    follow_up_question="请补充更具体的商品名、订单号、文档关键词，或说明你想查询的知识来源。",
                )
            return SufficiencyDecision(
                is_sufficient=False,
                next_action="rewrite",
                reason="当前检索结果为空，需要先把查询改写得更清楚。",
            )

        if is_document_round:
            if is_document_question:
                return SufficiencyDecision(
                    is_sufficient=True,
                    next_action="finish",
                    reason="当前问题更偏文档问答，文档证据已足够支持回答。",
                    confidence=result.confidence,
                )
            if (
                can_use_ecommerce
                and has_ecommerce_intent
                and preferred_ecommerce_tool is not None
            ):
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="当前问题带有明显电商意图，继续补充电商知识以获得更直接的答案。",
                    suggested_tool=preferred_ecommerce_tool,
                )
            return SufficiencyDecision(
                is_sufficient=True,
                next_action="finish",
                reason="文档证据已足够支持当前回答。",
                confidence=result.confidence,
            )

        if current_tool == "product_semantic_search":
            if self._contains_any(normalized_query, self.inventory_keywords) and top_product_id:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="已定位到候选商品，下一步补查库存状态。",
                    suggested_tool="inventory_lookup",
                    metadata={"resolved_query": top_product_id},
                )
            if self._contains_any(normalized_query, self.detail_keywords) and top_product_id:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="已定位到候选商品，下一步补充精确参数与价格。",
                    suggested_tool="product_detail_lookup",
                    metadata={"resolved_query": top_product_id},
                )
            if (
                self._contains_any(normalized_query, self.review_keywords)
                and "review_semantic_search" in plan.candidate_tools
                and "review_semantic_search" not in plan.attempted_tools
            ):
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="需要评价证据补充推荐理由或口碑信息。",
                    suggested_tool="review_semantic_search",
                )
            if (
                self._contains_any(normalized_query, self.order_keywords)
                and "order_semantic_search" in plan.candidate_tools
                and "order_semantic_search" not in plan.attempted_tools
            ):
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="需要订单证据补充物流或订单状态信息。",
                    suggested_tool="order_semantic_search",
                )
            if has_document_evidence and not has_ecommerce_intent:
                return SufficiencyDecision(
                    is_sufficient=True,
                    next_action="finish",
                    reason="已有文档证据兜底，当前无需继续扩展更多电商检索。",
                    confidence=result.confidence,
                )

        return SufficiencyDecision(
            is_sufficient=True,
            next_action="finish",
            reason="当前证据已足够支持基于事实的回答。",
            confidence=result.confidence,
        )

    def _contains_any(self, query: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in query for keyword in keywords)

    def _is_ecommerce_intent(self, query: str) -> bool:
        """识别是否出现明显电商意图。"""
        return self._contains_any(
            query,
            self.inventory_keywords
            + self.detail_keywords
            + self.review_keywords
            + self.order_keywords,
        )

    def _has_any_ecommerce_tool(self, candidate_tools: tuple[str, ...]) -> bool:
        """判断当前候选工具里是否允许使用电商知识。"""
        ecommerce_tools = {
            "product_semantic_search",
            "review_semantic_search",
            "order_semantic_search",
            "inventory_lookup",
            "product_detail_lookup",
        }
        return any(tool_name in ecommerce_tools for tool_name in candidate_tools)

    def _has_document_evidence(self, context: RetrievalContext) -> bool:
        """判断累计证据中是否已有文档来源。"""
        return any(
            str(document.metadata.get("namespace")) == "documents"
            for document in context.documents
        )

    def _resolve_preferred_ecommerce_tool(
        self,
        query: str,
        candidate_tools: tuple[str, ...],
    ) -> str | None:
        """按问题类型挑选最适合的首个电商检索工具。"""
        if self._contains_any(query, self.order_keywords) and "order_semantic_search" in candidate_tools:
            return "order_semantic_search"
        if self._contains_any(query, self.review_keywords) and "review_semantic_search" in candidate_tools:
            return "review_semantic_search"
        if any(
            tool_name in candidate_tools
            for tool_name in ("product_semantic_search", "inventory_lookup", "product_detail_lookup")
        ):
            return "product_semantic_search" if "product_semantic_search" in candidate_tools else None
        return None

    def _resolve_followup_ecommerce_tool(
        self,
        query: str,
        candidate_tools: tuple[str, ...],
        attempted_tools: tuple[str, ...],
    ) -> str | None:
        """当商品检索没有命中时，尝试切到更匹配的电商知识源。"""
        attempted = set(attempted_tools)
        if self._contains_any(query, self.order_keywords):
            if "order_semantic_search" in candidate_tools and "order_semantic_search" not in attempted:
                return "order_semantic_search"
        if self._contains_any(query, self.review_keywords):
            if "review_semantic_search" in candidate_tools and "review_semantic_search" not in attempted:
                return "review_semantic_search"
        return None

    def _resolve_top_product_id(self, records: list[dict[str, Any]]) -> str | None:
        for record in records:
            product_id = record.get("product_id") or record.get("citation_id")
            if isinstance(product_id, str) and product_id:
                return product_id
        return None


class EcommerceQueryRewriter(QueryRewriter):
    """Rewrite weak ecommerce queries into broader follow-up retrieval queries."""

    def invoke(
        self,
        input: RetrievalContext,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> QueryRewrite:
        """Generate the next retrieval query from the current active query."""
        del config, kwargs
        query = input.plan.active_query.strip()
        lowered = query.lower()
        rewritten = (
            lowered.replace("is it in stock", "inventory")
            .replace("in stock", "inventory")
            .replace("worth buying", "reviews")
            .replace("how is it", "specs reviews")
        ).strip()
        if rewritten == lowered:
            rewritten = f"{query} product specs reviews documents"
        logger.info(
            "Query rewriter generated follow-up query: from=%r, to=%r",
            query,
            rewritten,
        )
        return QueryRewrite(
            query=rewritten,
            reason="Broadened the query with product, review, and document terms for the next round.",
            metadata={"original_query": query},
        )


def create_agentic_knowledge_retriever(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: KnowledgeService | None = None,
    product_store: ProductCatalogStore | None = None,
    max_rounds: int = 3,
) -> AgenticRetriever:
    """构建电商场景的 AgenticRetriever，默认先查文档。"""
    current_settings = app_settings or settings
    resolved_knowledge_service = knowledge_service or create_knowledge_service(current_settings)
    resolved_product_store = product_store or ProductCatalogStore(data_dir=current_settings.data_dir)
    tools = build_agentic_retrieval_tools(
        app_settings=current_settings,
        knowledge_service=resolved_knowledge_service,
        product_store=resolved_product_store,
    )
    return AgenticRetriever(
        tools={tool.name: tool for tool in tools},
        default_tool="knowledge_document_search",
        sufficiency_judge=EcommerceSufficiencyJudge(),
        query_rewriter=EcommerceQueryRewriter(),
        max_rounds=max_rounds,
    )


def build_ecommerce_scene_definition(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: object | None = None,
    product_store: ProductCatalogStore | None = None,
    max_rounds: int = 3,
) -> SceneDefinition:
    """构建电商场景定义。"""
    current_settings = app_settings or settings
    resolved_knowledge_service = _resolve_knowledge_service(current_settings, knowledge_service)
    resolved_product_store = product_store or ProductCatalogStore(data_dir=current_settings.data_dir)
    return SceneDefinition(
        scene="ecommerce",
        name="Ecommerce Customer Service",
        description="Scene with product, review, order, and document retrieval for ecommerce support.",
        build_retriever=lambda: create_agentic_knowledge_retriever(
            current_settings,
            knowledge_service=resolved_knowledge_service,
            product_store=resolved_product_store,
            max_rounds=max_rounds,
        ),
        build_tools=lambda: (
            *build_retrieval_tools(
                app_settings=current_settings,
                knowledge_service=resolved_knowledge_service,
                product_store=resolved_product_store,
            ),
            *build_commerce_tools(app_settings=current_settings),
        ),
        system_prompt=ECOMMERCE_SYSTEM_PROMPT,
        fallback_policy=SceneFallbackPolicy(
            no_hit_message="No relevant ecommerce knowledge was found. Please provide a more specific product name, order detail, or document keyword."
        ),
        infer_complexity=infer_ecommerce_complexity,
        bootstrap=lambda: _bootstrap_scene(current_settings, resolved_knowledge_service),
        metadata={
            "supports_agentic_retrieval": True,
            "knowledge_sources": ("documents", "ecommerce"),
            "default_agent": "shopping_agent",
            "prompt_style": "ecommerce_customer_service",
        },
    )


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
    summary = preload_knowledge_base(
        app_settings=app_settings,
        store=knowledge_service.store,
    )
    return SceneBootstrapResult(
        metrics={
            "products_loaded": summary.products_loaded,
            "reviews_loaded": summary.reviews_loaded,
            "orders_loaded": summary.orders_loaded,
        }
    )


def _resolve_knowledge_service(
    current_settings: AppSettings,
    knowledge_service: object | None,
) -> KnowledgeService:
    if knowledge_service is not None:
        return knowledge_service  # type: ignore[return-value]
    return create_knowledge_service(current_settings)

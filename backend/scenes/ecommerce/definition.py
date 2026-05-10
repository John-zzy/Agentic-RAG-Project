from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict, Field

from backend.platform.config.settings import AppSettings, settings
from backend.scenes.base import (
    SceneBootstrapResult,
    SceneDefinition,
    SceneFallbackPolicy,
)
from backend.platform.knowledge.base.store import VectorSearchResult
from backend.platform.knowledge.base.text import truncate_snippet
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


class EcommerceKnowledgeBaseRetriever(BaseRetriever):
    """Aggregate ecommerce knowledge sources into a scene-level retriever."""

    knowledge_service: Any = Field(exclude=True)
    default_top_k: int = 5
    minimum_relevance: float = 0.18

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        """Adapt to the LangChain retriever protocol."""
        return self.search(query=query, top_k=self.default_top_k)

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """Merge product, review, order, and document results with ranking and dedupe."""
        requested_top_k = top_k or self.default_top_k
        product_results = self.knowledge_service.search_products(query=query, top_k=requested_top_k)
        review_results = self.knowledge_service.search_reviews(query=query, top_k=requested_top_k)
        order_results = self.knowledge_service.search_orders(query=query, top_k=requested_top_k)
        document_results = self.knowledge_service.search_document_chunks(query=query, top_k=requested_top_k)

        combined = (
            self._to_documents("products", product_results)
            + self._to_documents("reviews", review_results)
            + self._to_documents("orders", order_results)
            + self._to_documents("documents", document_results)
        )
        combined.sort(key=self._doc_score, reverse=True)

        deduped: list[Document] = []
        seen: set[tuple[str, str]] = set()
        for doc in combined:
            namespace = str(doc.metadata.get("namespace", "knowledge"))
            citation_id = str(doc.metadata.get("citation_id", "unknown"))
            key = (namespace, citation_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
            if len(deduped) >= requested_top_k:
                break
        return deduped

    def _to_documents(self, namespace: str, results: list[VectorSearchResult]) -> list[Document]:
        """Map vector search results into LangChain documents."""
        documents: list[Document] = []
        for result in results:
            score = float(result.score) if result.score is not None else None
            if score is not None and score < self.minimum_relevance:
                continue
            metadata = result.document.metadata
            citation_id = str(
                metadata.get("review_id")
                or metadata.get("product_id")
                or metadata.get("order_id")
                or metadata.get("document_id")
                or metadata.get("source_path")
                or metadata.get("id")
                or result.document.id
            )
            snippet = truncate_snippet(result.document.content)
            if not snippet:
                continue
            documents.append(
                Document(
                    page_content=snippet,
                    metadata={"namespace": namespace, "citation_id": citation_id, "score": score},
                )
            )
        return documents

    def _doc_score(self, doc: Document) -> float:
        """Extract score for ranking."""
        score = doc.metadata.get("score")
        if isinstance(score, int | float):
            return float(score)
        return -1.0


class EcommerceSufficiencyJudge(SufficiencyJudge):
    """Drive multi-round agentic retrieval decisions for ecommerce scene."""

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

    def invoke(
        self,
        input: RetrievalContext,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> SufficiencyDecision:
        """Choose finish, rewrite, switch_tool, or ask_user for the next round."""
        del config, kwargs
        plan = input.plan
        result = input.results[-1]
        normalized_query = plan.user_query.lower()
        current_tool = plan.selected_tool
        result_count = len(result.records)
        top_product_id = self._resolve_top_product_id(result.records)

        if result_count == 0:
            if current_tool != "knowledge_document_search" and "knowledge_document_search" not in plan.attempted_tools:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="Built-in ecommerce knowledge returned no hits; try uploaded knowledge documents next.",
                    suggested_tool="knowledge_document_search",
                )
            if plan.round_index >= plan.max_rounds:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="ask_user",
                    reason="No relevant evidence was found after the allowed retrieval attempts.",
                    follow_up_question="Please provide a more specific product name, file keyword, or preferred knowledge source.",
                )
            return SufficiencyDecision(
                is_sufficient=False,
                next_action="rewrite",
                reason="The current retrieval result is empty and needs a clearer query.",
            )

        if current_tool == "product_semantic_search":
            if self._contains_any(normalized_query, self.inventory_keywords) and top_product_id:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="Product candidates are available; inventory status should be confirmed next.",
                    suggested_tool="inventory_lookup",
                    metadata={"resolved_query": top_product_id},
                )
            if self._contains_any(normalized_query, self.detail_keywords) and top_product_id:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="Product candidates are available; exact details should be filled in next.",
                    suggested_tool="product_detail_lookup",
                    metadata={"resolved_query": top_product_id},
                )
            if self._contains_any(normalized_query, self.review_keywords) and "review_semantic_search" not in plan.attempted_tools:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="Need review evidence to support recommendation quality or user sentiment.",
                    suggested_tool="review_semantic_search",
                )
            if self._contains_any(normalized_query, self.order_keywords) and "order_semantic_search" not in plan.attempted_tools:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="Need order evidence to support order tracking or logistics inquiries.",
                    suggested_tool="order_semantic_search",
                )
            if "knowledge_document_search" not in plan.attempted_tools:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="Need to cross-check uploaded knowledge documents before finishing.",
                    suggested_tool="knowledge_document_search",
                )

        return SufficiencyDecision(
            is_sufficient=True,
            next_action="finish",
            reason="Current evidence is sufficient to support a grounded answer.",
            confidence=result.confidence,
        )

    def _contains_any(self, query: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in query for keyword in keywords)

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
    """Build the primary ecommerce AgenticRetriever implementation."""
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
        default_tool="product_semantic_search",
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
    """Build the ecommerce scene as the primary runtime implementation."""
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
            "knowledge_sources": ("products", "reviews", "orders", "documents"),
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

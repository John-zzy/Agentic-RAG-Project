from __future__ import annotations

import logging
from typing import Any

from backend.platform.rag.contracts import RecallStrategy, RetrievalResult
from backend.platform.tools import SceneTool, ToolResult
from backend.scenes.ecommerce.knowledge_service import KnowledgeService
from backend.scenes.ecommerce.tools.common import (
    build_retrieval_result,
    filter_vector_results_by_min_relevance,
    inject_named_product_match,
    to_tool_result,
)
from backend.scenes.ecommerce.tools.schemas import SemanticSearchInput
from backend.scenes.ecommerce.tools.stores import ProductCatalogStore


logger = logging.getLogger(__name__)


class ProductSemanticSearchTool(SceneTool):
    """商品语义检索工具，负责商品知识源召回与商品名精确增益。"""

    name = "product_semantic_search"
    description = "Search semantically similar products for the user's shopping intent."
    capability_type = "retrieval"
    args_schema = SemanticSearchInput

    def __init__(
        self,
        *,
        knowledge_service: KnowledgeService,
        product_store: ProductCatalogStore,
        default_top_k: int = 5,
    ) -> None:
        self._knowledge_service = knowledge_service
        self._product_store = product_store
        self._default_top_k = default_top_k

    def invoke(self, query: str, top_k: int = 5) -> ToolResult:
        return to_tool_result(self.retrieve(query=query, top_k=top_k))

    def retrieve(
        self,
        query: str,
        *,
        run_manager: Any | None = None,
        top_k: int | None = None,
        min_relevance_score: float | None = None,
        recall_strategy: RecallStrategy = "hybrid",
        rerank_enabled: bool = False,
        rerank_top_n: int | None = None,
    ) -> RetrievalResult:
        del run_manager, recall_strategy, rerank_top_n
        resolved_top_k = top_k or self._default_top_k
        logger.info("Product semantic search started: query=%r, top_k=%s", query, resolved_top_k)
        vector_results = filter_vector_results_by_min_relevance(
            self._knowledge_service.search_products(query=query, top_k=resolved_top_k),
            min_relevance_score,
        )
        vector_results = inject_named_product_match(query, vector_results, self._product_store)
        if rerank_enabled:
            logger.info("Rerank is configured but no rerank service is wired; preserving retrieval order.")
        return build_retrieval_result(
            tool_name=self.name,
            namespace="products",
            query=query,
            vector_results=vector_results,
        )

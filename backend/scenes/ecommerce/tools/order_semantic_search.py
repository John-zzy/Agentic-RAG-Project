from __future__ import annotations

import logging
from typing import Any

from backend.platform.rag.contracts import RecallStrategy, RetrievalResult
from backend.platform.tools import SceneTool, ToolResult
from backend.scenes.ecommerce.knowledge_service import KnowledgeService
from backend.scenes.ecommerce.tools.common import (
    build_retrieval_result,
    filter_vector_results_by_min_relevance,
    rank_order_results,
    to_tool_result,
)
from backend.scenes.ecommerce.tools.schemas import SemanticSearchInput


logger = logging.getLogger(__name__)


class OrderSemanticSearchTool(SceneTool):
    """订单语义检索工具，负责订单知识源召回与订单号精确排序。"""

    name = "order_semantic_search"
    description = "Search order information semantically for order tracking or status inquiries."
    capability_type = "retrieval"
    args_schema = SemanticSearchInput

    def __init__(self, *, knowledge_service: KnowledgeService, default_top_k: int = 5) -> None:
        self._knowledge_service = knowledge_service
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
        logger.info("Order semantic search started: query=%r, top_k=%s", query, resolved_top_k)
        vector_results = filter_vector_results_by_min_relevance(
            self._knowledge_service.search_orders(query=query, top_k=resolved_top_k),
            min_relevance_score,
        )
        vector_results = rank_order_results(query, vector_results)
        if rerank_enabled:
            logger.info("Rerank is configured but no rerank service is wired; preserving retrieval order.")
        return build_retrieval_result(
            tool_name=self.name,
            namespace="orders",
            query=query,
            vector_results=vector_results,
        )

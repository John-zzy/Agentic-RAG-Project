from __future__ import annotations

import logging
from typing import Any

from backend.platform.rag.contracts import RecallStrategy, RetrievalResult
from backend.platform.search_foundation import VectorSearchResult
from backend.platform.tools import SceneTool, ToolResult
from backend.scenes.ecommerce.knowledge_service import KnowledgeService
from backend.scenes.ecommerce.tools.common import (
    build_retrieval_result,
    filter_vector_results_by_min_relevance,
    to_tool_result,
)
from backend.scenes.ecommerce.tools.schemas import ReviewSemanticSearchInput


logger = logging.getLogger(__name__)


class ReviewSemanticSearchTool(SceneTool):
    """评价语义检索工具，负责评价证据召回和可选商品过滤。"""

    name = "review_semantic_search"
    description = "Search review evidence related to the user's shopping question."
    capability_type = "retrieval"
    args_schema = ReviewSemanticSearchInput

    def __init__(self, *, knowledge_service: KnowledgeService, default_top_k: int = 5) -> None:
        self._knowledge_service = knowledge_service
        self._default_top_k = default_top_k

    def invoke(self, query: str, top_k: int = 5, product_id: str | None = None) -> ToolResult:
        filters = {"product_id": product_id} if product_id else None
        vector_results = self._search_reviews(query=query, top_k=top_k, filters=filters)
        retrieval_result = build_retrieval_result(
            tool_name=self.name,
            namespace="reviews",
            query=query,
            vector_results=vector_results,
        )
        if filters:
            retrieval_result.metadata["filters"] = filters
        return to_tool_result(retrieval_result)

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
        del run_manager, recall_strategy, rerank_enabled, rerank_top_n
        resolved_top_k = top_k or self._default_top_k
        logger.info("Review semantic search started: query=%r, top_k=%s", query, resolved_top_k)
        vector_results = filter_vector_results_by_min_relevance(
            self._search_reviews(query=query, top_k=resolved_top_k, filters=None),
            min_relevance_score,
        )
        return build_retrieval_result(
            tool_name=self.name,
            namespace="reviews",
            query=query,
            vector_results=vector_results,
        )

    def _search_reviews(
        self,
        *,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[VectorSearchResult]:
        return self._knowledge_service.search_reviews(query=query, top_k=top_k, filters=filters)

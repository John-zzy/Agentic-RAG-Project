from __future__ import annotations

from typing import Any

from backend.platform.rag.contracts import RecallStrategy, RetrievalResult
from backend.platform.tools import SceneTool, ToolResult
from backend.scenes.ecommerce.tools.common import (
    build_citation,
    build_document,
    build_inventory_record,
    to_tool_result,
)
from backend.scenes.ecommerce.tools.schemas import ProductLookupInput
from backend.scenes.ecommerce.tools.stores import ProductCatalogStore


class InventoryLookupTool(SceneTool):
    """库存查询工具，按商品 ID 返回结构化库存状态。"""

    name = "inventory_lookup"
    description = "Look up inventory availability and stock status for a known product ID."
    capability_type = "retrieval"
    args_schema = ProductLookupInput

    def __init__(self, *, product_store: ProductCatalogStore) -> None:
        self._product_store = product_store

    def invoke(self, product_id: str) -> ToolResult:
        return to_tool_result(self.retrieve(query=product_id))

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
        del run_manager, top_k, min_relevance_score, recall_strategy, rerank_enabled, rerank_top_n
        product = self._product_store.find_product(query.strip())
        if product is None:
            return RetrievalResult.fail(
                tool_name=self.name,
                query=query,
                error=f"Product '{query}' was not found.",
                metadata={"namespace": "inventory"},
            )

        record = build_inventory_record(product)
        snippet = record["inventory_summary"]
        return RetrievalResult.ok(
            tool_name=self.name,
            query=query,
            records=[record],
            documents=[
                build_document(
                    snippet=snippet,
                    namespace="inventory",
                    citation_id=record["product_id"],
                    score=1.0,
                    extra_metadata={"product_id": record["product_id"]},
                )
            ],
            citations=[
                build_citation(
                    citation_id=record["product_id"],
                    namespace="inventory",
                    snippet=snippet,
                    metadata={"product_id": record["product_id"]},
                )
            ],
            confidence=1.0,
            metadata={"namespace": "inventory", "result_count": 1},
        )

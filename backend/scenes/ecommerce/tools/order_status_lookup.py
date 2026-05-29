from __future__ import annotations

from backend.platform.tools import SceneTool, ToolResult
from backend.scenes.ecommerce.tools.schemas import OrderStatusLookupInput
from backend.scenes.ecommerce.tools.stores import CommerceDataStore


class OrderStatusLookupTool(SceneTool):
    """订单状态查询工具，按订单号返回当前履约信息。"""

    name = "order_status_lookup"
    description = "Look up the current status and fulfillment details for an order."
    capability_type = "action"
    args_schema = OrderStatusLookupInput

    def __init__(self, *, store: CommerceDataStore) -> None:
        self._store = store

    def invoke(self, order_id: str) -> ToolResult:
        order = self._store.find_order(order_id)
        if order is None:
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Order '{order_id}' was not found.",
            )
        return ToolResult.ok(
            tool_name=self.name,
            records=[order],
            confidence=1.0,
        )

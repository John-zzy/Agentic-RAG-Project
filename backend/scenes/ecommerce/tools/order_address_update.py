from __future__ import annotations

from backend.platform.tools import SceneTool, ToolResult
from backend.scenes.ecommerce.tools.schemas import OrderAddressUpdateInput
from backend.scenes.ecommerce.tools.stores import CommerceDataStore


class OrderAddressUpdateTool(SceneTool):
    """订单地址修改工具，负责更新指定订单的收货地址。"""

    name = "order_address_update"
    description = "Update the shipping address saved on an existing order."
    capability_type = "action"
    args_schema = OrderAddressUpdateInput

    def __init__(self, *, store: CommerceDataStore) -> None:
        self._store = store

    def invoke(self, order_id: str, new_address: str) -> ToolResult:
        order = self._store.update_order_address(order_id, new_address)
        if order is None:
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Order '{order_id}' was not found.",
            )
        return ToolResult.ok(
            tool_name=self.name,
            records=[order],
            confidence=0.95,
        )

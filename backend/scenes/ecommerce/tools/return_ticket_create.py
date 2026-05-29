from __future__ import annotations

from uuid import uuid4

from backend.platform.tools import SceneTool, ToolResult
from backend.scenes.ecommerce.tools.schemas import ReturnTicketCreateInput
from backend.scenes.ecommerce.tools.stores import CommerceDataStore, utc_now


class ReturnTicketCreateTool(SceneTool):
    """退换货工单创建工具，负责登记退货或换货诉求。"""

    name = "return_ticket_create"
    description = "Create a return or exchange service ticket for an order."
    capability_type = "action"
    args_schema = ReturnTicketCreateInput

    def __init__(self, *, store: CommerceDataStore) -> None:
        self._store = store

    def invoke(self, order_id: str, reason: str, items: list[str]) -> ToolResult:
        order = self._store.find_order(order_id)
        if order is None:
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Order '{order_id}' was not found.",
            )

        ticket = self._store.create_service_ticket(
            {
                "ticket_id": f"RET-{uuid4().hex[:10]}",
                "ticket_type": "return",
                "order_id": order_id,
                "reason": reason,
                "items": items,
                "status": "open",
                "created_at": utc_now(),
            }
        )
        return ToolResult.ok(
            tool_name=self.name,
            records=[ticket],
            confidence=0.9,
            metadata={"linked_order_status": order.get("status")},
        )

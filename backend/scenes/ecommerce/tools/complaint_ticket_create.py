from __future__ import annotations

from uuid import uuid4

from backend.platform.tools import SceneTool, ToolResult
from backend.scenes.ecommerce.tools.schemas import ComplaintTicketCreateInput
from backend.scenes.ecommerce.tools.stores import CommerceDataStore, utc_now


class ComplaintTicketCreateTool(SceneTool):
    """投诉工单创建工具，负责登记订单相关投诉。"""

    name = "complaint_ticket_create"
    description = "Create a customer complaint ticket for order-related problems."
    capability_type = "action"
    args_schema = ComplaintTicketCreateInput

    def __init__(self, *, store: CommerceDataStore) -> None:
        self._store = store

    def invoke(
        self,
        order_id: str,
        message: str,
        contact: str | None = None,
    ) -> ToolResult:
        order = self._store.find_order(order_id)
        if order is None:
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Order '{order_id}' was not found.",
            )

        ticket = self._store.create_service_ticket(
            {
                "ticket_id": f"COM-{uuid4().hex[:10]}",
                "ticket_type": "complaint",
                "order_id": order_id,
                "message": message,
                "contact": contact,
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

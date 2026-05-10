from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from backend.platform.config.settings import AppSettings, settings
from backend.platform.tools import BaseJsonStore, ToolResult, build_structured_tool


ORDERS_FILE_NAME = "orders.json"
SERVICE_TICKETS_FILE_NAME = "service_tickets.json"


class OrderStatusLookupInput(BaseModel):
    """订单状态查询工具的输入参数。"""

    order_id: str = Field(min_length=1)


class OrderAddressUpdateInput(BaseModel):
    """订单地址修改工具的输入参数。"""

    order_id: str = Field(min_length=1)
    new_address: str = Field(min_length=1)


class ReturnTicketCreateInput(BaseModel):
    """退换货工单创建工具的输入参数。"""

    order_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    items: list[str] = Field(default_factory=list)


class ComplaintTicketCreateInput(BaseModel):
    """投诉工单创建工具的输入参数。"""

    order_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    contact: str | None = None


@dataclass
class CommerceDataStore(BaseJsonStore):
    """封装订单与售后工单的本地 JSON 读写，隔离工具层的数据访问细节。

    继承 BaseJsonStore 复用 JSON 文件读写能力，专注于订单和工单的业务操作。
    数据存储在本地 JSON 文件中（orders.json / service_tickets.json），
    适用于中小规模数据量和开发测试场景。
    """

    def load_orders(self) -> list[dict[str, Any]]:
        """读取订单列表；若文件不存在则返回空列表。"""
        return self._load_json_list(ORDERS_FILE_NAME)

    def save_orders(self, orders: list[dict[str, Any]]) -> None:
        """持久化订单列表到本地 JSON 文件。"""
        self._save_json_list(ORDERS_FILE_NAME, orders)

    def load_service_tickets(self) -> list[dict[str, Any]]:
        """读取售后工单列表；若文件不存在则返回空列表。"""
        return self._load_json_list(SERVICE_TICKETS_FILE_NAME)

    def save_service_tickets(self, tickets: list[dict[str, Any]]) -> None:
        """持久化售后工单列表到本地 JSON 文件。"""
        self._save_json_list(SERVICE_TICKETS_FILE_NAME, tickets)

    def find_order(self, order_id: str) -> dict[str, Any] | None:
        """按订单号精确查找订单，不存在时返回 None。

        遍历订单列表进行线性查找，适用于数据量较小的场景。
        """
        for order in self.load_orders():
            if order.get("order_id") == order_id:
                return order
        return None

    def update_order_address(self, order_id: str, new_address: str) -> dict[str, Any] | None:
        """更新指定订单的收货地址，并在命中时补充更新时间戳。

        先加载全部订单，修改匹配项后整体写回文件。
        返回更新后的订单字典，未找到时返回 None。
        """
        orders = self.load_orders()
        for order in orders:
            if order.get("order_id") == order_id:
                order["shipping_address"] = new_address
                order["updated_at"] = _utc_now()
                self.save_orders(orders)
                return order
        return None

    def create_service_ticket(self, payload: dict[str, Any]) -> dict[str, Any]:
        """追加一条售后工单记录并返回该记录。

        工单类型包括退货（return）和投诉（complaint），
        由调用方在 payload 中指定 ticket_type。
        """
        tickets = self.load_service_tickets()
        tickets.append(payload)
        self.save_service_tickets(tickets)
        return payload


def _build_order_status_lookup_tool(store: CommerceDataStore) -> BaseTool:
    """构建订单状态查询工具，供 LangChain Agent 直接调用。"""

    def order_status_lookup(order_id: str) -> ToolResult:
        """根据订单号返回订单详情；未命中时返回标准失败结果。"""
        order = store.find_order(order_id)
        if order is None:
            return ToolResult.fail(
                tool_name="order_status_lookup",
                error=f"Order '{order_id}' was not found.",
            )
        return ToolResult.ok(
            tool_name="order_status_lookup",
            records=[order],
            confidence=1.0,
        )
    return build_structured_tool(
        name="order_status_lookup",
        description="Look up the current status and fulfillment details for an order.",
        capability_type="action",
        args_schema=OrderStatusLookupInput,
        func=order_status_lookup,
    )


def _build_order_address_update_tool(store: CommerceDataStore) -> BaseTool:
    """构建订单地址修改工具，供 LangChain Agent 直接调用。"""

    def order_address_update(order_id: str, new_address: str) -> ToolResult:
        """更新指定订单的收货地址，并返回更新后的订单信息。"""
        order = store.update_order_address(order_id, new_address)
        if order is None:
            return ToolResult.fail(
                tool_name="order_address_update",
                error=f"Order '{order_id}' was not found.",
            )
        return ToolResult.ok(
            tool_name="order_address_update",
            records=[order],
            confidence=0.95,
        )
    return build_structured_tool(
        name="order_address_update",
        description="Update the shipping address saved on an existing order.",
        capability_type="action",
        args_schema=OrderAddressUpdateInput,
        func=order_address_update,
    )


def _build_return_ticket_create_tool(store: CommerceDataStore) -> BaseTool:
    """构建退换货工单工具，供 LangChain Agent 直接调用。"""

    def return_ticket_create(order_id: str, reason: str, items: list[str]) -> ToolResult:
        """为指定订单创建退货工单，并附带关联订单状态。"""
        order = store.find_order(order_id)
        if order is None:
            return ToolResult.fail(
                tool_name="return_ticket_create",
                error=f"Order '{order_id}' was not found.",
            )

        ticket = store.create_service_ticket(
            {
                "ticket_id": f"RET-{uuid4().hex[:10]}",
                "ticket_type": "return",
                "order_id": order_id,
                "reason": reason,
                "items": items,
                "status": "open",
                "created_at": _utc_now(),
            }
        )
        return ToolResult.ok(
            tool_name="return_ticket_create",
            records=[ticket],
            confidence=0.9,
            metadata={"linked_order_status": order.get("status")},
        )
    return build_structured_tool(
        name="return_ticket_create",
        description="Create a return or exchange service ticket for an order.",
        capability_type="action",
        args_schema=ReturnTicketCreateInput,
        func=return_ticket_create,
    )


def _build_complaint_ticket_create_tool(store: CommerceDataStore) -> BaseTool:
    """构建投诉工单工具，供 LangChain Agent 直接调用。"""

    def complaint_ticket_create(
        order_id: str,
        message: str,
        contact: str | None = None,
    ) -> ToolResult:
        """为指定订单创建投诉工单，并保留联系信息。"""
        order = store.find_order(order_id)
        if order is None:
            return ToolResult.fail(
                tool_name="complaint_ticket_create",
                error=f"Order '{order_id}' was not found.",
            )

        ticket = store.create_service_ticket(
            {
                "ticket_id": f"COM-{uuid4().hex[:10]}",
                "ticket_type": "complaint",
                "order_id": order_id,
                "message": message,
                "contact": contact,
                "status": "open",
                "created_at": _utc_now(),
            }
        )
        return ToolResult.ok(
            tool_name="complaint_ticket_create",
            records=[ticket],
            confidence=0.9,
            metadata={"linked_order_status": order.get("status")},
        )
    return build_structured_tool(
        name="complaint_ticket_create",
        description="Create a customer complaint ticket for order-related problems.",
        capability_type="action",
        args_schema=ComplaintTicketCreateInput,
        func=complaint_ticket_create,
    )


def build_commerce_tools(
    app_settings: AppSettings | None = None,
    *,
    store: CommerceDataStore | None = None,
) -> tuple[BaseTool, ...]:
    """按统一顺序构建订单与售后工具集合。"""
    current_settings = app_settings or settings
    data_store = store or CommerceDataStore(data_dir=current_settings.data_dir)
    return (
        _build_order_status_lookup_tool(data_store),
        _build_order_address_update_tool(data_store),
        _build_return_ticket_create_tool(data_store),
        _build_complaint_ticket_create_tool(data_store),
    )


def _utc_now() -> str:
    """生成 UTC ISO 时间戳，供工具写入更新时间和建单时间。"""
    return datetime.now(UTC).isoformat()

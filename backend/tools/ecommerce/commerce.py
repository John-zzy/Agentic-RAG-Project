from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.config.settings import AppSettings, settings
from backend.tools.base import AgentTool, ToolContext, ToolResult


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
class CommerceDataStore:
    """封装订单与售后工单的本地 JSON 读写，隔离工具层的数据访问细节。"""

    data_dir: Path

    @property
    def orders_path(self) -> Path:
        return self.data_dir / ORDERS_FILE_NAME

    @property
    def service_tickets_path(self) -> Path:
        return self.data_dir / SERVICE_TICKETS_FILE_NAME

    def load_orders(self) -> list[dict[str, Any]]:
        """读取订单列表；若文件不存在则返回空列表。"""
        return self._load_json_list(self.orders_path)

    def save_orders(self, orders: list[dict[str, Any]]) -> None:
        """持久化订单列表。"""
        self._write_json_list(self.orders_path, orders)

    def load_service_tickets(self) -> list[dict[str, Any]]:
        """读取售后工单列表。"""
        return self._load_json_list(self.service_tickets_path)

    def save_service_tickets(self, tickets: list[dict[str, Any]]) -> None:
        """持久化售后工单列表。"""
        self._write_json_list(self.service_tickets_path, tickets)

    def find_order(self, order_id: str) -> dict[str, Any] | None:
        """按订单号查找订单，不存在时返回 None。"""
        for order in self.load_orders():
            if order.get("order_id") == order_id:
                return order
        return None

    def update_order_address(self, order_id: str, new_address: str) -> dict[str, Any] | None:
        """更新订单收货地址，并在命中时补充更新时间。"""
        orders = self.load_orders()
        for order in orders:
            if order.get("order_id") == order_id:
                order["shipping_address"] = new_address
                order["updated_at"] = _utc_now()
                self.save_orders(orders)
                return order
        return None

    def create_service_ticket(self, payload: dict[str, Any]) -> dict[str, Any]:
        """追加一条售后工单记录并返回该记录。"""
        tickets = self.load_service_tickets()
        tickets.append(payload)
        self.save_service_tickets(tickets)
        return payload

    def _load_json_list(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json_list(self, path: Path, payload: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class OrderStatusLookupTool(AgentTool):
    """查询订单当前状态和履约信息的基础工具。"""

    name = "order_status_lookup"
    description = "Look up the current status and fulfillment details for an order."
    capability_type = "action"
    input_model = OrderStatusLookupInput

    def __init__(self, store: CommerceDataStore) -> None:
        self.store = store

    def invoke(
        self,
        tool_input: BaseModel | dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """根据订单号返回订单详情；未命中时返回标准失败结果。"""
        payload = self.parse_input(tool_input)
        order = self.store.find_order(payload.order_id)
        if order is None:
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Order '{payload.order_id}' was not found.",
            )
        return ToolResult.ok(
            tool_name=self.name,
            records=[order],
            confidence=1.0,
            metadata={"agent_name": context.agent_name if context else None},
        )


class OrderAddressUpdateTool(AgentTool):
    """修改订单收货地址的基础工具。"""

    name = "order_address_update"
    description = "Update the shipping address saved on an existing order."
    capability_type = "action"
    input_model = OrderAddressUpdateInput

    def __init__(self, store: CommerceDataStore) -> None:
        self.store = store

    def invoke(
        self,
        tool_input: BaseModel | dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """更新指定订单的收货地址，并返回更新后的订单信息。"""
        payload = self.parse_input(tool_input)
        order = self.store.update_order_address(payload.order_id, payload.new_address)
        if order is None:
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Order '{payload.order_id}' was not found.",
            )
        return ToolResult.ok(
            tool_name=self.name,
            records=[order],
            confidence=0.95,
            metadata={"updated_by_agent": context.agent_name if context else None},
        )


class ReturnTicketCreateTool(AgentTool):
    """创建退换货工单的基础工具。"""

    name = "return_ticket_create"
    description = "Create a return or exchange service ticket for an order."
    capability_type = "action"
    input_model = ReturnTicketCreateInput

    def __init__(self, store: CommerceDataStore) -> None:
        self.store = store

    def invoke(
        self,
        tool_input: BaseModel | dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """为指定订单创建退货工单，并附带关联订单状态。"""
        payload = self.parse_input(tool_input)
        order = self.store.find_order(payload.order_id)
        if order is None:
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Order '{payload.order_id}' was not found.",
            )

        ticket = self.store.create_service_ticket(
            {
                "ticket_id": f"RET-{uuid4().hex[:10]}",
                "ticket_type": "return",
                "order_id": payload.order_id,
                "reason": payload.reason,
                "items": payload.items,
                "status": "open",
                "created_at": _utc_now(),
                "created_by_agent": context.agent_name if context else None,
            }
        )
        return ToolResult.ok(
            tool_name=self.name,
            records=[ticket],
            confidence=0.9,
            metadata={"linked_order_status": order.get("status")},
        )


class ComplaintTicketCreateTool(AgentTool):
    """创建投诉工单的基础工具。"""

    name = "complaint_ticket_create"
    description = "Create a customer complaint ticket for order-related problems."
    capability_type = "action"
    input_model = ComplaintTicketCreateInput

    def __init__(self, store: CommerceDataStore) -> None:
        self.store = store

    def invoke(
        self,
        tool_input: BaseModel | dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """为指定订单创建投诉工单，并保留联系信息。"""
        payload = self.parse_input(tool_input)
        order = self.store.find_order(payload.order_id)
        if order is None:
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Order '{payload.order_id}' was not found.",
            )

        ticket = self.store.create_service_ticket(
            {
                "ticket_id": f"COM-{uuid4().hex[:10]}",
                "ticket_type": "complaint",
                "order_id": payload.order_id,
                "message": payload.message,
                "contact": payload.contact,
                "status": "open",
                "created_at": _utc_now(),
                "created_by_agent": context.agent_name if context else None,
            }
        )
        return ToolResult.ok(
            tool_name=self.name,
            records=[ticket],
            confidence=0.9,
            metadata={"linked_order_status": order.get("status")},
        )


def build_commerce_tools(
    app_settings: AppSettings | None = None,
    *,
    store: CommerceDataStore | None = None,
) -> tuple[AgentTool, ...]:
    """按统一顺序构建订单与售后工具集合。"""
    current_settings = app_settings or settings
    data_store = store or CommerceDataStore(data_dir=current_settings.data_dir)
    return (
        OrderStatusLookupTool(data_store),
        OrderAddressUpdateTool(data_store),
        ReturnTicketCreateTool(data_store),
        ComplaintTicketCreateTool(data_store),
    )


def _utc_now() -> str:
    """生成 UTC ISO 时间戳，供工具写入更新时间和建单时间。"""
    return datetime.now(UTC).isoformat()

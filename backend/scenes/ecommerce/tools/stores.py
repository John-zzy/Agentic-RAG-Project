from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from backend.platform.tools import BaseJsonStore


PRODUCTS_FILE_NAME = "products.json"
ORDERS_FILE_NAME = "orders.json"
SERVICE_TICKETS_FILE_NAME = "service_tickets.json"


@dataclass
class ProductCatalogStore(BaseJsonStore):
    """封装本地商品目录读取，供库存与详情工具复用。"""

    def load_products(self) -> list[dict[str, Any]]:
        """读取商品列表；若文件不存在则返回空列表。"""
        return self._load_json_list(PRODUCTS_FILE_NAME)

    def find_product(self, product_id: str) -> dict[str, Any] | None:
        """按商品 ID 精确查询结构化商品数据，不存在时返回 None。"""
        for product in self.load_products():
            if str(product.get("product_id")) == product_id:
                return product
        return None

    def find_product_by_query(self, query: str) -> dict[str, Any] | None:
        """在 query 中出现明确商品名时，直接解析对应商品。"""
        normalized_query = query.lower()
        for product in self.load_products():
            name = str(product.get("name", "")).lower()
            if name and name in normalized_query:
                return product
        return None


@dataclass
class CommerceDataStore(BaseJsonStore):
    """封装订单与售后工单的本地 JSON 读写，隔离工具层的数据访问细节。"""

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
        """按订单号精确查找订单，不存在时返回 None。"""
        for order in self.load_orders():
            if order.get("order_id") == order_id:
                return order
        return None

    def update_order_address(self, order_id: str, new_address: str) -> dict[str, Any] | None:
        """更新指定订单的收货地址，并在命中时补充更新时间戳。"""
        orders = self.load_orders()
        for order in orders:
            if order.get("order_id") == order_id:
                order["shipping_address"] = new_address
                order["updated_at"] = utc_now()
                self.save_orders(orders)
                return order
        return None

    def create_service_ticket(self, payload: dict[str, Any]) -> dict[str, Any]:
        """追加一条售后工单记录并返回该记录。"""
        tickets = self.load_service_tickets()
        tickets.append(payload)
        self.save_service_tickets(tickets)
        return payload


def utc_now() -> str:
    """生成 UTC ISO 时间戳，供工具写入更新时间和建单时间。"""
    return datetime.now(UTC).isoformat()

from __future__ import annotations

import json
from typing import Any

from backend.platform.knowledge.base.store import VectorStoreDocument


def build_product_document(product: dict[str, Any]) -> VectorStoreDocument:
    """将商品结构化数据转换为向量库商品文档。"""
    product_id = str(product["product_id"])
    specs = product.get("specs", {})
    inventory = product.get("inventory", {})
    specs_text = "；".join(f"{key}: {value}" for key, value in specs.items())
    inventory_text = "；".join(f"{key}: {value}" for key, value in inventory.items())
    content = (
        f"商品名称：{product['name']}\n"
        f"商品分类：{product['category']}\n"
        f"商品描述：{product['description']}\n"
        f"商品价格：{product['price']} {product['currency']}\n"
        f"商品规格：{specs_text}\n"
        f"库存信息：{inventory_text}"
    )

    return VectorStoreDocument(
        id=product_id,
        content=content,
        metadata={
            "source": "product",
            "product_id": product_id,
            "name": str(product["name"]),
            "category": str(product["category"]),
            "price": float(product["price"]),
            "currency": str(product["currency"]),
            "inventory_status": str(inventory.get("status", "unknown")),
            "inventory_quantity": int(inventory.get("quantity", 0)),
            "warehouse": str(inventory.get("warehouse", "")),
            "raw_payload": json.dumps(product, ensure_ascii=False),
        },
    )


def build_review_document(review: dict[str, Any]) -> VectorStoreDocument:
    """将评论结构化数据转换为向量库评论文档。"""
    review_id = str(review["review_id"])
    product_id = str(review["product_id"])
    content = (
        f"{review['title']}。"
        f"{review['content']}。"
        f"商品 {product_id}，评分 {review['rating']} 星。"
        f"用户 {review['user_name']}，创建时间 {review['created_at']}。"
    )

    return VectorStoreDocument(
        id=review_id,
        content=content,
        metadata={
            "source": "review",
            "review_id": review_id,
            "product_id": product_id,
            "rating": int(review["rating"]),
            "title": str(review["title"]),
            "user_name": str(review["user_name"]),
            "created_at": str(review["created_at"]),
        },
    )


def build_order_document(order: dict[str, Any]) -> VectorStoreDocument:
    """将订单结构化数据转换为向量库订单文档。"""
    order_id = str(order["order_id"])
    items_text = "；".join(
        f"{item['name']}(商品ID:{item['product_id']}) x{item['quantity']} 单价{item['unit_price']}"
        for item in order.get("items", [])
    )
    content = (
        f"订单编号：{order_id}\n"
        f"用户ID：{order.get('user_id', '')}\n"
        f"订单状态：{order.get('status', '')}\n"
        f"创建时间：{order.get('created_at', '')}\n"
        f"收货地址：{order.get('shipping_address', '')}\n"
        f"订单商品：{items_text}\n"
        f"订单金额：{order.get('total_amount', 0)} {order.get('currency', '')}"
    )

    return VectorStoreDocument(
        id=order_id,
        content=content,
        metadata={
            "source": "order",
            "order_id": order_id,
            "user_id": str(order.get("user_id", "")),
            "status": str(order.get("status", "")),
            "created_at": str(order.get("created_at", "")),
            "paid_at": str(order.get("paid_at", "")),
            "shipped_at": str(order.get("shipped_at", "")),
            "delivered_at": str(order.get("delivered_at", "")),
            "total_amount": float(order.get("total_amount", 0)),
            "currency": str(order.get("currency", "")),
            "carrier": str(order.get("carrier", "")),
            "tracking_no": str(order.get("tracking_no", "")),
            "shipping_address": str(order.get("shipping_address", "")),
        },
    )

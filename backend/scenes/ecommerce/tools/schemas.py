from __future__ import annotations

from pydantic import BaseModel, Field


class SemanticSearchInput(BaseModel):
    """语义检索工具的输入参数。"""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class ReviewSemanticSearchInput(SemanticSearchInput):
    """评价语义检索工具的输入参数。"""

    product_id: str | None = None


class ProductLookupInput(BaseModel):
    """商品精确查询类工具的输入参数。"""

    product_id: str = Field(min_length=1)


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

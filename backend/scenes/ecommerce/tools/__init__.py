from __future__ import annotations

from langchain_core.tools import BaseTool

from backend.platform.config.settings import AppSettings, settings
from backend.platform.rag.contracts import RetrievalTool
from backend.platform.tools import build_retrieval_tool, build_scene_structured_tool
from backend.scenes.ecommerce.knowledge_service import KnowledgeService, create_knowledge_service
from backend.scenes.ecommerce.tools.complaint_ticket_create import ComplaintTicketCreateTool
from backend.scenes.ecommerce.tools.inventory_lookup import InventoryLookupTool
from backend.scenes.ecommerce.tools.order_address_update import OrderAddressUpdateTool
from backend.scenes.ecommerce.tools.order_semantic_search import OrderSemanticSearchTool
from backend.scenes.ecommerce.tools.order_status_lookup import OrderStatusLookupTool
from backend.scenes.ecommerce.tools.product_detail_lookup import ProductDetailLookupTool
from backend.scenes.ecommerce.tools.product_semantic_search import ProductSemanticSearchTool
from backend.scenes.ecommerce.tools.return_ticket_create import ReturnTicketCreateTool
from backend.scenes.ecommerce.tools.review_semantic_search import ReviewSemanticSearchTool
from backend.scenes.ecommerce.tools.stores import (
    CommerceDataStore,
    ProductCatalogStore,
    SERVICE_TICKETS_FILE_NAME,
)


def build_ecommerce_structured_retrieval_tools(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: KnowledgeService | None = None,
    product_store: ProductCatalogStore | None = None,
) -> tuple[BaseTool, ...]:
    """构建电商检索工具的 StructuredTool 暴露形态。"""
    return tuple(
        build_scene_structured_tool(tool)
        for tool in _build_ecommerce_retrieval_tool_instances(
            app_settings=app_settings,
            knowledge_service=knowledge_service,
            product_store=product_store,
        )
    )


def build_ecommerce_agentic_retrieval_tools(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: KnowledgeService | None = None,
    product_store: ProductCatalogStore | None = None,
) -> tuple[RetrievalTool, ...]:
    """构建电商检索工具的 Agentic Retrieval 暴露形态。"""
    return tuple(
        build_retrieval_tool(tool)
        for tool in _build_ecommerce_retrieval_tool_instances(
            app_settings=app_settings,
            knowledge_service=knowledge_service,
            product_store=product_store,
        )
    )


def build_ecommerce_action_tools(
    app_settings: AppSettings | None = None,
    *,
    store: CommerceDataStore | None = None,
) -> tuple[BaseTool, ...]:
    """构建电商订单与售后 action 工具。"""
    current_settings = app_settings or settings
    data_store = store or CommerceDataStore(data_dir=current_settings.data_dir)
    return tuple(
        build_scene_structured_tool(tool)
        for tool in (
            OrderStatusLookupTool(store=data_store),
            OrderAddressUpdateTool(store=data_store),
            ReturnTicketCreateTool(store=data_store),
            ComplaintTicketCreateTool(store=data_store),
        )
    )


def _build_ecommerce_retrieval_tool_instances(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: KnowledgeService | None = None,
    product_store: ProductCatalogStore | None = None,
) -> tuple[
    ProductSemanticSearchTool,
    ReviewSemanticSearchTool,
    OrderSemanticSearchTool,
    InventoryLookupTool,
    ProductDetailLookupTool,
]:
    """构建电商检索工具类实例；scene 决定是否纳入工具范围。"""
    current_settings = app_settings or settings
    resolved_knowledge_service = knowledge_service or create_knowledge_service(current_settings)
    resolved_product_store = product_store or ProductCatalogStore(data_dir=current_settings.data_dir)
    return (
        ProductSemanticSearchTool(
            knowledge_service=resolved_knowledge_service,
            product_store=resolved_product_store,
        ),
        ReviewSemanticSearchTool(knowledge_service=resolved_knowledge_service),
        OrderSemanticSearchTool(knowledge_service=resolved_knowledge_service),
        InventoryLookupTool(product_store=resolved_product_store),
        ProductDetailLookupTool(product_store=resolved_product_store),
    )


__all__ = [
    "CommerceDataStore",
    "ComplaintTicketCreateTool",
    "InventoryLookupTool",
    "OrderAddressUpdateTool",
    "OrderSemanticSearchTool",
    "OrderStatusLookupTool",
    "ProductCatalogStore",
    "ProductDetailLookupTool",
    "ProductSemanticSearchTool",
    "ReturnTicketCreateTool",
    "ReviewSemanticSearchTool",
    "SERVICE_TICKETS_FILE_NAME",
    "build_ecommerce_action_tools",
    "build_ecommerce_agentic_retrieval_tools",
    "build_ecommerce_structured_retrieval_tools",
]

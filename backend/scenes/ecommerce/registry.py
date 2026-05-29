from __future__ import annotations

from langchain_core.tools import BaseTool

from backend.platform.config.settings import AppSettings
from backend.platform.tools import ToolRegistration, ToolRegistry
from backend.scenes.ecommerce.definition import build_ecommerce_scene_definition


def build_default_tool_registry(app_settings: AppSettings | None = None) -> ToolRegistry:
    """构建默认工具注册表，scene 负责给出当前范围内的工具集合。"""
    current_settings = app_settings or AppSettings()
    registry = ToolRegistry()
    scene_definition = build_ecommerce_scene_definition(app_settings=current_settings)
    for tool in scene_definition.build_tools():
        registry.register(_build_registration(tool))

    return registry


def _build_registration(tool: BaseTool) -> ToolRegistration:
    """根据工具名生成默认分组和访问范围。"""
    if tool.name == "order_semantic_search":
        return ToolRegistration(
            tool=tool,
            group="retrieval",
            allowed_agents=(),
            expose_via_mcp=False,
        )
    if tool.name in {
        "product_semantic_search",
        "review_semantic_search",
        "inventory_lookup",
        "product_detail_lookup",
    }:
        return ToolRegistration(
            tool=tool,
            group="retrieval",
            allowed_agents=("shopping_agent",),
            expose_via_mcp=tool.name == "inventory_lookup",
        )
    return _build_commerce_registration(tool)


def _build_commerce_registration(tool: BaseTool) -> ToolRegistration:
    """根据 commerce 工具名称生成默认分组和 Agent 白名单配置。"""
    if tool.name == "order_status_lookup":
        return ToolRegistration(
            tool=tool,
            group="commerce_order",
            allowed_agents=("order_agent", "after_sale_agent"),
            expose_via_mcp=True,
        )
    if tool.name == "order_address_update":
        return ToolRegistration(
            tool=tool,
            group="commerce_order",
            allowed_agents=("order_agent",),
            expose_via_mcp=True,
        )
    if tool.name in {"return_ticket_create", "complaint_ticket_create"}:
        return ToolRegistration(
            tool=tool,
            group="commerce_after_sale",
            allowed_agents=("after_sale_agent",),
            expose_via_mcp=True,
        )
    raise ValueError(f"Unsupported commerce tool registration: {tool.name}")

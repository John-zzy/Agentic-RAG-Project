from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from backend.config.settings import AppSettings
from backend.tools.ecommerce.commerce import build_commerce_tools
from backend.tools.ecommerce.retrieval import build_retrieval_tools


@dataclass(frozen=True)
class ToolRegistration:
    """描述一个工具在注册表中的分组、白名单和暴露方式。"""

    tool: BaseTool
    group: str
    allowed_agents: tuple[str, ...]
    expose_via_mcp: bool = False


class ToolRegistry:
    """集中维护工具注册、查询和按 Agent 过滤的注册表。"""

    def __init__(self) -> None:
        self._registrations: dict[str, ToolRegistration] = {}

    def register(self, registration: ToolRegistration) -> None:
        """注册一个工具；同名工具会被后续注册覆盖。"""
        self._registrations[registration.tool.name] = registration

    def get_tool(self, name: str) -> BaseTool:
        """按工具名获取具体工具实例。"""
        return self._registrations[name].tool

    def list_tools(self) -> list[ToolRegistration]:
        """返回所有工具注册信息。"""
        return list(self._registrations.values())

    def list_tools_for_agent(self, agent_name: str) -> list[ToolRegistration]:
        """按 Agent 白名单过滤可使用的工具。"""
        return [
            registration
            for registration in self._registrations.values()
            if agent_name in registration.allowed_agents
        ]

    def list_mcp_tools(self) -> list[ToolRegistration]:
        """返回需要通过 MCP 远程暴露的工具集合。"""
        return [
            registration
            for registration in self._registrations.values()
            if registration.expose_via_mcp
        ]


def build_default_tool_registry(app_settings: AppSettings | None = None) -> ToolRegistry:
    """构建默认工具注册表，集中注入 retrieval 与 commerce 两类工具。"""
    current_settings = app_settings or AppSettings()
    registry = ToolRegistry()

    for tool in build_retrieval_tools(current_settings):
        registry.register(
            ToolRegistration(
                tool=tool,
                group="retrieval",
                allowed_agents=("shopping_agent",),
                expose_via_mcp=tool.name == "inventory_lookup",
            )
        )

    for tool in build_commerce_tools(current_settings):
        registry.register(_build_commerce_registration(tool))

    return registry


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

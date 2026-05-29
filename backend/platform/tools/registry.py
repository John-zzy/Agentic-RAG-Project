from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class ToolRegistration:
    """描述一个工具在注册表中的分组、白名单和暴露方式。"""

    tool: BaseTool
    group: str
    allowed_agents: tuple[str, ...]
    expose_via_mcp: bool = False


class ToolRegistry:
    """只管理工具元数据和访问范围，不承载具体业务逻辑。"""

    def __init__(self) -> None:
        self._registrations: dict[str, ToolRegistration] = {}

    def register(self, registration: ToolRegistration) -> None:
        """注册工具；同名工具直接拒绝，避免装配阶段悄悄覆盖。"""
        tool_name = registration.tool.name
        if tool_name in self._registrations:
            raise ValueError(f"Duplicate tool registration: {tool_name}")
        self._registrations[tool_name] = registration

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

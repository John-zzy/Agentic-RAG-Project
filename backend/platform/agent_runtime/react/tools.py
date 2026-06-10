from __future__ import annotations

from langchain.tools import BaseTool

from collections.abc import Mapping
from typing import Any

from backend.platform.agent_runtime.tooling.langchain import (
    build_langchain_tools_from_executor,
)
from backend.platform.agent_runtime.tooling.executor import ToolExecutor


def build_react_tools(
    tool_executor: ToolExecutor,
    *,
    default_inputs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[BaseTool, ...]:
    """Build LangChain tools for the ReAct provider from platform tool definitions."""

    return build_langchain_tools_from_executor(
        tool_executor,
        default_inputs=default_inputs,
    )

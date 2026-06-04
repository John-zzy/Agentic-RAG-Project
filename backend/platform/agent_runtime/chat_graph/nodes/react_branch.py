from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_react_branch_node(dependencies: ChatGraphDependencies):
    """归一化 ReAct 分支输出。"""

    prepared = dependencies.prepared

    def react_branch(state: RuntimeGraphState) -> dict[str, Any]:
        if str(state.get("agent_mode") or prepared.agent_mode) != "react":
            return {}
        react_run = state.get("react_run") or getattr(prepared, "react_run", None)
        if react_run is None:
            return {}
        return {
            "react_run": dict(react_run),
            "current_turn_id": state.get("current_turn_id")
            or getattr(prepared, "current_turn_id", None),
            "current_tool_call": state.get("current_tool_call")
            or getattr(prepared, "current_tool_call", None),
        }

    return react_branch


from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_plan_branch_node(dependencies: ChatGraphDependencies):
    """归一化 Plan 分支输出。"""

    prepared = dependencies.prepared

    def plan_branch(state: RuntimeGraphState) -> dict[str, Any]:
        if str(state.get("agent_mode") or prepared.agent_mode) != "plan":
            return {}
        plan_run = state.get("plan_run") or getattr(prepared, "plan_run", None)
        if plan_run is None:
            return {}
        return {
            "plan_run": dict(plan_run),
            "current_step_id": state.get("current_step_id")
            or getattr(prepared, "current_step_id", None),
            "current_tool_call": state.get("current_tool_call")
            or getattr(prepared, "current_tool_call", None),
        }

    return plan_branch


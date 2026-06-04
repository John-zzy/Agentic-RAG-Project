from __future__ import annotations

from typing import Any

from backend.application.runtime.graph_runtime_parts.chat_graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_select_mode_node(dependencies: ChatGraphDependencies):
    """确认本轮使用的顶层 Agent mode。"""

    prepared = dependencies.prepared

    def select_mode(state: RuntimeGraphState) -> dict[str, Any]:
        agent_mode_reason = state.get("agent_mode_reason") or getattr(
            prepared,
            "agent_mode_reason",
            None,
        )
        return {
            "agent_mode": str(state.get("agent_mode") or prepared.agent_mode),
            "agent_mode_reason": agent_mode_reason,
            "agent_mode_signals": dict(
                state.get("agent_mode_signals")
                or getattr(prepared, "agent_mode_signals", None)
                or {}
            ),
        }

    return select_mode

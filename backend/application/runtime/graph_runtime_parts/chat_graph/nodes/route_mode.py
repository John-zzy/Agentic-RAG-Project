from __future__ import annotations

from typing import Any

from backend.application.runtime.graph_runtime_parts.chat_graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_route_mode_node(dependencies: ChatGraphDependencies):
    """记录 route_mode 边界，不改变业务结果。"""

    del dependencies

    def route_mode(state: RuntimeGraphState) -> dict[str, Any]:
        return {
            "metadata": {
                **dict(state.get("metadata") or {}),
                "chat_graph_route_ready": True,
            }
        }

    return route_mode


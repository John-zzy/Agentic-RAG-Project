from __future__ import annotations

from typing import Any

from backend.platform.rag.orchestration.retrieval_graph.config import AgenticRagGraphDependencies
from backend.platform.rag.orchestration.retrieval_graph.state import AgenticRagGraphState


def build_route_next_action_node(_dependencies: AgenticRagGraphDependencies):
    """仅作为路由边界，保持 route 判断留在 edges。"""

    def route_next_action(state: AgenticRagGraphState) -> dict[str, Any]:
        decision = state.get("current_decision")
        if decision is None:
            raise ValueError("Agentic RAG route node requires a sufficiency decision.")
        return {
            "route_next_action": decision.next_action,
        }

    return route_next_action


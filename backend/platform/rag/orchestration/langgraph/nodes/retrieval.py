from __future__ import annotations

from typing import Any

from backend.platform.rag.orchestration.langgraph.config import AgenticRagGraphDependencies
from backend.platform.rag.orchestration.langgraph.state import (
    AgenticRagGraphState,
    _documents_to_payload,
)


def build_retrieval_node(dependencies: AgenticRagGraphDependencies):
    """执行检索工具本体，只负责取数，不夹带路由。"""

    retriever = dependencies.retriever

    def retrieval(state: AgenticRagGraphState) -> dict[str, Any]:
        plan = state["plan"]
        result = retriever._run_tool(plan, dependencies.run_manager)
        return {
            "current_result": result,
            "results": [*state["results"], result],
            "candidate_docs": _documents_to_payload(result.documents),
            "route_next_action": None,
        }

    return retrieval

from __future__ import annotations

from typing import Any

from backend.platform.rag.contracts import RetrievalContext
from backend.platform.rag.orchestration.retrieval_graph.config import AgenticRagGraphDependencies
from backend.platform.rag.orchestration.retrieval_graph.state import AgenticRagGraphState


def build_sufficiency_check_node(dependencies: AgenticRagGraphDependencies):
    """调用现有 judge，只写决策结果，不在这里做分支跳转。"""

    retriever = dependencies.retriever

    def sufficiency_check(state: AgenticRagGraphState) -> dict[str, Any]:
        plan = state["plan"]
        decision = retriever._judge(
            RetrievalContext(
                plan=plan,
                results=list(state["results"]),
                documents=list(state["documents"]),
            ),
            dependencies.run_manager,
        )
        return {
            "current_decision": decision,
            "follow_up_question": decision.follow_up_question,
            "route_next_action": decision.next_action,
        }

    return sufficiency_check


from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_resolve_answer_mode_node(dependencies: ChatGraphDependencies):
    """把 answer_mode 归一化为后续 final synthesis 可消费的状态。"""

    prepared = dependencies.prepared

    def resolve_answer_mode(state: RuntimeGraphState) -> dict[str, Any]:
        state_citations = list(state.get("citations") or [])
        prepared_citations = [
            citation.model_dump() if hasattr(citation, "model_dump") else dict(citation)
            for citation in prepared.citations
        ]
        retrieval_trace = state.get("retrieval_trace")
        if not retrieval_trace:
            retrieval_trace = prepared.retrieval_trace.model_dump()
        return {
            "answer_mode": str(state.get("answer_mode") or prepared.answer_mode),
            "knowledge_used": bool(state.get("knowledge_used", prepared.knowledge_used)),
            "citations": state_citations or prepared_citations,
            "retrieval_trace": dict(retrieval_trace),
        }

    return resolve_answer_mode



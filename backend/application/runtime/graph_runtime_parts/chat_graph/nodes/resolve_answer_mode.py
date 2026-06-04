from __future__ import annotations

from typing import Any

from backend.application.runtime.graph_runtime_parts.chat_graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_resolve_answer_mode_node(dependencies: ChatGraphDependencies):
    """把 answer_mode 归一化为后续 final synthesis 可消费的状态。"""

    prepared = dependencies.prepared

    def resolve_answer_mode(state: RuntimeGraphState) -> dict[str, Any]:
        return {
            "answer_mode": str(state.get("answer_mode") or prepared.answer_mode),
            "knowledge_used": bool(state.get("knowledge_used", prepared.knowledge_used)),
            "citations": list(state.get("citations") or [citation.model_dump() for citation in prepared.citations]),
            "retrieval_trace": dict(
                state.get("retrieval_trace") or prepared.retrieval_trace.model_dump()
            ),
        }

    return resolve_answer_mode


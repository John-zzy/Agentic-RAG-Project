from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_persist_turn_node(dependencies: ChatGraphDependencies):
    """把归一化后的 branch result 写成最终 graph state。"""

    build_agent_runtime_success_update = dependencies.build_agent_runtime_success_update

    def persist_turn(state: RuntimeGraphState) -> dict[str, Any]:
        citations = [
            citation if isinstance(citation, dict) else citation.model_dump()
            for citation in list(state.get("citations") or [])
        ]
        answer = str(state.get("answer") or "")
        knowledge_used = bool(state.get("knowledge_used", False))
        agent_update = build_agent_runtime_success_update(
            state=state,
            answer=answer,
            citations=citations,
            knowledge_used=knowledge_used,
        )
        return {
            "status": "succeeded",
            "final_state": "succeeded",
            "state_event": "success",
            **agent_update,
        }

    return persist_turn




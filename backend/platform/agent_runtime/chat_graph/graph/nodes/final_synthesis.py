from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from backend.platform.agent_runtime.chat_graph.graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_final_synthesis_node(dependencies: ChatGraphDependencies):
    """调用现有 answer builder 产出最终回答。"""

    prepared = dependencies.prepared
    answer_builder = dependencies.answer_builder

    def final_synthesis(state: RuntimeGraphState) -> dict[str, Any]:
        answer_prepared = (
            dependencies.build_prepared_from_state(prepared, state)
            if dependencies.build_prepared_from_state is not None
            else prepared
        )
        answer, citations = answer_builder(answer_prepared)
        return {
            "answer": answer,
            "citations": [citation.model_dump() for citation in citations],
            "knowledge_used": answer_prepared.knowledge_used,
            "messages": [AIMessage(content=answer)],
        }

    return final_synthesis



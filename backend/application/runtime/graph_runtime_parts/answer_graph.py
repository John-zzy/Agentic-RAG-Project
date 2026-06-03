from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from backend.application.runtime.graph_runtime_parts.contracts import AnswerBuilder, PreparedGraphTurn
from backend.platform.workflow.langgraph.state import RuntimeGraphState


class AnswerGraphMixin:
    def _compile_answer_graph(
        self,
        *,
        prepared: PreparedGraphTurn,
        answer_builder: AnswerBuilder,
    ) -> Any:
        builder = StateGraph(RuntimeGraphState)

        def answer_node(state: RuntimeGraphState) -> dict[str, Any]:
            answer, citations = answer_builder(prepared)
            agent_update = self._build_agent_runtime_success_update(
                state=state,
                answer=answer,
                citations=citations,
                knowledge_used=prepared.knowledge_used,
            )
            # LangGraph 只记运行状态；聊天记录仍由 ChatService 写入 session 表。
            return {
                "answer": answer,
                "citations": [citation.model_dump() for citation in citations],
                "knowledge_used": prepared.knowledge_used,
                "messages": [AIMessage(content=answer)],
                "status": "succeeded",
                "final_state": "succeeded",
                "state_event": "success",
                **agent_update,
            }

        builder.add_node("answer", answer_node)
        builder.add_edge(START, "answer")
        builder.add_edge("answer", END)
        return builder.compile(checkpointer=self.checkpointer)




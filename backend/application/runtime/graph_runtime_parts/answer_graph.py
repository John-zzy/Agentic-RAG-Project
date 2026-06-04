from __future__ import annotations

from typing import Any

from backend.application.runtime.graph_runtime_parts.chat_graph.config import ChatGraphDependencies
from backend.application.runtime.graph_runtime_parts.chat_graph.graph import build_chat_graph
from backend.application.runtime.graph_runtime_parts.contracts import AnswerBuilder, PreparedGraphTurn


class AnswerGraphMixin:
    def _compile_answer_graph(
        self,
        *,
        prepared: PreparedGraphTurn,
        answer_builder: AnswerBuilder,
    ) -> Any:
        return build_chat_graph(
            ChatGraphDependencies(
                prepared=prepared,
                answer_builder=answer_builder,
                build_agent_runtime_success_update=self._build_agent_runtime_success_update,
            ),
            checkpointer=self.checkpointer,
        )




from __future__ import annotations

from typing import Any, Callable

from backend.platform.agent_runtime.chat_graph.config import ChatGraphDependencies
from backend.platform.agent_runtime.chat_graph.graph import build_chat_graph
from backend.platform.agent_runtime.chat_graph.contracts import AnswerBuilder, PreparedGraphTurn


class AnswerGraphMixin:
    def _compile_answer_graph(
        self,
        *,
        prepared: PreparedGraphTurn,
        answer_builder: AnswerBuilder,
        select_agent_mode: Callable[[PreparedGraphTurn], dict[str, Any]] | None = None,
        run_agent_runtime: Callable[[PreparedGraphTurn, dict[str, Any]], dict[str, Any]] | None = None,
        build_prepared_from_state: Callable[
            [PreparedGraphTurn, dict[str, Any]],
            PreparedGraphTurn,
        ] | None = None,
        build_hitl_wait_update: Callable[
            [PreparedGraphTurn, dict[str, Any]],
            dict[str, Any],
        ] | None = None,
    ) -> Any:
        return build_chat_graph(
            ChatGraphDependencies(
                prepared=prepared,
                answer_builder=answer_builder,
                build_agent_runtime_success_update=self._build_agent_runtime_success_update,
                select_agent_mode=select_agent_mode,
                run_agent_runtime=run_agent_runtime,
                build_prepared_from_state=build_prepared_from_state,
                build_hitl_wait_update=build_hitl_wait_update,
            ),
            checkpointer=self.checkpointer,
        )





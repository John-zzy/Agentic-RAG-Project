from __future__ import annotations

from threading import Lock
from typing import Any

from backend.platform.agent_runtime.chat_graph.contracts import (
    AnswerBuilder,
    HistoryLoader,
    PreparedGraphTurn,
    RuntimeGraphResult,
)
from backend.platform.agent_runtime.chat_graph.runtime_parts.agent_state import (
    AgentRuntimeStateProjectionMixin,
)
from backend.platform.agent_runtime.chat_graph.runtime_parts.answer_graph import (
    AnswerGraphMixin,
)
from backend.platform.agent_runtime.chat_graph.runtime_parts.hitl import HitlRuntimeMixin
from backend.platform.agent_runtime.chat_graph.runtime_parts.state_store import (
    RuntimeStateStoreMixin,
)
from backend.platform.agent_runtime.graph_logging import (
    log_graph_invoke_end,
    log_graph_invoke_error,
    log_graph_invoke_start,
)
from backend.platform.workflow.langgraph.checkpointer import SQLiteLangGraphCheckpointer
from backend.platform.workflow.langgraph.lifecycle import GraphRunLifecycleRecorder


class ChatGraphRuntime(
    RuntimeStateStoreMixin,
    AnswerGraphMixin,
    HitlRuntimeMixin,
    AgentRuntimeStateProjectionMixin,
):
    """平台层 ChatGraph 运行入口，负责 checkpoint、HITL 和 run lifecycle。"""

    def __init__(
        self,
        *,
        checkpointer: SQLiteLangGraphCheckpointer,
        lifecycle: GraphRunLifecycleRecorder | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.lifecycle = lifecycle or GraphRunLifecycleRecorder()
        # 同一等待点的 resume 可能带副作用，单进程内串行化避免重复执行。
        self._resume_lock = Lock()

    def invoke(
        self,
        *,
        prepared: PreparedGraphTurn,
        answer_builder: AnswerBuilder,
        history_loader: HistoryLoader,
        select_agent_mode: Any | None = None,
        build_react_graph_deps: Any | None = None,
        build_plan_graph_deps: Any | None = None,
        build_prepared_from_state: Any | None = None,
        build_hitl_wait_update: Any | None = None,
    ) -> RuntimeGraphResult:
        """同步执行 ChatGraph，返回完整的运行结果。"""
        config = self.build_config(prepared)
        run = self.lifecycle.create_run(
            thread_id=prepared.session_id,
            request_id=prepared.request_id,
            metadata=dict(config["metadata"]),
        )
        self._mark_agent_run_started(run=run, prepared=prepared)

        graph = self._compile_answer_graph(
            prepared=prepared,
            answer_builder=answer_builder,
            select_agent_mode=select_agent_mode,
            build_react_graph_deps=build_react_graph_deps,
            build_plan_graph_deps=build_plan_graph_deps,
            build_prepared_from_state=build_prepared_from_state,
            build_hitl_wait_update=build_hitl_wait_update,
        )

        try:
            input_state = self.build_input_state(
                prepared=prepared,
                history_loader=history_loader,
                config=config,
                run_id=run.run_id,
            )
            log_graph_invoke_start(graph_name="chat_graph", payload=input_state)
            output = graph.invoke(
                input_state,
                config,
            )
            log_graph_invoke_end(graph_name="chat_graph", payload=output)
        except Exception as exc:
            log_graph_invoke_error(graph_name="chat_graph", error=exc)
            self.lifecycle.mark_failed(run, exc)
            raise

        if output.get("status") == "waiting_user":
            self.lifecycle.mark_waiting_user(run)
        else:
            self.lifecycle.mark_succeeded(run)

        return RuntimeGraphResult(
            answer=str(output.get("answer") or ""),
            citations=list(output.get("citations") or []),
            state=output,
            config=config,
            run_id=run.run_id,
        )

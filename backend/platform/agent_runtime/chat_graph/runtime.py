from __future__ import annotations

from threading import Lock
from typing import Any

from backend.platform.agent_runtime.chat_graph.contracts import (
    AnswerBuilder,
    HistoryLoader,
    PreparedGraphTurn,
    RuntimeGraphResult,
)
from backend.platform.agent_runtime.chat_graph.projection import (
    project_runtime_graph_state,
)
from backend.platform.agent_runtime.chat_graph.runtime_parts.agent_state import (
    AgentRuntimeStateProjectionMixin,
)
from backend.platform.agent_runtime.chat_graph.runtime_parts.answer_graph import (
    AnswerGraphMixin,
)
from backend.platform.agent_runtime.chat_graph.runtime_parts.hitl import HitlRuntimeMixin
from backend.platform.agent_runtime.chat_graph.runtime_parts.recovery import (
    RuntimeRecoveryMixin,
)
from backend.platform.agent_runtime.tooling.idempotency import ToolIdempotencyStore
from backend.platform.agent_runtime.chat_graph.runtime_parts.state_store import (
    RuntimeStateStoreMixin,
)
from backend.platform.agent_runtime.observability.graph_logging import (
    log_graph_invoke_end,
    log_graph_invoke_error,
    log_graph_invoke_start,
)
from backend.platform.workflow.langgraph.checkpointer import SQLiteLangGraphCheckpointer
from backend.platform.workflow.langgraph.guards import GuardedNodeFailureError
from backend.platform.workflow.langgraph.lifecycle import GraphRunLifecycleRecorder


class ChatGraphRuntime(
    RuntimeStateStoreMixin,
    AnswerGraphMixin,
    HitlRuntimeMixin,
    RuntimeRecoveryMixin,
    AgentRuntimeStateProjectionMixin,
):
    """平台层 ChatGraph 运行入口，负责 checkpoint、HITL 和 run lifecycle。"""

    def __init__(
        self,
        *,
        checkpointer: SQLiteLangGraphCheckpointer,
        lifecycle: GraphRunLifecycleRecorder | None = None,
        tool_idempotency_store: ToolIdempotencyStore | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.lifecycle = lifecycle or GraphRunLifecycleRecorder()
        self.tool_idempotency_store = tool_idempotency_store
        self._build_react_deps: Any | None = None
        self._build_plan_graph_deps: Any | None = None
        # 同一等待点的 resume 可能带副作用，单进程内串行化避免重复执行。
        self._resume_lock = Lock()

    def invoke(
        self,
        *,
        prepared: PreparedGraphTurn,
        answer_builder: AnswerBuilder,
        history_loader: HistoryLoader,
        select_agent_mode: Any | None = None,
        build_react_deps: Any | None = None,
        build_plan_graph_deps: Any | None = None,
        build_prepared_from_state: Any | None = None,
        build_hitl_wait_update: Any | None = None,
    ) -> RuntimeGraphResult:
        """同步执行 ChatGraph，返回完整的运行结果。"""
        self._build_react_deps = build_react_deps
        self._build_plan_graph_deps = build_plan_graph_deps
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
            build_react_deps=build_react_deps,
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
            context = self.build_context(prepared=prepared, config=config)
            log_graph_invoke_start(graph_name="chat_graph", payload=input_state)
            output = graph.invoke(
                input_state,
                config,
                context=context,
            )
            log_graph_invoke_end(graph_name="chat_graph", payload=output)
        except Exception as exc:
            log_graph_invoke_error(graph_name="chat_graph", error=exc)
            public_error = _unwrap_guarded_node_error(exc)
            self.lifecycle.mark_failed(run, public_error)
            if public_error is exc:
                raise
            raise public_error

        projection = project_runtime_graph_state(output)

        if projection.status == "waiting_user":
            self.lifecycle.mark_waiting_user(run)
        elif projection.status == "failed":
            self.lifecycle.mark_failed(
                run,
                projection.self_check_failure or "ChatGraph finished with failed status.",
            )
        elif projection.status == "cancelled":
            self.lifecycle.mark_cancelled(run)
        else:
            self.lifecycle.mark_succeeded(run)

        return RuntimeGraphResult(
            answer=projection.answer,
            citations=projection.citations,
            state=projection.state,
            config=config,
            run_id=run.run_id,
        )


def _unwrap_guarded_node_error(exc: Exception) -> Exception:
    # API 边界继续使用业务异常映射，guard failure 只作为内部审计事实保留。
    if isinstance(exc, GuardedNodeFailureError) and isinstance(exc.__cause__, Exception):
        return exc.__cause__
    return exc

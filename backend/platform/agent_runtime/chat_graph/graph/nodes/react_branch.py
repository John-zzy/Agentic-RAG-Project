from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.graph.config import ChatGraphDependencies
from backend.platform.agent_runtime.core.contracts import ReActRun
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_react_branch_node(dependencies: ChatGraphDependencies):
    """在 ChatGraph 分支节点内执行 ReAct 子图。"""

    prepared = dependencies.prepared

    def react_branch(state: RuntimeGraphState) -> dict[str, Any]:
        if str(state.get("agent_mode") or prepared.agent_mode) != "react":
            return {}
        if dependencies.build_react_deps is None:
            return {}

        provider_deps = dependencies.build_react_deps(prepared, state)
        run = provider_deps.build_runtime().run(
            session_id=provider_deps.session_id,
            request_id=provider_deps.request_id,
            user_goal=provider_deps.user_goal,
            react_run_id=provider_deps.react_run_id or f"react-{provider_deps.request_id}",
            initial_run=provider_deps.initial_run
            or _react_run(prepared=prepared, state=state),
        )
        result = dict(provider_deps.project_result(run)) if provider_deps.project_result else {}
        return {
            **_chat_graph_result_fields(result),
            "react_run": run.model_dump() if hasattr(run, "model_dump") else run,
        }

    return react_branch


def _react_run(
    *,
    prepared: Any,
    state: RuntimeGraphState,
) -> ReActRun | None:
    react_run = state.get("react_run") or getattr(prepared, "react_run", None)
    if react_run is None:
        return None
    return react_run if isinstance(react_run, ReActRun) else ReActRun.model_validate(react_run)


def _chat_graph_result_fields(result: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "documents",
        "citations",
        "retrieval_trace",
        "knowledge_used",
        "final_decision",
        "answer_mode",
        "follow_up_question",
        "tool_event",
        "current_turn_id",
        "current_tool_call",
        "tool_observation",
        "agent_mode",
        "agent_mode_reason",
        "agent_mode_signals",
    }
    return {key: result[key] for key in allowed if key in result}


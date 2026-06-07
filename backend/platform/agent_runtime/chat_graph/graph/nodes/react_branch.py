from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.graph.config import ChatGraphDependencies
from backend.platform.agent_runtime.contracts import ReActRun
from backend.platform.agent_runtime.react.graph import build_react_graph
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_react_branch_node(dependencies: ChatGraphDependencies):
    """在 ChatGraph 分支节点内执行 ReAct 子图。"""

    prepared = dependencies.prepared

    def react_branch(state: RuntimeGraphState) -> dict[str, Any]:
        if str(state.get("agent_mode") or prepared.agent_mode) != "react":
            return {}
        if dependencies.build_react_graph_deps is None:
            return {}

        graph_deps = dependencies.build_react_graph_deps(prepared, state)
        graph = build_react_graph(graph_deps)
        result = graph.invoke(_react_graph_input(prepared=prepared, state=state))
        run = result.get("run")
        return {
            **_chat_graph_result_fields(result),
            "react_run": run.model_dump() if hasattr(run, "model_dump") else run,
        }

    return react_branch


def _react_graph_input(
    *,
    prepared: Any,
    state: RuntimeGraphState,
) -> dict[str, Any]:
    react_run = state.get("react_run") or getattr(prepared, "react_run", None)
    if react_run is None:
        return {}
    return {
        "run": react_run
        if isinstance(react_run, ReActRun)
        else ReActRun.model_validate(react_run)
    }


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


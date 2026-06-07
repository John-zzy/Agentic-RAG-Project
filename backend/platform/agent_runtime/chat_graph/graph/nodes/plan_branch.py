from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.graph.config import ChatGraphDependencies
from backend.platform.agent_runtime.contracts import PlanRun
from backend.platform.agent_runtime.plan.graph import build_plan_graph
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_plan_branch_node(dependencies: ChatGraphDependencies):
    """在 ChatGraph 分支节点内执行 Plan 子图。"""

    prepared = dependencies.prepared

    def plan_branch(state: RuntimeGraphState) -> dict[str, Any]:
        if str(state.get("agent_mode") or prepared.agent_mode) != "plan":
            return {}
        if dependencies.build_plan_graph_deps is None:
            return {}

        graph_deps = dependencies.build_plan_graph_deps(prepared, state)
        graph = build_plan_graph(graph_deps)
        result = graph.invoke(_plan_graph_input(prepared=prepared, state=state))
        plan_run = result.get("plan_run")
        return {
            **_chat_graph_result_fields(result),
            "plan_run": (
                plan_run.model_dump()
                if hasattr(plan_run, "model_dump")
                else plan_run
            ),
        }

    return plan_branch


def _plan_graph_input(
    *,
    prepared: Any,
    state: RuntimeGraphState,
) -> dict[str, Any]:
    plan_run = state.get("plan_run") or getattr(prepared, "plan_run", None)
    if plan_run is None:
        return {}
    return {
        "plan_run": plan_run
        if isinstance(plan_run, PlanRun)
        else PlanRun.model_validate(plan_run)
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
        "current_step_id",
        "current_tool_call",
        "tool_observation",
        "agent_mode",
        "agent_mode_reason",
        "agent_mode_signals",
    }
    return {key: result[key] for key in allowed if key in result}


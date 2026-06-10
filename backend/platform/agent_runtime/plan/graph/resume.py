from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from backend.platform.agent_runtime.core.contracts import PlanRun, PlanStep
from backend.platform.agent_runtime.plan.graph.edges import (
    HANDLE_WAITING_USER,
    SELECT_NEXT_STEP,
    SYNTHESIZE_PLAN_RESULT,
    build_handle_retry_edge,
)
from backend.platform.agent_runtime.plan.graph.nodes import (
    build_execute_step_node,
    build_handle_retry_node,
    build_handle_waiting_user_node,
    build_select_next_step_node,
    build_synthesize_plan_result_node,
    build_synthesize_result_node,
)
from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState
from backend.platform.agent_runtime.plan.state_ops import (
    continue_after_approve,
    continue_after_reject,
    continue_after_respond,
)
from backend.platform.agent_runtime.tooling.executor import ToolExecutor
from backend.platform.workflow.langgraph.resume import extract_resume_payload_from_command


@dataclass(frozen=True)
class PlanHitlResumeGraphDependencies:
    """Dependencies for graph-owned Plan HITL continuation."""

    graph_dependencies: PlanGraphDependencies
    tool_executor: ToolExecutor


def build_plan_hitl_resume_graph(
    dependencies: PlanHitlResumeGraphDependencies,
    *,
    checkpointer: Any | None = None,
) -> Any:
    builder = StateGraph(PlanGraphState)
    builder.add_node("resume_waiting_step", _build_resume_waiting_step_node(dependencies))
    builder.add_node(SELECT_NEXT_STEP, build_select_next_step_node(dependencies.graph_dependencies))
    builder.add_node("execute_step", build_execute_step_node(dependencies.graph_dependencies))
    builder.add_node("handle_retry", build_handle_retry_node(dependencies.graph_dependencies))
    builder.add_node(HANDLE_WAITING_USER, build_handle_waiting_user_node(dependencies.graph_dependencies))
    builder.add_node(
        SYNTHESIZE_PLAN_RESULT,
        build_synthesize_plan_result_node(dependencies.graph_dependencies),
    )
    builder.add_node("synthesize_result", build_synthesize_result_node(dependencies.graph_dependencies))
    builder.add_edge(START, "resume_waiting_step")
    builder.add_conditional_edges(
        "resume_waiting_step",
        lambda state: (
            SELECT_NEXT_STEP
            if state.get("route") == SELECT_NEXT_STEP
            else "synthesize_result"
        ),
    )
    builder.add_edge(SELECT_NEXT_STEP, "execute_step")
    builder.add_edge("execute_step", "handle_retry")
    builder.add_conditional_edges("handle_retry", build_handle_retry_edge())
    builder.add_edge(HANDLE_WAITING_USER, "synthesize_result")
    builder.add_edge(SYNTHESIZE_PLAN_RESULT, "synthesize_result")
    builder.add_edge("synthesize_result", END)
    return builder.compile(checkpointer=checkpointer)


def _build_resume_waiting_step_node(dependencies: PlanHitlResumeGraphDependencies):
    def resume_waiting_step(state: PlanGraphState) -> dict[str, Any]:
        command = _coerce_resume_command(state.get("resume_command"))
        command_payload = extract_resume_payload_from_command(command)
        plan_run = _coerce_run(command_payload.get("plan_run"))
        payload = dict(command_payload.get("resume_payload") or {})
        action = str(payload.get("action") or "")
        waiting_step = _waiting_step(plan_run)

        if action == "reject":
            reason = str(payload.get("reason") or "User rejected the waiting Plan step.")
            plan_run = continue_after_reject(
                run=plan_run,
                reason=reason,
                pending_tool_call=_pending_tool_call(
                    command_payload=command_payload,
                    payload=payload,
                ),
            )
            return {
                "plan_run": plan_run,
                "step": waiting_step,
                "answer": "已拒绝该人工等待项，未执行待审批调用。",
                "status": "cancelled",
            }

        if action == "approve":
            proposed_tool_call = dict(
                command_payload.get("proposed_tool_call")
                or payload.get("proposed_tool_call")
                or {}
            )
            plan_run = continue_after_approve(
                run=plan_run,
                tool_executor=dependencies.tool_executor,
                approval_result={"approved": True},
                pending_tool_call=proposed_tool_call,
            )
            latest_observation = plan_run.observations[-1] if plan_run.observations else None
            return {
                "plan_run": plan_run,
                "step": _matching_step(plan_run, waiting_step.step_id),
                "answer": _answer_from_plan_run(plan_run),
                "status": plan_run.workflow_status,
                "route": SELECT_NEXT_STEP if plan_run.workflow_status == "running" else None,
                "tool_result": (
                    latest_observation.model_dump()
                    if latest_observation is not None
                    else None
                ),
            }

        if action == "respond":
            response = str(payload.get("response") or "").strip()
            plan_run = continue_after_respond(
                run=plan_run,
                response=response,
                source=str(payload.get("source") or "freeform"),
                suggestion_id=(
                    str(payload.get("suggestion_id"))
                    if payload.get("suggestion_id") is not None
                    else None
                ),
                metadata=dict(payload.get("metadata") or {}),
            )
            return {
                "plan_run": plan_run,
                "step": _matching_step(plan_run, waiting_step.step_id),
                "route": SELECT_NEXT_STEP,
            }

        raise ValueError("Unsupported Plan resume action.")

    return resume_waiting_step


def _coerce_resume_command(value: Any) -> Command:
    if isinstance(value, Command):
        return value
    raise ValueError("Command(resume=...) is required for Plan HITL resume.")


def _coerce_run(value: Any) -> PlanRun:
    if isinstance(value, PlanRun):
        return value
    if isinstance(value, Mapping):
        return PlanRun.model_validate(dict(value))
    raise ValueError("plan_run checkpoint is required for Plan HITL resume.")


def _waiting_step(plan_run: PlanRun) -> PlanStep:
    if plan_run.workflow_status != "waiting_user":
        raise ValueError(f"Plan run is not waiting_user: {plan_run.workflow_status}.")
    if not plan_run.current_step_id:
        raise ValueError("Plan run has no current waiting step.")
    for step in plan_run.steps:
        if step.step_id == plan_run.current_step_id:
            if step.status != "waiting_user":
                raise ValueError("Plan current step is not waiting_user.")
            return step
    raise ValueError("Plan current waiting step was not found.")


def _pending_tool_call(
    *,
    command_payload: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return dict(
        command_payload.get("proposed_tool_call")
        or payload.get("proposed_tool_call")
        or {}
    )


def _matching_step(plan_run: PlanRun, step_id: str) -> PlanStep | None:
    for step in plan_run.steps:
        if step.step_id == step_id:
            return step
    return None


def _answer_from_plan_run(plan_run: PlanRun) -> str:
    return str(
        plan_run.final_answer
        or plan_run.result_summary
        or plan_run.error
        or ""
    )

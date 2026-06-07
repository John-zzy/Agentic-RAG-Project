from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.platform.agent_runtime.contracts import PlanRun, PlanStep
from backend.platform.agent_runtime.plan.executor import PlanExecutor
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState


@dataclass(frozen=True)
class PlanHitlResumeGraphDependencies:
    """Dependencies for graph-owned Plan HITL continuation."""

    executor: PlanExecutor


def build_plan_hitl_resume_graph(
    dependencies: PlanHitlResumeGraphDependencies,
    *,
    checkpointer: Any | None = None,
) -> Any:
    builder = StateGraph(PlanGraphState)
    builder.add_node("resume_waiting_step", _build_resume_waiting_step_node(dependencies))
    builder.add_edge(START, "resume_waiting_step")
    builder.add_edge("resume_waiting_step", END)
    return builder.compile(checkpointer=checkpointer)


def _build_resume_waiting_step_node(dependencies: PlanHitlResumeGraphDependencies):
    def resume_waiting_step(state: PlanGraphState) -> dict[str, Any]:
        plan_run = _coerce_run(state.get("plan_run"))
        payload = dict(state.get("resume_payload") or {})
        action = str(payload.get("action") or "")
        waiting_step = _waiting_step(plan_run)

        if action == "reject":
            reason = str(payload.get("reason") or "User rejected the waiting Plan step.")
            plan_run = dependencies.executor.continue_after_reject(
                run=plan_run,
                reason=reason,
                pending_tool_call=_pending_tool_call(
                    state=state,
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
                state.get("proposed_tool_call")
                or payload.get("proposed_tool_call")
                or {}
            )
            plan_run = dependencies.executor.continue_after_approve(
                run=plan_run,
                approval_result={"approved": True},
                pending_tool_call=proposed_tool_call,
            )
            latest_observation = plan_run.observations[-1] if plan_run.observations else None
            return {
                "plan_run": plan_run,
                "step": _matching_step(plan_run, waiting_step.step_id),
                "answer": _answer_from_plan_run(plan_run),
                "status": plan_run.workflow_status,
                "tool_result": (
                    latest_observation.model_dump()
                    if latest_observation is not None
                    else None
                ),
            }

        if action == "respond":
            response = str(payload.get("response") or "").strip()
            plan_run = dependencies.executor.continue_after_respond(
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
            plan_run = dependencies.executor.run(plan_run)
            return {
                "plan_run": plan_run,
                "step": _matching_step(plan_run, waiting_step.step_id),
                "answer": _answer_from_plan_run(plan_run),
                "status": plan_run.workflow_status,
                "response_result": {
                    "status": plan_run.workflow_status,
                    "answer": _answer_from_plan_run(plan_run),
                    "plan_run": plan_run.model_dump(),
                },
            }

        raise ValueError("Unsupported Plan resume action.")

    return resume_waiting_step


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
    state: PlanGraphState,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return dict(
        state.get("proposed_tool_call")
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

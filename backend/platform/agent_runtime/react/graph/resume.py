from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.platform.agent_runtime.contracts import ReActRun, ToolObservation
from backend.platform.agent_runtime.react.graph.state import ReActGraphState
from backend.platform.workflow.state_machine import validate_transition


@dataclass(frozen=True)
class ReActHitlResumeGraphDependencies:
    """Dependencies for graph-owned ReAct HITL continuation."""

    approve_executor: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None
    respond_handler: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None] | None = None


def build_react_hitl_resume_graph(
    dependencies: ReActHitlResumeGraphDependencies,
    *,
    checkpointer: Any | None = None,
) -> Any:
    builder = StateGraph(ReActGraphState)
    builder.add_node("resume_waiting_turn", _build_resume_waiting_turn_node(dependencies))
    builder.add_edge(START, "resume_waiting_turn")
    builder.add_edge("resume_waiting_turn", END)
    return builder.compile(checkpointer=checkpointer)


def _build_resume_waiting_turn_node(dependencies: ReActHitlResumeGraphDependencies):
    def resume_waiting_turn(state: ReActGraphState) -> dict[str, Any]:
        run = _coerce_run(state.get("run"))
        payload = dict(state.get("resume_payload") or {})
        action = str(payload.get("action") or "")
        waiting_turn = _waiting_turn(run)

        if action == "reject":
            reason = str(payload.get("reason") or "User rejected the waiting ReAct turn.")
            _record_react_continuation(
                run=run,
                waiting_turn=waiting_turn,
                payload=payload,
                extra={"reason": reason, "side_effect_executed": False},
            )
            run.workflow_status = validate_transition(run.workflow_status, "resume_reject")
            waiting_turn.status = "cancelled"
            run.result_summary = reason
            run.error = None
            run.current_turn_id = None
            run.current_tool_call = None
            return {
                "run": run,
                "answer": "已拒绝该人工等待项，未执行待审批调用。",
                "status": "cancelled",
            }

        if action == "approve":
            proposed_tool_call = dict(
                state.get("proposed_tool_call")
                or payload.get("proposed_tool_call")
                or {}
            )
            if dependencies.approve_executor is None:
                raise ValueError("approve_executor is required for ReAct approve resume.")
            tool_result = dict(dependencies.approve_executor(proposed_tool_call) or {})
            observation = ToolObservation.model_validate(tool_result)
            _record_react_continuation(
                run=run,
                waiting_turn=waiting_turn,
                payload=payload,
                extra={
                    "approved": True,
                    "pending_tool_call": proposed_tool_call,
                    "side_effect_executed": True,
                },
            )
            run.workflow_status = validate_transition(run.workflow_status, "resume_approve")
            waiting_turn.status = "succeeded" if observation.success else "failed"
            waiting_turn.observation = observation
            waiting_turn.observation_summary = observation.result_summary
            waiting_turn.result_summary = waiting_turn.observation_summary
            waiting_turn.error = observation.error
            run.observations.append(observation)
            run.current_turn_id = None
            run.current_tool_call = None
            if waiting_turn.status == "failed":
                run.workflow_status = validate_transition(run.workflow_status, "fail")
                run.error = waiting_turn.error or waiting_turn.result_summary
                run.result_summary = run.error or "Approved ReAct tool failed."
                answer = run.result_summary
            else:
                run.workflow_status = validate_transition(run.workflow_status, "success")
                run.error = None
                run.result_summary = waiting_turn.result_summary or "Approved ReAct tool executed."
                run.final_answer = run.result_summary
                answer = "已批准并执行待审批操作。"
            return {
                "run": run,
                "answer": answer,
                "status": run.workflow_status,
                "tool_result": tool_result,
            }

        if action == "respond":
            if dependencies.respond_handler is None:
                raise ValueError("respond_handler is required for ReAct respond resume.")
            _record_react_continuation(
                run=run,
                waiting_turn=waiting_turn,
                payload=payload,
                extra={
                    "response": payload.get("response"),
                    "source": payload.get("source"),
                    "suggestion_id": payload.get("suggestion_id"),
                },
            )
            run.workflow_status = validate_transition(run.workflow_status, "resume_respond")
            waiting_turn.status = "succeeded"
            run.current_turn_id = None
            run.current_tool_call = None
            response_result = dict(
                dependencies.respond_handler(
                    payload,
                    dict(state.get("accepted_state") or {}),
                )
                or {}
            )
            if isinstance(response_result.get("react_run"), Mapping):
                run = ReActRun.model_validate(dict(response_result["react_run"]))
            elif str(response_result.get("status") or "") == "succeeded":
                run.workflow_status = validate_transition(run.workflow_status, "success")
                run.final_answer = str(response_result.get("answer") or run.result_summary or "")
                run.result_summary = run.final_answer
                run.error = None
            return {
                "run": run,
                "answer": str(response_result.get("answer") or run.result_summary or ""),
                "status": str(response_result.get("status") or run.workflow_status),
                "response_result": response_result,
            }

        raise ValueError("Unsupported ReAct resume action.")

    return resume_waiting_turn


def _coerce_run(value: Any) -> ReActRun:
    if isinstance(value, ReActRun):
        return value
    if isinstance(value, Mapping):
        return ReActRun.model_validate(dict(value))
    raise ValueError("react_run checkpoint is required for ReAct HITL resume.")


def _waiting_turn(run: ReActRun):
    if run.workflow_status != "waiting_user":
        raise ValueError(f"ReAct run is not waiting_user: {run.workflow_status}.")
    if not run.current_turn_id:
        raise ValueError("ReAct run has no current waiting turn.")
    for turn in run.turns:
        if turn.turn_id == run.current_turn_id:
            if turn.status != "waiting_user":
                raise ValueError("ReAct current turn is not waiting_user.")
            return turn
    raise ValueError("ReAct current waiting turn was not found.")


def _record_react_continuation(
    *,
    run: ReActRun,
    waiting_turn: Any,
    payload: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> None:
    continuation = {
        "mode": "react",
        "action": payload.get("action"),
        "react_run_id": run.react_run_id,
        "waiting_turn_id": waiting_turn.turn_id,
        "continued_from_turn_id": waiting_turn.turn_id,
        "metadata": dict(payload.get("metadata") or {}),
        **dict(extra),
    }
    history = list(run.metadata.get("continuations") or [])
    history.append(continuation)
    run.metadata["resume"] = continuation
    run.metadata["continuations"] = history
    turn_metadata = dict(waiting_turn.metadata or {})
    turn_metadata["continuation"] = continuation
    waiting_turn.metadata = turn_metadata

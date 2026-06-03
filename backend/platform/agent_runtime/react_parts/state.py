from __future__ import annotations

from collections.abc import Mapping

from backend.platform.agent_runtime.contracts import (
    ReActRun,
    ReActTurn,
    RetryMetadata,
    ToolObservation,
)
from backend.platform.workflow.state_machine import is_terminal, validate_transition


def transition(run: ReActRun, event: str) -> None:
    previous_status = run.workflow_status
    next_status = validate_transition(previous_status, event)
    run.workflow_status = next_status
    transitions = list(run.metadata.get("workflow_transitions") or [])
    transitions.append(
        {
            "from": previous_status,
            "event": event,
            "to": next_status,
        }
    )
    run.metadata["workflow_transitions"] = transitions


def ensure_running(run: ReActRun) -> None:
    if run.workflow_status == "running":
        return
    if run.workflow_status == "retrying":
        transition(run, "retry")
        return
    if run.workflow_status == "created":
        transition(run, "run_start")
        return
    raise ValueError(f"ReAct run cannot execute from state: {run.workflow_status}.")


def ensure_react_run_can_continue(run: ReActRun) -> None:
    if is_terminal(run.workflow_status):
        raise ValueError(
            f"ReAct run is already terminal and cannot continue: {run.workflow_status}."
        )
    if run.workflow_status == "waiting_user":
        raise ValueError("ReAct run is waiting for user input and cannot continue directly.")


def persist_turn_observation(
    *,
    run: ReActRun,
    turn: ReActTurn,
    observation: ToolObservation,
) -> None:
    turn.observation = observation
    turn.observation_summary = observation.result_summary
    turn.result_summary = observation.result_summary
    turn.error = observation.error
    run.observations.append(observation)
    run.current_tool_call = observation.execution


def attach_observation_hitl_metadata(
    *,
    observation: ToolObservation,
    hitl_metadata: dict,
) -> ToolObservation:
    metadata = dict(observation.metadata)
    metadata["hitl"] = dict(hitl_metadata)
    return observation.model_copy(update={"metadata": metadata})


def prepare_observation_for_turn(
    *,
    run: ReActRun,
    turn: ReActTurn,
    observation: ToolObservation,
) -> ToolObservation:
    if not observation.requires_user:
        return observation
    return attach_observation_hitl_metadata(
        observation=observation,
        hitl_metadata=build_react_hitl_metadata(
            run=run,
            turn=turn,
            user_prompt=observation.user_prompt or observation.result_summary,
            source="tool_observation",
        ),
    )


def mark_waiting_on_observation(
    *,
    run: ReActRun,
    turn: ReActTurn,
    observation: ToolObservation,
) -> None:
    hitl_metadata = dict(observation.metadata.get("hitl") or {})
    turn.status = "waiting_user"
    turn.metadata["hitl"] = hitl_metadata
    run.current_turn_id = turn.turn_id
    run.metadata["hitl"] = hitl_metadata
    transition(run, "interrupt")


def build_react_hitl_metadata(
    *,
    run: ReActRun,
    turn: ReActTurn,
    user_prompt: str,
    source: str,
) -> dict:
    return {
        "mode": "react",
        "react_run_id": run.react_run_id,
        "current_turn_id": turn.turn_id,
        "user_prompt": user_prompt,
        "source": source,
    }


def next_retry_metadata(
    *,
    turn: ReActTurn,
    observation: ToolObservation,
) -> RetryMetadata:
    return turn.retry_metadata.model_copy(
        update={
            "attempt": turn.retry_metadata.attempt + 1,
            "retryable": observation.retryable,
            "last_error": observation.error,
        }
    )


def record_retry_metadata(
    *,
    run: ReActRun,
    turn: ReActTurn,
    observation: ToolObservation,
) -> None:
    """记录 run 级 retry 恢复点，避免只在 turn 内部留下局部状态。"""
    retry = {
        "attempt": turn.retry_metadata.attempt,
        "max_attempts": turn.retry_metadata.max_attempts,
        "latest_error": observation_error(observation),
        "current_turn_id": turn.turn_id,
        "tool_name": turn.tool_name,
        "tool_call_id": observation.tool_call_id,
        "retryable": observation.retryable,
    }
    run.metadata["retry"] = retry
    history = list(run.metadata.get("retry_history") or [])
    history.append(retry)
    run.metadata["retry_history"] = history


def attempted_tools(run: ReActRun) -> list[str]:
    attempted: list[str] = []
    seen: set[str] = set()
    for turn in run.turns:
        if not turn.tool_name or turn.tool_name in seen:
            continue
        seen.add(turn.tool_name)
        attempted.append(turn.tool_name)
    return attempted


def resume_metadata(run: ReActRun) -> dict:
    value = run.metadata.get("resume")
    return dict(value) if isinstance(value, Mapping) else {}


def latest_final_decision(run: ReActRun) -> str | None:
    for observation in reversed(run.observations):
        if decision := observation_final_decision(observation):
            return decision
    return None


def observation_final_decision(observation: ToolObservation) -> str | None:
    sources = (
        observation.metadata,
        observation.output if isinstance(observation.output, Mapping) else {},
        observation.trace.get("retrieval_trace")
        if isinstance(observation.trace.get("retrieval_trace"), Mapping)
        else {},
    )
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        value = source.get("final_decision")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def observation_error(observation: ToolObservation) -> str:
    return observation.error or observation.result_summary

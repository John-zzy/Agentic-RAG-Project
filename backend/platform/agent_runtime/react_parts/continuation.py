from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from backend.platform.agent_runtime.contracts import (
    AgentRuntimeModel,
    ReActRun,
    ReActTurn,
)
from backend.platform.agent_runtime.react_parts.state import transition


ReActContinuationAction = Literal["respond", "approve", "reject"]


class ReActContinuationInput(AgentRuntimeModel):
    """ReAct waiting_user 恢复输入，只保存可审计的人类决策。"""

    action: ReActContinuationAction
    response: str | None = None
    source: str | None = None
    suggestion_id: str | None = None
    approval_result: dict[str, Any] = Field(default_factory=dict)
    pending_tool_call: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReActContinuationManager:
    """把 waiting_user 恢复为同一个 ReActRun 的后续状态。"""

    def apply(self, *, run: ReActRun, continuation: ReActContinuationInput) -> ReActRun:
        waiting_turn = _resolve_waiting_turn(run)
        if continuation.action == "respond":
            self._apply_respond(run=run, turn=waiting_turn, continuation=continuation)
            return run
        if continuation.action == "approve":
            self._apply_approve(run=run, turn=waiting_turn, continuation=continuation)
            return run
        self._apply_reject(run=run, turn=waiting_turn, continuation=continuation)
        return run

    def _apply_respond(
        self,
        *,
        run: ReActRun,
        turn: ReActTurn,
        continuation: ReActContinuationInput,
    ) -> None:
        response = str(continuation.response or "").strip()
        if not response:
            raise ValueError("respond continuation requires a non-empty response.")
        metadata = _build_continuation_metadata(
            run=run,
            turn=turn,
            continuation=continuation.model_copy(update={"response": response}),
            extra={
                "response": response,
                "source": continuation.source or "freeform",
                "suggestion_id": continuation.suggestion_id,
            },
        )
        _record_running_continuation(
            run=run,
            turn=turn,
            event="resume_respond",
            metadata=metadata,
        )

    def _apply_approve(
        self,
        *,
        run: ReActRun,
        turn: ReActTurn,
        continuation: ReActContinuationInput,
    ) -> None:
        pending_tool_call = _pending_tool_call_payload(
            run=run,
            provided=continuation.pending_tool_call,
        )
        metadata = _build_continuation_metadata(
            run=run,
            turn=turn,
            continuation=continuation,
            extra={
                "approved": True,
                "approval_result": dict(continuation.approval_result),
                "pending_tool_call": pending_tool_call,
            },
        )
        _record_running_continuation(
            run=run,
            turn=turn,
            event="resume_approve",
            metadata=metadata,
        )

    def _apply_reject(
        self,
        *,
        run: ReActRun,
        turn: ReActTurn,
        continuation: ReActContinuationInput,
    ) -> None:
        reason = _non_empty_text(
            continuation.reason,
            default="User rejected the waiting ReAct turn.",
        )
        metadata = _build_continuation_metadata(
            run=run,
            turn=turn,
            continuation=continuation,
            extra={
                "reason": reason,
                "pending_tool_call": _pending_tool_call_payload(
                    run=run,
                    provided=continuation.pending_tool_call,
                ),
                "side_effect_executed": False,
            },
        )
        _append_continuation_history(run=run, turn=turn, metadata=metadata)
        transition(run, "resume_reject")
        turn.status = "cancelled"
        turn.metadata["continuation"] = metadata
        run.metadata["resume"] = metadata
        run.result_summary = reason
        run.error = None
        run.current_turn_id = None
        run.current_tool_call = None


def _resolve_waiting_turn(run: ReActRun) -> ReActTurn:
    if run.workflow_status != "waiting_user":
        raise ValueError(f"ReAct continuation requires waiting_user state, got {run.workflow_status}.")
    waiting_turn_id = str(
        run.current_turn_id
        or (run.metadata.get("hitl") or {}).get("current_turn_id")
        or ""
    )
    if not waiting_turn_id:
        raise ValueError("ReAct continuation requires a current waiting turn.")
    for turn in run.turns:
        if turn.turn_id == waiting_turn_id:
            if turn.status != "waiting_user":
                raise ValueError("ReAct continuation turn is not waiting_user.")
            return turn
    raise ValueError("ReAct continuation waiting turn was not found.")


def _record_running_continuation(
    *,
    run: ReActRun,
    turn: ReActTurn,
    event: Literal["resume_respond", "resume_approve"],
    metadata: dict[str, Any],
) -> None:
    _ensure_continuation_turn_budget(run=run, metadata=metadata)
    _append_continuation_history(run=run, turn=turn, metadata=metadata)
    transition(run, event)
    turn.status = "succeeded"
    turn.metadata["continuation"] = metadata
    run.metadata["resume"] = metadata
    run.error = None
    run.current_turn_id = None
    run.current_tool_call = None


def _ensure_continuation_turn_budget(
    *,
    run: ReActRun,
    metadata: dict[str, Any],
) -> None:
    if len(run.turns) < run.max_turns:
        return
    previous_max_turns = run.max_turns
    # 用户已经补充或批准后，必须至少给 LLM 一次机会消费 resume context。
    run.max_turns = len(run.turns) + 1
    extension = {
        "reason": "waiting_user_continuation",
        "previous_max_turns": previous_max_turns,
        "extended_max_turns": run.max_turns,
    }
    metadata["budget_extension"] = extension
    run.metadata["continuation_budget_extension"] = extension


def _build_continuation_metadata(
    *,
    run: ReActRun,
    turn: ReActTurn,
    continuation: ReActContinuationInput,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "mode": "react",
        "action": continuation.action,
        "react_run_id": run.react_run_id,
        "waiting_turn_id": turn.turn_id,
        "continued_from_turn_id": turn.turn_id,
        "metadata": dict(continuation.metadata),
        **dict(extra),
    }


def _append_continuation_history(
    *,
    run: ReActRun,
    turn: ReActTurn,
    metadata: Mapping[str, Any],
) -> None:
    history = list(run.metadata.get("continuations") or [])
    history.append(dict(metadata))
    run.metadata["continuations"] = history
    turn.metadata["continuations"] = list(turn.metadata.get("continuations") or []) + [
        dict(metadata)
    ]


def _pending_tool_call_payload(
    *,
    run: ReActRun,
    provided: Mapping[str, Any],
) -> dict[str, Any]:
    if provided:
        return dict(provided)
    if run.current_tool_call is None:
        return {}
    return run.current_tool_call.model_dump()


def _non_empty_text(value: str | None, *, default: str) -> str:
    text = str(value or "").strip()
    return text or default

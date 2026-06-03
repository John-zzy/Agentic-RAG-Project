from __future__ import annotations

from backend.platform.agent_runtime.contracts import ReActRun, ReActTurn, ToolObservation
from backend.platform.agent_runtime.react_parts.state import (
    ensure_running,
    mark_waiting_on_observation,
    next_retry_metadata,
    observation_error,
    persist_turn_observation,
    prepare_observation_for_turn,
    record_retry_metadata,
    transition,
)
from backend.platform.agent_runtime.tool_executor import ToolExecutor


class ReActToolTurnExecutor:
    """执行单个 tool_call turn，并负责 observation 先落盘再推进状态。"""

    def __init__(self, *, tool_executor: ToolExecutor) -> None:
        self._tool_executor = tool_executor

    def execute(self, *, run: ReActRun, turn: ReActTurn) -> str | None:
        while True:
            turn.status = "running"
            ensure_running(run)
            observation = self._execute_tool_action(turn)
            observation = prepare_observation_for_turn(run=run, turn=turn, observation=observation)
            persist_turn_observation(run=run, turn=turn, observation=observation)
            turn.retry_metadata = next_retry_metadata(turn=turn, observation=observation)
            if not observation.success and observation.retryable:
                record_retry_metadata(run=run, turn=turn, observation=observation)

            if observation.requires_user:
                mark_waiting_on_observation(run=run, turn=turn, observation=observation)
                return None
            if observation.success:
                turn.status = "succeeded"
                turn.error = None
                run.error = None
                run.current_tool_call = None
                return None
            if not observation.retryable:
                turn.status = "failed"
                return observation_error(observation)

            turn.status = "retrying"
            transition(run, "tool_error_retryable")
            run.error = observation_error(observation)
            if turn.retry_metadata.attempt >= turn.retry_metadata.max_attempts:
                turn.status = "failed"
                return observation_error(observation)
            transition(run, "retry")

    def _execute_tool_action(self, turn: ReActTurn) -> ToolObservation:
        return self._tool_executor.execute(
            tool_name=turn.action.tool_name or "",
            input_payload=turn.action.input,
            attempt=turn.retry_metadata.attempt,
            max_attempts=turn.retry_metadata.max_attempts,
        )

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from backend.platform.agent_runtime.core.contracts import (
    ToolExecutionMetadata,
    ToolObservation,
)
from backend.platform.rag.contracts import RetrievalResult
from backend.platform.tools.base import ToolResult


ObservationStatus = Literal["succeeded", "failed", "retryable", "waiting_user", "cancelled"]


class ToolObservationMiddleware:
    """Normalize LangChain-style tool outputs and errors into ToolObservation."""

    def normalize(
        self,
        *,
        tool_name: str,
        result: Any = None,
        error: BaseException | None = None,
        execution: ToolExecutionMetadata | None = None,
        retryable: bool | None = None,
    ) -> ToolObservation:
        execution = execution or ToolExecutionMetadata(tool_name=tool_name)
        if error is not None:
            resolved_retryable = retryable if retryable is not None else _is_retryable(error)
            return ToolObservation(
                tool_name=tool_name,
                success=False,
                result_summary=f"{tool_name} failed: {error}",
                retryable=resolved_retryable,
                error=str(error),
                execution=execution.model_copy(update={"retryable": resolved_retryable}),
            )
        return normalize_tool_result(
            tool_name=tool_name,
            result=result,
            execution=execution,
        )


def normalize_tool_result(
    *,
    tool_name: str,
    result: Any,
    execution: ToolExecutionMetadata,
) -> ToolObservation:
    if isinstance(result, ToolObservation):
        return result.model_copy(
            update={
                "execution": result.execution or execution,
                "tool_call_id": result.tool_call_id or execution.tool_call_id,
            }
        )
    if isinstance(result, ToolResult):
        return _tool_result_to_observation(
            tool_name=tool_name,
            result=result,
            execution=execution,
        )
    if isinstance(result, RetrievalResult):
        from backend.platform.agent_runtime.tooling.rag import retrieval_result_to_observation

        return retrieval_result_to_observation(
            adapter_name=tool_name,
            result=result,
            execution=execution,
        ).model_copy(update={"tool_call_id": execution.tool_call_id})
    if isinstance(result, Mapping):
        try:
            return ToolObservation.model_validate(dict(result))
        except Exception:
            pass
    return ToolObservation(
        tool_name=tool_name,
        success=True,
        output=result,
        result_summary=f"{tool_name} succeeded.",
        execution=execution,
        tool_call_id=execution.tool_call_id,
    )


def observation_status(observation: ToolObservation) -> ObservationStatus:
    if observation.metadata.get("cancelled"):
        return "cancelled"
    if observation.requires_user:
        return "waiting_user"
    if observation.success:
        return "succeeded"
    if observation.retryable:
        return "retryable"
    return "failed"


def _tool_result_to_observation(
    *,
    tool_name: str,
    result: ToolResult,
    execution: ToolExecutionMetadata,
) -> ToolObservation:
    requires_user = bool(result.metadata.get("requires_user"))
    retryable = bool(result.metadata.get("retryable", not result.success))
    return ToolObservation(
        tool_name=tool_name,
        success=result.success,
        output={
            "records": list(result.records),
            "confidence": result.confidence,
            "metadata": dict(result.metadata),
        },
        result_summary=_tool_result_summary(tool_name=tool_name, result=result),
        citations=list(result.citations) if result.success else [],
        trace=dict(result.metadata.get("trace") or {}),
        retryable=retryable,
        requires_user=requires_user,
        user_prompt=result.metadata.get("user_prompt"),
        error=result.error,
        execution=execution.model_copy(update={"retryable": retryable}),
        tool_call_id=execution.tool_call_id,
        metadata=dict(result.metadata),
    )


def _tool_result_summary(*, tool_name: str, result: ToolResult) -> str:
    if result.success:
        return f"{tool_name} succeeded with {len(result.records)} record(s)."
    return f"{tool_name} failed: {result.error or 'unknown error'}."


def _is_retryable(error: BaseException) -> bool:
    return isinstance(error, (TimeoutError, ConnectionError))

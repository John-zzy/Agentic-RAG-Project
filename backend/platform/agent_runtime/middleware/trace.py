from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from backend.platform.agent_runtime.core.contracts import AgentRuntimeModel, ToolObservation
from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext


TraceEventType = Literal["model_call", "tool_call", "hitl_wait", "runtime_event"]

_BLOCKED_KEY_PARTS = (
    "api_key",
    "arguments",
    "authorization",
    "checkpoint",
    "chain_of_thought",
    "full_history",
    "history",
    "input_payload",
    "messages",
    "password",
    "prompt",
    "raw",
    "secret",
    "tool_args",
)


class RuntimeTraceEvent(AgentRuntimeModel):
    event_type: TraceEventType
    session_id: str
    request_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeTraceMiddleware:
    """Record safe model/tool metadata without leaking prompts, tool args or secrets."""

    def __init__(self) -> None:
        self._events: list[RuntimeTraceEvent] = []

    @property
    def events(self) -> tuple[RuntimeTraceEvent, ...]:
        return tuple(self._events)

    def record_model_call(
        self,
        *,
        context: AgentRuntimeContext,
        latency_ms: float,
        retry_count: int = 0,
        provider: str | None = None,
        complexity: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeTraceEvent:
        return self.record(
            event_type="model_call",
            context=context,
            metadata={
                "latency_ms": latency_ms,
                "retry_count": retry_count,
                "provider": provider or context.provider_name,
                "complexity": complexity or context.complexity,
                **dict(metadata or {}),
            },
        )

    def record_tool_call(
        self,
        *,
        context: AgentRuntimeContext,
        observation: ToolObservation,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeTraceEvent:
        status = "succeeded" if observation.success else "failed"
        if observation.requires_user:
            status = "waiting_user"
        return self.record(
            event_type="tool_call",
            context=context,
            metadata={
                "tool_name": observation.tool_name,
                "tool_status": status,
                "retryable": observation.retryable,
                "error_classification": _summarize_error(observation.error),
                **dict(metadata or {}),
            },
        )

    def record(
        self,
        *,
        event_type: TraceEventType,
        context: AgentRuntimeContext,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeTraceEvent:
        event = RuntimeTraceEvent(
            event_type=event_type,
            session_id=context.session_id,
            request_id=context.request_id,
            metadata=sanitize_for_trace(
                {
                    **context.to_safe_metadata(),
                    **dict(metadata or {}),
                }
            ),
        )
        self._events.append(event)
        return event


def sanitize_for_trace(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if any(part in key_lower for part in _BLOCKED_KEY_PARTS):
                continue
            sanitized[key_text] = sanitize_for_trace(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_trace(item) for item in value]
    return value


def _summarize_error(error: str | None) -> str | None:
    if not error:
        return None
    lowered = error.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "not allowed" in lowered or "invalid input" in lowered:
        return "validation"
    return "provider_or_tool_error"

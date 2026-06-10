from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext
from backend.platform.agent_runtime.middleware.model_guard import (
    GuardedModelResult,
    ModelGuardMiddleware,
)
from backend.platform.agent_runtime.middleware.trace import RuntimeTraceMiddleware

T = TypeVar("T")


class GuardedModelCallError(RuntimeError):
    """Raised when a shared guarded model call cannot produce a typed output."""

    def __init__(self, result: GuardedModelResult) -> None:
        self.result = result
        super().__init__(result.error or "Model invocation failed.")


class SharedModelCallGuard:
    """Small helper for non-agent graph nodes that still need middleware model guard."""

    def __init__(
        self,
        *,
        guard: ModelGuardMiddleware | None = None,
        trace: RuntimeTraceMiddleware | None = None,
    ) -> None:
        self._guard = guard or ModelGuardMiddleware()
        self._trace = trace

    def invoke(
        self,
        operation: Callable[[], T],
        *,
        context: AgentRuntimeContext,
        metadata: Mapping[str, Any] | None = None,
        token_metadata: Mapping[str, Any] | None = None,
        output_type: type[T] | None = None,
    ) -> T:
        result = self.invoke_raw(
            operation,
            context=context,
            metadata=metadata,
            token_metadata=token_metadata,
        )
        if not result.success:
            raise GuardedModelCallError(result)
        output = result.output
        if output_type is not None and not isinstance(output, output_type):
            raise TypeError(
                f"Guarded model call returned {type(output).__name__}; "
                f"expected {output_type.__name__}."
            )
        return output

    def invoke_raw(
        self,
        operation: Callable[[], Any],
        *,
        context: AgentRuntimeContext,
        metadata: Mapping[str, Any] | None = None,
        token_metadata: Mapping[str, Any] | None = None,
    ) -> GuardedModelResult:
        result = self._guard.invoke(
            operation,
            context=context,
            metadata=metadata,
            token_metadata=token_metadata,
        )
        if self._trace is not None:
            self._trace.record_model_call(
                context=context,
                latency_ms=result.metadata.latency_ms,
                retry_count=result.metadata.retry_count,
                provider=result.metadata.provider,
                complexity=result.metadata.complexity,
                metadata={
                    "fallback_used": result.metadata.fallback_used,
                    "error_classification": result.metadata.error_classification,
                    **dict(metadata or {}),
                },
            )
        return result


def default_model_call_context(
    *,
    session_id: str,
    request_id: str,
    scene: str = "platform",
    mounted_knowledge_sources: tuple[str, ...] = ("documents",),
    complexity: str = "simple",
    provider_name: str | None = None,
    workflow_metadata: Mapping[str, Any] | None = None,
    request_metadata: Mapping[str, Any] | None = None,
) -> AgentRuntimeContext:
    """构建 platform 子图可复用的最小模型调用上下文。"""
    return AgentRuntimeContext.build(
        session_id=session_id,
        request_id=request_id,
        scene=scene,
        mounted_knowledge_sources=mounted_knowledge_sources,
        complexity=complexity,
        provider_name=provider_name,
        workflow=workflow_metadata,
        request_metadata=request_metadata,
    )

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import Field

from backend.platform.agent_runtime.core.contracts import AgentRuntimeModel
from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext
from backend.platform.agent_runtime.middleware.trace import sanitize_for_trace
from backend.platform.models.llm.guards.classifier import ModelFailureClassifier


RetryPredicate = type[BaseException] | tuple[type[BaseException], ...] | Callable[[BaseException], bool]
FallbackFactory = Callable[[BaseException], Any]


class ModelCallMetadata(AgentRuntimeModel):
    provider: str | None = None
    complexity: str
    latency_ms: float
    retry_count: int = Field(default=0, ge=0)
    fallback_used: bool = False
    token_usage: dict[str, Any] = Field(default_factory=dict)
    error_classification: dict[str, Any] | None = None


class GuardedModelResult(AgentRuntimeModel):
    success: bool
    output: Any = None
    error: str | None = None
    metadata: ModelCallMetadata


@dataclass(frozen=True, slots=True)
class ModelGuardPolicy:
    max_attempts: int = 2
    retry_on: RetryPredicate = (TimeoutError, ConnectionError)
    fallback: Any | FallbackFactory | None = None


class ModelGuardMiddleware:
    """Central model call guard for retry, fallback, empty-output and observability."""

    def __init__(
        self,
        *,
        policy: ModelGuardPolicy | None = None,
        classifier: ModelFailureClassifier | None = None,
    ) -> None:
        self._policy = policy or ModelGuardPolicy()
        self._classifier = classifier or ModelFailureClassifier()

    def invoke(
        self,
        operation: Callable[[], Any],
        *,
        context: AgentRuntimeContext,
        metadata: Mapping[str, Any] | None = None,
        token_metadata: Mapping[str, Any] | None = None,
    ) -> GuardedModelResult:
        started = perf_counter()
        failure: BaseException | None = None
        attempts = max(1, self._policy.max_attempts)
        for attempt in range(1, attempts + 1):
            try:
                output = _ensure_non_empty(operation())
                return GuardedModelResult(
                    success=True,
                    output=output,
                    metadata=self._metadata(
                        context=context,
                        started=started,
                        retry_count=attempt - 1,
                        token_metadata=token_metadata,
                    ),
                )
            except Exception as exc:
                failure = exc
                if attempt < attempts and _should_retry(exc, self._policy.retry_on):
                    continue
                classification = self._classify(
                    exc,
                    context=context,
                    attempt=attempt,
                    max_attempts=attempts,
                    metadata=metadata,
                )
                fallback = self._resolve_fallback(exc)
                return GuardedModelResult(
                    success=fallback is not None,
                    output=fallback,
                    error=str(exc),
                    metadata=self._metadata(
                        context=context,
                        started=started,
                        retry_count=attempt - 1,
                        fallback_used=fallback is not None,
                        token_metadata=token_metadata,
                        error_classification=classification,
                    ),
                )

        exc = failure or RuntimeError("Model invocation failed.")
        classification = self._classify(
            exc,
            context=context,
            attempt=attempts,
            max_attempts=attempts,
            metadata=metadata,
        )
        return GuardedModelResult(
            success=False,
            error=str(exc),
            metadata=self._metadata(
                context=context,
                started=started,
                retry_count=max(0, attempts - 1),
                token_metadata=token_metadata,
                error_classification=classification,
            ),
        )

    def _metadata(
        self,
        *,
        context: AgentRuntimeContext,
        started: float,
        retry_count: int,
        fallback_used: bool = False,
        token_metadata: Mapping[str, Any] | None = None,
        error_classification: Mapping[str, Any] | None = None,
    ) -> ModelCallMetadata:
        return ModelCallMetadata(
            provider=context.provider_name,
            complexity=context.complexity,
            latency_ms=round((perf_counter() - started) * 1000, 3),
            retry_count=retry_count,
            fallback_used=fallback_used,
            token_usage=dict(token_metadata or {}),
            error_classification=dict(error_classification) if error_classification else None,
        )

    def _classify(
        self,
        exc: BaseException,
        *,
        context: AgentRuntimeContext,
        attempt: int,
        max_attempts: int,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        record = self._classifier.build_record(
            exc,
            call_method="invoke",
            complexity=context.complexity,
            attempt=attempt,
            max_attempts=max_attempts,
            metadata=sanitize_for_trace(metadata or {}),
        )
        return dict(record.to_payload())

    def _resolve_fallback(self, exc: BaseException) -> Any:
        fallback = self._policy.fallback
        if fallback is None:
            return None
        if callable(fallback):
            return fallback(exc)
        return fallback


def _ensure_non_empty(output: Any) -> Any:
    if output is None:
        raise ValueError("Model returned empty content")
    if isinstance(output, str):
        stripped = output.strip()
        if not stripped:
            raise ValueError("Model returned empty content")
        return stripped
    if isinstance(output, (list, tuple, dict, set)) and not output:
        raise ValueError("Model returned empty content")
    return output


def _should_retry(exc: BaseException, retry_on: RetryPredicate) -> bool:
    if isinstance(retry_on, type):
        return isinstance(exc, retry_on)
    if callable(retry_on):
        return bool(retry_on(exc))
    return isinstance(exc, retry_on)

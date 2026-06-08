from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from backend.platform.models.llm.guards.classifier import ModelFailureClassifier
from backend.platform.models.llm.guards.config import ModelGuardConfig
from backend.platform.models.llm.guards.errors import ModelGuardFailureError


class ModelGuardAdapter:
    """保护 LangChain runnable 调用，统一空输出、重试和失败分类。"""

    def __init__(
        self,
        *,
        classifier: ModelFailureClassifier | None = None,
    ) -> None:
        self._classifier = classifier or ModelFailureClassifier()

    def invoke(
        self,
        operation: Callable[[], Any],
        *,
        config: ModelGuardConfig,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        failure: BaseException | None = None
        for attempt in range(1, config.retry.max_attempts + 1):
            try:
                content = operation()
                return self._ensure_non_empty_content(content)
            except Exception as exc:
                failure = exc
                if self._should_retry(exc, config=config, attempt=attempt):
                    continue
                raise self._guard_error(
                    exc,
                    call_method="invoke",
                    config=config,
                    attempt=attempt,
                    metadata=metadata,
                ) from exc
        raise self._guard_error(
            failure or RuntimeError("Model invocation failed."),
            call_method="invoke",
            config=config,
            attempt=config.retry.max_attempts,
            metadata=metadata,
        )

    def stream(
        self,
        operation: Callable[[], Iterator[Any]],
        *,
        config: ModelGuardConfig,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[Any]:
        failure: BaseException | None = None
        for attempt in range(1, config.retry.max_attempts + 1):
            try:
                yielded = False
                for chunk in operation():
                    if not chunk:
                        continue
                    yielded = True
                    yield chunk
                if yielded:
                    return
                raise ValueError("Model returned empty streaming content")
            except Exception as exc:
                failure = exc
                if self._should_retry(exc, config=config, attempt=attempt):
                    continue
                raise self._guard_error(
                    exc,
                    call_method="stream",
                    config=config,
                    attempt=attempt,
                    metadata=metadata,
                ) from exc
        raise self._guard_error(
            failure or RuntimeError("Model streaming failed."),
            call_method="stream",
            config=config,
            attempt=config.retry.max_attempts,
            metadata=metadata,
        )

    def _ensure_non_empty_content(self, content: Any) -> Any:
        if not content:
            raise ValueError("Model returned empty content")
        if isinstance(content, str):
            return content.strip()
        return content

    def _should_retry(
        self,
        exc: BaseException,
        *,
        config: ModelGuardConfig,
        attempt: int,
    ) -> bool:
        if attempt >= config.retry.max_attempts:
            return False
        predicate = config.retry.retry_on
        if isinstance(predicate, type):
            return isinstance(exc, predicate)
        if callable(predicate):
            return bool(predicate(exc))
        return isinstance(exc, tuple(predicate))

    def _guard_error(
        self,
        exc: BaseException,
        *,
        call_method: str,
        config: ModelGuardConfig,
        attempt: int,
        metadata: Mapping[str, Any] | None,
    ) -> ModelGuardFailureError:
        record = self._classifier.build_record(
            exc,
            call_method=call_method,
            complexity=config.complexity,
            attempt=attempt,
            max_attempts=config.retry.max_attempts,
            metadata={
                "timeout_seconds": config.timeout_seconds,
                **dict(metadata or {}),
            },
        )
        return ModelGuardFailureError(record.message, failure_payload=record.to_payload())

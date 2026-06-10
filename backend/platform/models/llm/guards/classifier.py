from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ModelFailureClassifier:
    """把模型调用异常归类为稳定 FailureRecord。"""

    def build_record(
        self,
        exc: BaseException,
        *,
        call_method: str,
        complexity: str,
        attempt: int | None,
        max_attempts: int | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        from backend.platform.agent_runtime.quality.failures import build_failure_record

        failure_metadata = {
            "call_method": call_method,
            "complexity": complexity,
            **dict(metadata or {}),
        }
        category = self._explicit_category(exc)
        if category is not None:
            failure_metadata["category"] = category.value
        return build_failure_record(
            exc,
            source="model",
            attempt=attempt,
            max_attempts=max_attempts,
            metadata=failure_metadata,
        )

    def _explicit_category(self, exc: BaseException) -> Any:
        from backend.platform.agent_runtime.quality.failures import FailureCategory

        if _is_rate_limit_error(exc):
            return FailureCategory.MODEL_RATE_LIMIT
        if _is_empty_output_error(exc):
            return FailureCategory.MODEL_EMPTY_OUTPUT
        return None


def _is_rate_limit_error(exc: BaseException) -> bool:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return "ratelimit" in name or "rate_limit" in name or "rate limit" in message


def _is_empty_output_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and str(exc) in {
        "Model returned empty content",
        "Model returned empty streaming content",
    }

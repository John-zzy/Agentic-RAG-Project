from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


RetryPredicate = type[BaseException] | Sequence[type[BaseException]] | Callable[[BaseException], bool]


@dataclass(frozen=True, slots=True)
class ModelRetryConfig:
    """模型调用框架异常重试配置，不承载业务 fallback 策略。"""

    max_attempts: int = 2
    retry_on: RetryPredicate = (TimeoutError, ConnectionError)


@dataclass(frozen=True, slots=True)
class ModelGuardConfig:
    """模型 guard 的统一配置入口。"""

    complexity: str = "unknown"
    timeout_seconds: float | None = None
    retry: ModelRetryConfig = ModelRetryConfig()

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta

from langgraph.types import RetryPolicy, TimeoutPolicy


RetryPredicate = type[Exception] | Sequence[type[Exception]] | Callable[[Exception], bool]


@dataclass(frozen=True, slots=True)
class RetryPolicyConfig:
    """节点级框架异常重试配置，不承载业务 observation retry。"""

    max_attempts: int = 2
    initial_interval: float = 0.25
    backoff_factor: float = 2.0
    max_interval: float = 5.0
    jitter: bool = True
    retry_on: RetryPredicate = (TimeoutError, ConnectionError)


@dataclass(frozen=True, slots=True)
class GuardTimeoutConfig:
    """LangGraph 节点单次 attempt 的超时配置。"""

    run_timeout_seconds: float | None = None
    idle_timeout_seconds: float | None = None
    refresh_on: str = "auto"


def build_retry_policy(config: RetryPolicyConfig | None = None) -> RetryPolicy:
    resolved = config or RetryPolicyConfig()
    if resolved.max_attempts < 1:
        raise ValueError("max_attempts must be greater than or equal to 1.")
    if resolved.initial_interval <= 0:
        raise ValueError("initial_interval must be greater than 0.")
    if resolved.backoff_factor < 1:
        raise ValueError("backoff_factor must be greater than or equal to 1.")
    if resolved.max_interval <= 0:
        raise ValueError("max_interval must be greater than 0.")
    return RetryPolicy(
        initial_interval=resolved.initial_interval,
        backoff_factor=resolved.backoff_factor,
        max_interval=resolved.max_interval,
        max_attempts=resolved.max_attempts,
        jitter=resolved.jitter,
        retry_on=resolved.retry_on,
    )


def build_timeout_policy(config: GuardTimeoutConfig | None = None) -> TimeoutPolicy | None:
    resolved = config or GuardTimeoutConfig()
    _validate_timeout_seconds(resolved.run_timeout_seconds, "run_timeout_seconds")
    _validate_timeout_seconds(resolved.idle_timeout_seconds, "idle_timeout_seconds")
    if resolved.run_timeout_seconds is None and resolved.idle_timeout_seconds is None:
        return None
    if resolved.refresh_on not in {"auto", "heartbeat"}:
        raise ValueError("refresh_on must be 'auto' or 'heartbeat'.")
    return TimeoutPolicy(
        run_timeout=_seconds_to_timedelta(resolved.run_timeout_seconds),
        idle_timeout=_seconds_to_timedelta(resolved.idle_timeout_seconds),
        refresh_on=resolved.refresh_on,
    )


def _validate_timeout_seconds(value: float | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")


def _seconds_to_timedelta(value: float | None) -> timedelta | None:
    return timedelta(seconds=value) if value is not None else None

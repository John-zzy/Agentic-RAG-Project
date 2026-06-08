from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy, TimeoutPolicy

from backend.platform.agent_runtime.failures import build_failure_record
from backend.platform.workflow.langgraph.guards.config import (
    GuardTimeoutConfig,
    RetryPolicyConfig,
    build_retry_policy,
    build_timeout_policy,
)
from backend.platform.workflow.langgraph.guards.error_adapter import (
    GuardErrorHandler,
    build_error_handler,
    clear_last_failure,
    store_last_failure,
)
from backend.platform.workflow.langgraph.guards.metadata import build_guard_metadata


@dataclass(frozen=True, slots=True)
class GuardedNodeConfig:
    """StateGraph.add_node 的 guard 参数集合。"""

    action: Callable[[Any], Any]
    metadata: dict[str, Any]
    retry_policy: RetryPolicy
    timeout: TimeoutPolicy | None
    error_handler: GuardErrorHandler

    def add_node_kwargs(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "retry_policy": self.retry_policy,
            "timeout": self.timeout,
            "error_handler": self.error_handler,
        }


def build_guarded_node_config(
    *,
    graph_name: str,
    node_name: str,
    node: Callable[[Any], Any],
    source: str = "runtime",
    retry_config: RetryPolicyConfig | None = None,
    timeout_config: GuardTimeoutConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GuardedNodeConfig:
    """集中构造 LangGraph guard，图 builder 不再散落框架参数。"""

    guard_metadata = build_guard_metadata(
        graph_name=graph_name,
        node_name=node_name,
        source=source,
        metadata=metadata,
    )
    return GuardedNodeConfig(
        action=wrap_guarded_node(
            graph_name=graph_name,
            node_name=node_name,
            node=node,
            source=source,
            max_attempts=(retry_config or RetryPolicyConfig()).max_attempts,
            metadata=metadata,
        ),
        metadata=guard_metadata,
        retry_policy=build_retry_policy(retry_config),
        timeout=build_timeout_policy(timeout_config),
        error_handler=build_error_handler(),
    )


def register_guarded_node(
    builder: StateGraph,
    node_name: str,
    node: Callable[[Any], Any],
    *,
    graph_name: str,
    source: str = "runtime",
    retry_config: RetryPolicyConfig | None = None,
    timeout_config: GuardTimeoutConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    config = build_guarded_node_config(
        graph_name=graph_name,
        node_name=node_name,
        node=node,
        source=source,
        retry_config=retry_config,
        timeout_config=timeout_config,
        metadata=metadata,
    )
    builder.add_node(node_name, config.action, **config.add_node_kwargs())


def wrap_guarded_node(
    *,
    graph_name: str,
    node_name: str,
    node: Callable[[Any], Any],
    source: str = "runtime",
    max_attempts: int = 2,
    metadata: Mapping[str, Any] | None = None,
) -> Callable[[Any], Any]:
    """捕获异常并写入 state，随后交回 LangGraph retry/error_handler。"""

    def guarded_node(state: Any) -> Any:
        try:
            result = node(state)
        except Exception as exc:
            store_last_failure(
                state,
                build_failure_record(
                    exc,
                    source=source,
                    graph_name=graph_name,
                    node_name=node_name,
                    request_id=_correlation_value(state, "request_id"),
                    run_id=_run_id(state),
                    session_id=_correlation_value(state, "session_id"),
                    turn_id=_state_value(state, "current_turn_id"),
                    step_id=_state_value(state, "current_step_id"),
                    tool_name=_tool_name(state),
                    max_attempts=max_attempts,
                    metadata=metadata,
                ),
                cause=exc,
            )
            raise
        clear_last_failure(state)
        return result

    return guarded_node


def _state_value(state: Any, key: str) -> str | None:
    if isinstance(state, Mapping):
        value = state.get(key)
        return str(value) if value is not None else None
    return None


def _correlation_value(state: Any, key: str) -> str | None:
    value = _state_value(state, key)
    if value is not None:
        return value
    run = _run_payload(state)
    run_value = run.get(key)
    return str(run_value) if run_value is not None else None


def _run_id(state: Any) -> str | None:
    value = _state_value(state, "run_id")
    if value is not None:
        return value
    run = _run_payload(state)
    for key in ("react_run_id", "plan_run_id", "agent_run_id", "run_id"):
        if run.get(key):
            return str(run[key])
    return None


def _tool_name(state: Any) -> str | None:
    if not isinstance(state, Mapping):
        return None
    current_tool_call = state.get("current_tool_call")
    if isinstance(current_tool_call, Mapping) and current_tool_call.get("tool_name"):
        return str(current_tool_call["tool_name"])
    tool_observation = state.get("tool_observation")
    if isinstance(tool_observation, Mapping) and tool_observation.get("tool_name"):
        return str(tool_observation["tool_name"])
    return None


def _run_payload(state: Any) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        return {}
    for key in ("run", "plan_run", "react_run"):
        value = state.get(key)
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "model_dump"):
            return dict(value.model_dump())
    return {}

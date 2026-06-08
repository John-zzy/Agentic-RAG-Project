from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.platform.agent_runtime.failures import FailureRecord

GUARD_FAILURES_METADATA_KEY = "failures"
GUARD_LAST_FAILURE_METADATA_KEY = "_guard_last_failure"
GUARD_LAST_EXCEPTION_METADATA_KEY = "_guard_last_exception"


class GuardedNodeFailureError(RuntimeError):
    """节点 guard 已记录 failure 后抛出的终止异常。"""

    def __init__(self, failure_payload: Mapping[str, Any]) -> None:
        self.failure_payload = dict(failure_payload)
        message = str(self.failure_payload.get("message") or "Guarded node failed.")
        super().__init__(message)


class GuardErrorHandler:
    """把 guard 捕获到的异常 payload 合并回 graph state，然后终止失败路径。"""

    def __init__(self, *, metadata_key: str = GUARD_FAILURES_METADATA_KEY) -> None:
        self._metadata_key = metadata_key

    def __call__(self, state: Any) -> dict[str, Any]:
        metadata = _failure_target_metadata(state)
        failure_payload = metadata.get(GUARD_LAST_FAILURE_METADATA_KEY)
        if not isinstance(failure_payload, Mapping):
            return {}

        updated_metadata = dict(metadata)
        # error_handler 不接收异常对象，只消费 wrapper 放入 state 的可序列化事实。
        updated_metadata.pop(GUARD_LAST_FAILURE_METADATA_KEY, None)
        cause = updated_metadata.pop(GUARD_LAST_EXCEPTION_METADATA_KEY, None)
        updated_metadata[self._metadata_key] = [
            *extract_guard_failures(updated_metadata, metadata_key=self._metadata_key),
            dict(failure_payload),
        ]
        # error_handler 返回 update 会让 LangGraph 继续执行出边；这里必须抛出，避免失败节点被后续节点标记为成功。
        _replace_failure_target_metadata(state, updated_metadata)
        if isinstance(cause, BaseException):
            raise GuardedNodeFailureError(failure_payload) from cause
        raise GuardedNodeFailureError(failure_payload)


def build_error_handler(*, metadata_key: str = GUARD_FAILURES_METADATA_KEY) -> GuardErrorHandler:
    return GuardErrorHandler(metadata_key=metadata_key)


def extract_guard_failures(
    metadata_or_state: Mapping[str, Any],
    *,
    metadata_key: str = GUARD_FAILURES_METADATA_KEY,
) -> list[dict[str, Any]]:
    metadata = _metadata_from_mapping(metadata_or_state)
    failures = metadata.get(metadata_key, [])
    if not isinstance(failures, list):
        return []
    return [dict(item) for item in failures if isinstance(item, Mapping)]


def store_last_failure(
    state: Any,
    failure: FailureRecord,
    *,
    cause: BaseException | None = None,
) -> None:
    metadata = _ensure_failure_target_metadata(state)
    metadata[GUARD_LAST_FAILURE_METADATA_KEY] = failure.to_payload()
    if cause is not None:
        metadata[GUARD_LAST_EXCEPTION_METADATA_KEY] = cause


def clear_last_failure(state: Any) -> None:
    metadata = _failure_target_metadata(state)
    if isinstance(metadata, dict):
        metadata.pop(GUARD_LAST_FAILURE_METADATA_KEY, None)
        metadata.pop(GUARD_LAST_EXCEPTION_METADATA_KEY, None)


def _state_metadata(state: Any) -> Mapping[str, Any]:
    if isinstance(state, Mapping):
        metadata = state.get("metadata")
        if isinstance(metadata, Mapping):
            return metadata
    return {}


def _failure_target_metadata(state: Any) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        return {}
    for key in ("metadata", "run", "plan_run", "react_run"):
        metadata = _metadata_for_state_key(state, key)
        if metadata:
            return metadata
    return {}


def _ensure_failure_target_metadata(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    for key in ("metadata", "run", "plan_run", "react_run"):
        if key in state:
            return _ensure_metadata_for_state_key(state, key)
    metadata: dict[str, Any] = {}
    state["metadata"] = metadata
    return metadata


def _metadata_for_state_key(state: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key == "metadata":
        metadata = state.get("metadata")
        return metadata if isinstance(metadata, Mapping) else {}
    value = state.get(key)
    payload = _model_or_mapping(value)
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _ensure_metadata_for_state_key(state: dict[str, Any], key: str) -> dict[str, Any]:
    if key == "metadata":
        return _ensure_state_metadata(state)
    value = state.get(key)
    if isinstance(value, dict):
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        value["metadata"] = metadata
        return metadata
    if hasattr(value, "metadata") and isinstance(value.metadata, dict):
        return value.metadata
    return _ensure_state_metadata(state)


def _ensure_state_metadata(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    metadata = state.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    state["metadata"] = metadata
    return metadata


def _metadata_from_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = value.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata
    return value


def _replace_failure_target_metadata(state: Any, metadata: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    if "metadata" in state:
        state["metadata"] = metadata
        return
    for key in ("run", "plan_run", "react_run"):
        value = state.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            value["metadata"] = metadata
            return
        if hasattr(value, "metadata") and isinstance(value.metadata, dict):
            value.metadata.clear()
            value.metadata.update(metadata)
            return


def _model_or_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return {}

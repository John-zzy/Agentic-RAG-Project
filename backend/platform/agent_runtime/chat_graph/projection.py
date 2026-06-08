from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from backend.platform.workflow.langgraph.state import RuntimeGraphState
from backend.platform.workflow.state_machine import WorkflowRunState, ensure_workflow_state


@dataclass(frozen=True)
class RuntimeGraphProjection:
    """ChatGraph 对外消费的统一投影，不暴露 LangGraph 原始返回形态。"""

    state: RuntimeGraphState
    status: WorkflowRunState
    answer: str
    citations: list[Any]
    self_check_failure: str | None

    @classmethod
    def from_state(cls, state: RuntimeGraphState) -> RuntimeGraphProjection:
        status = ensure_workflow_state(state.get("status", "running"))
        return cls(
            state=state,
            status=status,
            answer=str(state.get("answer") or ""),
            citations=list(state.get("citations") or []),
            self_check_failure=_self_check_failure_message(state),
        )


def project_runtime_graph_state(value: Any) -> RuntimeGraphProjection:
    """把 LangGraph typed state 收敛成 runtime projection；非法状态直接失败。"""
    if not isinstance(value, Mapping):
        raise TypeError("ChatGraph must return typed RuntimeGraphState.")
    state = _normalize_runtime_graph_state(value)
    return RuntimeGraphProjection.from_state(state)


def _normalize_runtime_graph_state(value: Mapping[str, Any]) -> RuntimeGraphState:
    if not value.get("session_id"):
        raise ValueError("RuntimeGraphState.session_id is required.")
    if not value.get("request_id"):
        raise ValueError("RuntimeGraphState.request_id is required.")
    if "answer" not in value:
        raise ValueError("RuntimeGraphState.answer is required.")
    if "citations" not in value:
        raise ValueError("RuntimeGraphState.citations is required.")
    return dict(value)  # type: ignore[return-value]


def _self_check_failure_message(state: RuntimeGraphState) -> str | None:
    metadata = state.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    report = metadata.get("self_check")
    if not isinstance(report, Mapping):
        return None
    reasons = report.get("failure_reasons")
    if isinstance(reasons, list) and reasons:
        return str(reasons[0])
    return None

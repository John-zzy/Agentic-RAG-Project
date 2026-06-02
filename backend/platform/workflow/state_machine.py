from __future__ import annotations

from typing import Literal


WorkflowRunState = Literal[
    "created",
    "planning",
    "running",
    "waiting_user",
    "retrying",
    "succeeded",
    "failed",
    "cancelled",
]
WorkflowRunEvent = Literal[
    "plan_start",
    "run_start",
    "interrupt",
    "resume_approve",
    "resume_respond",
    "resume_reject",
    "tool_error_retryable",
    "tool_error_final",
    "retry",
    "success",
    "fail",
    "cancel",
]

WORKFLOW_RUN_STATES: frozenset[str] = frozenset(
    {
        "created",
        "planning",
        "running",
        "waiting_user",
        "retrying",
        "succeeded",
        "failed",
        "cancelled",
    }
)
TERMINAL_WORKFLOW_RUN_STATES: frozenset[WorkflowRunState] = frozenset(
    {"succeeded", "failed", "cancelled"}
)

_TRANSITIONS: dict[tuple[WorkflowRunState, WorkflowRunEvent], WorkflowRunState] = {
    ("created", "plan_start"): "planning",
    ("created", "run_start"): "running",
    ("created", "fail"): "failed",
    ("created", "cancel"): "cancelled",
    ("planning", "run_start"): "running",
    ("planning", "interrupt"): "waiting_user",
    ("planning", "tool_error_final"): "failed",
    ("planning", "fail"): "failed",
    ("planning", "cancel"): "cancelled",
    ("running", "interrupt"): "waiting_user",
    ("running", "tool_error_retryable"): "retrying",
    ("running", "tool_error_final"): "failed",
    ("running", "success"): "succeeded",
    ("running", "fail"): "failed",
    ("running", "cancel"): "cancelled",
    ("waiting_user", "resume_approve"): "running",
    ("waiting_user", "resume_respond"): "running",
    ("waiting_user", "resume_reject"): "cancelled",
    ("waiting_user", "cancel"): "cancelled",
    ("retrying", "retry"): "running",
    ("retrying", "tool_error_retryable"): "retrying",
    ("retrying", "tool_error_final"): "failed",
    ("retrying", "fail"): "failed",
    ("retrying", "cancel"): "cancelled",
}


class WorkflowStateError(ValueError):
    """工作流状态机错误的基础类型，供 runtime 统一转换为业务错误。"""


class UnknownWorkflowStateError(WorkflowStateError):
    """状态值不在统一 WorkflowRunState 集合内时抛出。"""


class InvalidWorkflowTransitionError(WorkflowStateError):
    """状态事件组合不在转移表内时抛出。"""

    def __init__(self, current_state: str, event: str) -> None:
        self.current_state = current_state
        self.event = event
        super().__init__(
            f"Invalid workflow transition: {current_state} + {event}."
        )


def ensure_workflow_state(state: str) -> WorkflowRunState:
    """校验并返回统一工作流状态，避免未知状态写入 checkpoint 或 lifecycle。"""
    if state not in WORKFLOW_RUN_STATES:
        raise UnknownWorkflowStateError(f"Unknown workflow state: {state}.")
    return state  # type: ignore[return-value]


def is_terminal(state: str) -> bool:
    """判断状态是否为终态；终态不能被 resume、retry 或重新写成 running。"""
    return ensure_workflow_state(state) in TERMINAL_WORKFLOW_RUN_STATES


def validate_transition(current_state: str, event: str) -> WorkflowRunState:
    """根据集中转移表校验事件，并返回事件对应的下一状态。"""
    current = ensure_workflow_state(current_state)
    key = (current, event)
    try:
        return _TRANSITIONS[key]  # type: ignore[index]
    except KeyError as exc:
        raise InvalidWorkflowTransitionError(current_state, event) from exc

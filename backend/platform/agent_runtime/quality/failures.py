from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from json import JSONDecodeError
from typing import Any
from uuid import uuid4

from langgraph.errors import NodeTimeoutError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.platform.agent_runtime.core.validation import (
    AgentRuntimeValidationError,
    PlanDependencyValidationError,
    ToolAccessValidationError,
    ToolInputValidationError,
)


class FailureCategory(StrEnum):
    """Agent Runtime 对内使用的稳定失败分类。"""

    RUNTIME_ERROR = "runtime_error"
    RUNTIME_TIMEOUT = "runtime_timeout"
    CHECKPOINT_ERROR = "checkpoint_error"
    TOOL_ERROR = "tool_error"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_CONNECTION_ERROR = "tool_connection_error"
    TOOL_VALIDATION_ERROR = "tool_validation_error"
    MODEL_ERROR = "model_error"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_RATE_LIMIT = "model_rate_limit"
    MODEL_CONNECTION_ERROR = "model_connection_error"
    MODEL_EMPTY_OUTPUT = "model_empty_output"
    MODEL_SCHEMA_ERROR = "model_schema_error"
    RETRIEVAL_ERROR = "retrieval_error"
    RETRIEVAL_TIMEOUT = "retrieval_timeout"
    CHECKPOINT_TIMEOUT = "checkpoint_timeout"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    HUMAN_CANCELLED = "human_cancelled"


class FailureRecord(BaseModel):
    """可写入 checkpoint/run metadata 的失败事实。"""

    model_config = ConfigDict(extra="forbid")

    failure_id: str = Field(default_factory=lambda: str(uuid4()))
    category: FailureCategory
    retryable: bool
    message: str
    source: str
    exception_type: str | None = None
    graph_name: str | None = None
    node_name: str | None = None
    request_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    step_id: str | None = None
    tool_name: str | None = None
    attempt: int | None = Field(default=None, ge=0)
    max_attempts: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """输出纯 JSON 友好的 payload，避免 checkpoint 保存异常对象。"""

        return self.model_dump(mode="json")


def classify_exception(
    exc: BaseException,
    *,
    source: str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[FailureCategory, bool]:
    """按异常类型归类；业务节点可通过 metadata 提供 source 细分。"""

    if isinstance(exc, (ToolAccessValidationError, ToolInputValidationError)):
        return FailureCategory.TOOL_VALIDATION_ERROR, False
    if isinstance(exc, PlanDependencyValidationError):
        return FailureCategory.DEPENDENCY_BLOCKED, False
    if isinstance(exc, (ValidationError, JSONDecodeError)):
        return FailureCategory.MODEL_SCHEMA_ERROR, False

    category = _timeout_category(exc, source)
    if category is not None:
        return category, True
    category = _connection_category(exc, source)
    if category is not None:
        return category, True
    category = _source_category(source, metadata or {})
    return category, False


def build_failure_record(
    exc: BaseException,
    *,
    source: str,
    message: str | None = None,
    graph_name: str | None = None,
    node_name: str | None = None,
    request_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    step_id: str | None = None,
    tool_name: str | None = None,
    attempt: int | None = None,
    max_attempts: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FailureRecord:
    """从异常生成稳定失败记录，供 guard/model/tool adapter 复用。"""

    category, retryable = classify_exception(exc, source=source, metadata=metadata)
    return FailureRecord(
        category=category,
        retryable=retryable,
        message=message or str(exc) or exc.__class__.__name__,
        source=source,
        exception_type=exc.__class__.__name__,
        graph_name=graph_name,
        node_name=node_name,
        request_id=request_id,
        run_id=run_id,
        session_id=session_id,
        turn_id=turn_id,
        step_id=step_id,
        tool_name=tool_name,
        attempt=attempt,
        max_attempts=max_attempts,
        metadata=dict(metadata or {}),
    )


def failure_record_from_payload(payload: Mapping[str, Any]) -> FailureRecord:
    """从已序列化 payload 还原，便于测试和后续恢复流程读取。"""

    return FailureRecord.model_validate(dict(payload))


def _timeout_category(exc: BaseException, source: str) -> FailureCategory | None:
    if not isinstance(exc, (TimeoutError, NodeTimeoutError)):
        return None
    if source == "model":
        return FailureCategory.MODEL_TIMEOUT
    if source == "tool":
        return FailureCategory.TOOL_TIMEOUT
    if source == "retrieval":
        return FailureCategory.RETRIEVAL_TIMEOUT
    if source == "checkpoint":
        return FailureCategory.CHECKPOINT_TIMEOUT
    return FailureCategory.RUNTIME_TIMEOUT


def _connection_category(exc: BaseException, source: str) -> FailureCategory | None:
    if not isinstance(exc, ConnectionError):
        return None
    if source == "model":
        return FailureCategory.MODEL_CONNECTION_ERROR
    if source == "tool":
        return FailureCategory.TOOL_CONNECTION_ERROR
    return FailureCategory.RUNTIME_ERROR


def _source_category(source: str, metadata: Mapping[str, Any]) -> FailureCategory:
    if metadata.get("category") in FailureCategory:
        return FailureCategory(str(metadata["category"]))
    if source == "model":
        return FailureCategory.MODEL_ERROR
    if source == "tool":
        return FailureCategory.TOOL_ERROR
    if source == "retrieval":
        return FailureCategory.RETRIEVAL_ERROR
    if source == "checkpoint":
        return FailureCategory.CHECKPOINT_ERROR
    return FailureCategory.RUNTIME_ERROR

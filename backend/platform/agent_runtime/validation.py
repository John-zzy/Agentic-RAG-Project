from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.platform.agent_runtime.contracts import PlanStep, RetryMetadata


class AgentRuntimeValidationError(ValueError):
    """Agent Runtime 合同校验错误的基础类型。"""


class ToolAccessValidationError(AgentRuntimeValidationError):
    """工具不在当前 scene/runtime allowlist 中时抛出。"""


class ToolInputValidationError(AgentRuntimeValidationError):
    """工具输入无法通过 args schema 校验时抛出。"""


class PlanDependencyValidationError(AgentRuntimeValidationError):
    """PlanStep 依赖关系非法或存在环时抛出。"""


def ensure_tool_allowed(tool_name: str, allowed_tools: Sequence[str] | set[str]) -> str:
    """校验工具名是否属于当前 Agent 可调用集合。"""
    allowed = set(allowed_tools)
    if not tool_name:
        raise ToolAccessValidationError("tool_name is required.")
    if tool_name not in allowed:
        raise ToolAccessValidationError(f"Tool is not allowed: {tool_name}.")
    return tool_name


def validate_plan_tool_allowlist(
    steps: Sequence[PlanStep],
    allowed_tools: Sequence[str] | set[str],
) -> None:
    """批量校验 Planner 生成的 step 是否只引用可用工具。"""
    for step in steps:
        ensure_tool_allowed(step.tool_name, allowed_tools)


def validate_tool_input(
    *,
    tool_name: str,
    input_payload: Mapping[str, Any] | None,
    args_schema: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """按工具 args_schema 校验输入；无 schema 时只接受 mapping 结构。"""
    payload = dict(input_payload or {})
    if args_schema is None:
        return payload
    if not isinstance(args_schema, type) or not issubclass(args_schema, BaseModel):
        raise ToolInputValidationError(f"Invalid args schema for tool: {tool_name}.")

    try:
        validated = args_schema.model_validate(payload)
    except ValidationError as exc:
        raise ToolInputValidationError(f"Invalid input for tool: {tool_name}.") from exc
    return validated.model_dump()


def validate_plan_dependencies(steps: Sequence[PlanStep]) -> None:
    """校验 PlanStep 依赖 DAG，拒绝重复 step、未知依赖、自依赖和环。"""
    step_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for step in steps:
        if step.step_id in step_ids:
            duplicate_ids.add(step.step_id)
        step_ids.add(step.step_id)
    if duplicate_ids:
        raise PlanDependencyValidationError(
            f"Duplicate plan step ids: {', '.join(sorted(duplicate_ids))}."
        )

    dependencies_by_step: dict[str, list[str]] = {}
    for step in steps:
        dependencies = list(step.depends_on)
        dependencies_by_step[step.step_id] = dependencies
        if step.step_id in dependencies:
            raise PlanDependencyValidationError(
                f"Plan step cannot depend on itself: {step.step_id}."
            )
        unknown = sorted(set(dependencies) - step_ids)
        if unknown:
            raise PlanDependencyValidationError(
                f"Unknown dependencies for step {step.step_id}: {', '.join(unknown)}."
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visited:
            return
        if step_id in visiting:
            raise PlanDependencyValidationError(
                f"Plan step dependency cycle detected at: {step_id}."
            )
        visiting.add(step_id)
        for dependency_id in dependencies_by_step[step_id]:
            visit(dependency_id)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in dependencies_by_step:
        visit(step_id)


def build_retry_metadata(
    *,
    attempt: int = 0,
    max_attempts: int = 2,
    retryable: bool = True,
    last_error: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RetryMetadata:
    """生成统一重试元数据，供 turn、step 和 tool call 共享默认值。"""
    return RetryMetadata(
        attempt=attempt,
        max_attempts=max_attempts,
        retryable=retryable,
        last_error=last_error,
        metadata=dict(metadata or {}),
    )

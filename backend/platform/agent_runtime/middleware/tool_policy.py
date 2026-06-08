from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.platform.agent_runtime.core.contracts import AgentRuntimeModel
from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext
from backend.platform.agent_runtime.core.validation import (
    ToolAccessValidationError,
    ToolInputValidationError,
    ensure_tool_allowed,
    validate_tool_input,
)


ToolRiskLevel = Literal["low", "medium", "high"]


class ToolPolicyDecision(AgentRuntimeModel):
    allowed: bool
    tool_name: str
    risk_level: ToolRiskLevel = "low"
    input_payload: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class ToolPolicyConfig:
    allowed_tools: set[str] = field(default_factory=set)
    tool_source_scope: dict[str, tuple[str, ...]] = field(default_factory=dict)
    high_risk_tools: set[str] = field(default_factory=set)
    risk_by_tool: dict[str, ToolRiskLevel] = field(default_factory=dict)
    max_calls_per_tool: int | None = None
    retryable_error_types: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError)


class ToolPolicyMiddleware:
    """Validate tool scope, input schema, risk and call limits before invocation."""

    def __init__(self, config: ToolPolicyConfig) -> None:
        self._config = config
        self._call_counts: dict[tuple[str, str], int] = defaultdict(int)

    def validate(
        self,
        *,
        tool_name: str,
        input_payload: Mapping[str, Any] | None,
        context: AgentRuntimeContext,
        args_schema: type[BaseModel] | None = None,
    ) -> ToolPolicyDecision:
        try:
            ensure_tool_allowed(tool_name, self._config.allowed_tools)
            self._ensure_source_scope(tool_name=tool_name, context=context)
            self._ensure_call_limit(tool_name=tool_name, context=context)
            validated_input = validate_tool_input(
                tool_name=tool_name,
                input_payload=input_payload,
                args_schema=args_schema,
            )
        except (ToolAccessValidationError, ToolInputValidationError, ValueError) as exc:
            return ToolPolicyDecision(
                allowed=False,
                tool_name=tool_name,
                retryable=False,
                reason=str(exc),
                metadata={"classification": "policy_rejection"},
            )

        self._call_counts[(context.request_id, tool_name)] += 1
        risk_level = self._risk_level(tool_name)
        return ToolPolicyDecision(
            allowed=True,
            tool_name=tool_name,
            risk_level=risk_level,
            input_payload=validated_input,
            retryable=True,
            metadata={
                "mounted_knowledge_sources": list(context.mounted_knowledge_sources),
                "risk_level": risk_level,
            },
        )

    def classify_retry(self, exc: BaseException) -> bool:
        return isinstance(exc, self._config.retryable_error_types)

    def _ensure_source_scope(self, *, tool_name: str, context: AgentRuntimeContext) -> None:
        required_sources = set(self._config.tool_source_scope.get(tool_name) or ())
        if not required_sources:
            return
        mounted_sources = set(context.mounted_knowledge_sources)
        if not required_sources.issubset(mounted_sources):
            missing = ", ".join(sorted(required_sources - mounted_sources))
            raise ToolAccessValidationError(
                f"Tool is unavailable for mounted knowledge sources: {tool_name}; missing {missing}."
            )

    def _ensure_call_limit(self, *, tool_name: str, context: AgentRuntimeContext) -> None:
        max_calls = self._config.max_calls_per_tool
        if max_calls is None:
            return
        if self._call_counts[(context.request_id, tool_name)] >= max_calls:
            raise ToolAccessValidationError(f"Tool call limit exceeded: {tool_name}.")

    def _risk_level(self, tool_name: str) -> ToolRiskLevel:
        if tool_name in self._config.high_risk_tools:
            return "high"
        configured = self._config.risk_by_tool.get(tool_name)
        if configured:
            return configured
        lowered = tool_name.lower()
        if any(marker in lowered for marker in ("delete", "external", "update", "write")):
            return "high"
        return "low"


def build_tool_policy_config(
    *,
    allowed_tools: Sequence[str],
    tool_source_scope: Mapping[str, Sequence[str]] | None = None,
    high_risk_tools: Sequence[str] = (),
    risk_by_tool: Mapping[str, ToolRiskLevel] | None = None,
    max_calls_per_tool: int | None = None,
) -> ToolPolicyConfig:
    return ToolPolicyConfig(
        allowed_tools=set(allowed_tools),
        tool_source_scope={
            name: tuple(sources)
            for name, sources in dict(tool_source_scope or {}).items()
        },
        high_risk_tools=set(high_risk_tools),
        risk_by_tool=dict(risk_by_tool or {}),
        max_calls_per_tool=max_calls_per_tool,
    )

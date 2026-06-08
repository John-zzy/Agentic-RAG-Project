from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from backend.platform.agent_runtime.core.contracts import AgentRuntimeModel
from backend.platform.agent_runtime.tooling.rag import (
    AGENTIC_RAG_TOOL_NAME,
    NATIVE_RAG_TOOL_NAME,
)


class RuntimeToolPolicy(AgentRuntimeModel):
    """单轮 Agent 运行可见的工具范围，来源于 scene 和会话挂载知识源。"""

    allowed_tools: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    candidate_retrieval_tools: list[str] = Field(default_factory=list)
    default_retrieval_tool: str | None = None
    default_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    high_risk_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        allowed_tools: Sequence[str],
        candidate_retrieval_tools: Sequence[str],
        default_inputs: Mapping[str, Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RuntimeToolPolicy":
        raw_metadata = dict(metadata or {})
        agent_runtime = raw_metadata.get("agent_runtime")
        runtime_policy = (
            dict(agent_runtime.get("tool_policy") or {})
            if isinstance(agent_runtime, Mapping)
            else {}
        )
        raw_policy = dict(raw_metadata.get("tool_policy") or {})
        raw_policy.update(runtime_policy)

        allowed = _deduplicate(allowed_tools)
        retrieval_tools = [
            tool_name
            for tool_name in (AGENTIC_RAG_TOOL_NAME, NATIVE_RAG_TOOL_NAME)
            if tool_name in allowed
        ]
        preferred = _deduplicate(
            _load_string_list(raw_policy.get("preferred_tools")) + retrieval_tools + allowed
        )
        high_risk_tools = [
            tool_name
            for tool_name in _load_string_list(raw_policy.get("high_risk_tools"))
            if tool_name in allowed
        ]
        default_retrieval_tool = _resolve_default_retrieval_tool(
            allowed_tools=allowed,
            configured=raw_policy.get("default_retrieval_tool"),
            retrieval_tools=retrieval_tools,
        )
        return cls(
            allowed_tools=allowed,
            preferred_tools=[tool_name for tool_name in preferred if tool_name in allowed],
            candidate_retrieval_tools=_deduplicate(candidate_retrieval_tools),
            default_retrieval_tool=default_retrieval_tool,
            default_inputs=_coerce_default_inputs(default_inputs),
            high_risk_tools=high_risk_tools,
            metadata=raw_policy,
        )

    def require_default_retrieval_tool(self) -> str:
        """返回默认检索工具；缺失时给出明确配置错误。"""
        if self.default_retrieval_tool:
            return self.default_retrieval_tool
        raise ValueError("No retrieval tool is available for current scene and mounted sources.")


def _resolve_default_retrieval_tool(
    *,
    allowed_tools: Sequence[str],
    configured: Any,
    retrieval_tools: Sequence[str],
) -> str | None:
    if isinstance(configured, str) and configured in allowed_tools:
        return configured
    for tool_name in retrieval_tools:
        if tool_name in allowed_tools:
            return tool_name
    return None


def _coerce_default_inputs(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    inputs: dict[str, dict[str, Any]] = {}
    for tool_name, payload in (value or {}).items():
        if isinstance(payload, Mapping):
            inputs[str(tool_name)] = dict(payload)
    return inputs


def _load_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def _deduplicate(values: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        resolved.append(item)
    return resolved

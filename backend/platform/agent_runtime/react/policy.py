from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from backend.platform.agent_runtime.core.contracts import AgentRuntimeModel


ReActNoEvidenceAction = Literal["ask_user", "final_answer", "stop"]


class ReActScenePolicy(AgentRuntimeModel):
    """scene 暴露给 ReAct selector 的最小策略。"""

    preferred_tools: list[str] = Field(default_factory=list)
    allow_multi_tool: bool = True
    max_turns: int = Field(default=2, ge=1)
    no_evidence_action: ReActNoEvidenceAction = "ask_user"
    high_risk_tools: list[str] = Field(default_factory=list)
    high_risk_action: str = "require_approval"
    tool_input_hints: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any] | None,
        *,
        default_preferred_tools: list[str] | None = None,
        default_max_turns: int = 2,
        default_no_evidence_action: ReActNoEvidenceAction = "ask_user",
        tool_input_hints: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "ReActScenePolicy":
        raw_policy = _load_react_policy_mapping(metadata)
        return cls(
            preferred_tools=_load_string_list(
                raw_policy.get("preferred_tools"),
                fallback=default_preferred_tools or [],
            ),
            allow_multi_tool=bool(raw_policy.get("allow_multi_tool", True)),
            max_turns=_load_positive_int(
                raw_policy.get("max_turns"),
                fallback=default_max_turns,
            ),
            no_evidence_action=_load_no_evidence_action(
                raw_policy.get("no_evidence_action"),
                fallback=default_no_evidence_action,
            ),
            high_risk_tools=_load_string_list(
                raw_policy.get("high_risk_tools") or raw_policy.get("approval_required_tools"),
            ),
            high_risk_action=str(raw_policy.get("high_risk_action") or "require_approval"),
            tool_input_hints=_coerce_tool_input_hints(tool_input_hints),
            metadata=dict(raw_policy),
        )


def public_scene_policy(policy: ReActScenePolicy) -> dict[str, Any]:
    return {
        "preferred_tools": list(policy.preferred_tools),
        "allow_multi_tool": policy.allow_multi_tool,
        "max_turns": policy.max_turns,
        "no_evidence_action": policy.no_evidence_action,
        "high_risk_tools": list(policy.high_risk_tools),
        "high_risk_action": policy.high_risk_action,
        "tool_input_hints": dict(policy.tool_input_hints),
    }


def _load_react_policy_mapping(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    agent_runtime = metadata.get("agent_runtime")
    if isinstance(agent_runtime, Mapping):
        react_policy = agent_runtime.get("react")
        if isinstance(react_policy, Mapping):
            return dict(react_policy)
    react_policy = metadata.get("react_policy")
    return dict(react_policy) if isinstance(react_policy, Mapping) else {}


def _load_string_list(value: Any, *, fallback: list[str] | None = None) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return list(fallback or [])


def _load_positive_int(value: Any, *, fallback: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return fallback
        if parsed > 0:
            return parsed
    return fallback


def _load_no_evidence_action(
    value: Any,
    *,
    fallback: ReActNoEvidenceAction,
) -> ReActNoEvidenceAction:
    if value in {"ask_user", "final_answer", "stop"}:
        return value
    return fallback


def _coerce_tool_input_hints(
    tool_input_hints: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if tool_input_hints is None:
        return {}
    resolved: dict[str, dict[str, Any]] = {}
    for tool_name, input_payload in tool_input_hints.items():
        if isinstance(input_payload, Mapping):
            resolved[str(tool_name)] = dict(input_payload)
    return resolved

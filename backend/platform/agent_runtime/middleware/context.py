from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from backend.platform.agent_runtime.core.contracts import AgentRuntimeModel
from backend.platform.workflow.state_machine import WorkflowRunState


_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "checkpoint",
    "chain_of_thought",
    "history",
    "messages",
    "password",
    "prompt",
    "raw",
    "secret",
    "token",
)


class WorkflowRuntimeMetadata(AgentRuntimeModel):
    """Workflow metadata needed by middleware without reading global state."""

    run_id: str | None = None
    thread_id: str | None = None
    checkpoint_ns: str | None = None
    node_name: str | None = None
    interrupt_id: str | None = None
    status: WorkflowRunState | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SafeAuditMetadata(AgentRuntimeModel):
    """Safe audit dimensions that can be serialized with traces or state."""

    trace_id: str | None = None
    actor_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRuntimeContext(AgentRuntimeModel):
    """Typed runtime context shared by prompt, model, tool, HITL and trace middleware."""

    session_id: str
    request_id: str
    scene: str
    mounted_knowledge_sources: tuple[str, ...] = ("documents",)
    complexity: str = "simple"
    provider_name: str | None = None
    workflow: WorkflowRuntimeMetadata = Field(default_factory=WorkflowRuntimeMetadata)
    audit: SafeAuditMetadata = Field(default_factory=SafeAuditMetadata)
    scene_metadata: dict[str, Any] = Field(default_factory=dict)
    request_metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        session_id: str,
        request_id: str,
        scene: str,
        mounted_knowledge_sources: Sequence[str] = ("documents",),
        complexity: str = "simple",
        provider_name: str | None = None,
        workflow: WorkflowRuntimeMetadata | Mapping[str, Any] | None = None,
        audit: SafeAuditMetadata | Mapping[str, Any] | None = None,
        scene_metadata: Mapping[str, Any] | None = None,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> "AgentRuntimeContext":
        return cls(
            session_id=session_id,
            request_id=request_id,
            scene=scene,
            mounted_knowledge_sources=tuple(mounted_knowledge_sources),
            complexity=complexity,
            provider_name=provider_name,
            workflow=_coerce_workflow(workflow),
            audit=_coerce_audit(audit),
            scene_metadata=dict(scene_metadata or {}),
            request_metadata=dict(request_metadata or {}),
        )

    def to_safe_metadata(self) -> dict[str, Any]:
        """Serialize context for audit without secrets, prompts, history or raw payloads."""

        return _sanitize_mapping(
            {
                "session_id": self.session_id,
                "request_id": self.request_id,
                "scene": self.scene,
                "mounted_knowledge_sources": list(self.mounted_knowledge_sources),
                "complexity": self.complexity,
                "provider_name": self.provider_name,
                "workflow": self.workflow.model_dump(),
                "audit": self.audit.model_dump(),
                "scene_metadata": self.scene_metadata,
                "request_metadata": self.request_metadata,
            }
        )


def _coerce_workflow(
    value: WorkflowRuntimeMetadata | Mapping[str, Any] | None,
) -> WorkflowRuntimeMetadata:
    if isinstance(value, WorkflowRuntimeMetadata):
        return value
    return WorkflowRuntimeMetadata.model_validate(dict(value or {}))


def _coerce_audit(value: SafeAuditMetadata | Mapping[str, Any] | None) -> SafeAuditMetadata:
    if isinstance(value, SafeAuditMetadata):
        return value
    return SafeAuditMetadata.model_validate(dict(value or {}))


def _sanitize_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        key_lower = str(key).lower()
        if any(part in key_lower for part in _SENSITIVE_KEY_PARTS):
            continue
        sanitized[str(key)] = _sanitize_value(value)
    return sanitized


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(item) for item in value]
    return value

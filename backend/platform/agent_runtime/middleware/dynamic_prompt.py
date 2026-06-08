from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from backend.platform.agent_runtime.core.contracts import AgentRuntimeModel
from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext
from backend.platform.agent_runtime.middleware.trace import sanitize_for_trace


class DynamicPromptInput(AgentRuntimeModel):
    """Scene-owned prompt material and safe runtime context for prompt composition."""

    scene_prompt: str
    history_view: tuple[str, ...] = ()
    mounted_knowledge_policy: str | None = None
    resume_metadata: dict[str, Any] = Field(default_factory=dict)


class DynamicPromptResult(AgentRuntimeModel):
    """Composed prompt plus safe metadata for audit or tests."""

    system_prompt: str
    sections: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DynamicPromptMiddleware:
    """Compose prompts without moving scene-specific wording into platform code."""

    def compose(
        self,
        *,
        context: AgentRuntimeContext,
        prompt_input: DynamicPromptInput,
    ) -> DynamicPromptResult:
        sections = [prompt_input.scene_prompt.strip()]
        if prompt_input.history_view:
            sections.append("\n".join(item.strip() for item in prompt_input.history_view if item.strip()))
        if prompt_input.mounted_knowledge_policy:
            sections.append(prompt_input.mounted_knowledge_policy.strip())
        safe_resume_metadata = sanitize_for_trace(prompt_input.resume_metadata)
        if safe_resume_metadata:
            sections.append(f"Resume metadata: {safe_resume_metadata}")

        filtered_sections = [section for section in sections if section]
        return DynamicPromptResult(
            system_prompt="\n\n".join(filtered_sections),
            sections=filtered_sections,
            metadata={
                "session_id": context.session_id,
                "request_id": context.request_id,
                "scene": context.scene,
                "mounted_knowledge_sources": list(context.mounted_knowledge_sources),
                "has_resume_metadata": bool(safe_resume_metadata),
            },
        )

    def compose_from_parts(
        self,
        *,
        context: AgentRuntimeContext,
        scene_prompt: str,
        history_view: Sequence[str] = (),
        mounted_knowledge_policy: str | None = None,
        resume_metadata: Mapping[str, Any] | None = None,
    ) -> DynamicPromptResult:
        return self.compose(
            context=context,
            prompt_input=DynamicPromptInput(
                scene_prompt=scene_prompt,
                history_view=tuple(history_view),
                mounted_knowledge_policy=mounted_knowledge_policy,
                resume_metadata=dict(resume_metadata or {}),
            ),
        )

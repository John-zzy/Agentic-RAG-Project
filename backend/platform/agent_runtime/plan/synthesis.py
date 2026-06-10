from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import Field

from backend.platform.agent_runtime.core.contracts import (
    AgentRuntimeModel,
    PlanStep,
    ToolObservation,
)


class PlanSynthesisContext(AgentRuntimeModel):
    """Plan final synthesizer 的输入，只包含已完成步骤和工具观察。"""

    plan_run_id: str
    session_id: str
    request_id: str
    user_goal: str
    context_summary: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    observations: list[ToolObservation] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanSynthesisResult(AgentRuntimeModel):
    """Plan 图写回 PlanRun 的最终汇总结果。"""

    final_answer: str
    result_summary: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_used: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanFinalSynthesizer(Protocol):
    """从成功 PlanStep 汇总最终回答的中立协议。"""

    def synthesize(self, context: PlanSynthesisContext) -> PlanSynthesisResult:
        """Return the final answer for a completed PlanRun."""


class StepSummarySynthesizer:
    """默认汇总器：按成功 step 的 result_summary 生成最终回答。"""

    def synthesize(self, context: PlanSynthesisContext) -> PlanSynthesisResult:
        summaries = [
            step.result_summary or f"{step.step_id} succeeded."
            for step in context.steps
            if step.status == "succeeded"
        ]
        final_answer = "\n".join(summaries) if summaries else "No successful plan steps were collected."
        citations = _deduplicate_citations(context.citations)
        return PlanSynthesisResult(
            final_answer=final_answer,
            result_summary=f"Synthesized {len(summaries)} successful plan step(s).",
            citations=citations,
            knowledge_used=bool(citations),
            metadata={"step_count": len(context.steps)},
        )


PlanSummarySynthesizer = StepSummarySynthesizer


def _deduplicate_citations(citations: Sequence[Mapping[str, Any] | dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for citation in citations:
        citation_dict = dict(citation)
        citation_id = str(citation_dict.get("citation_id") or citation_dict)
        if citation_id in seen:
            continue
        seen.add(citation_id)
        deduplicated.append(citation_dict)
    return deduplicated

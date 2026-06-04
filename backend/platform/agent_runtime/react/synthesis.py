from __future__ import annotations

from typing import Protocol

from pydantic import Field

from backend.platform.agent_runtime.contracts import (
    AgentRuntimeModel,
    ReActTurn,
    ToolObservation,
)


class ReActSynthesisContext(AgentRuntimeModel):
    """最终汇总器输入，只包含已完成的顶层观察和引用。"""

    react_run_id: str
    session_id: str
    request_id: str
    user_goal: str
    turns: list[ReActTurn] = Field(default_factory=list)
    observations: list[ToolObservation] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    turn_order: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ReActSynthesisResult(AgentRuntimeModel):
    """ReAct 顶层最终回答和可写入 run metadata 的引用信息。"""

    final_answer: str
    result_summary: str = ""
    citations: list[dict] = Field(default_factory=list)
    knowledge_used: bool = False
    metadata: dict = Field(default_factory=dict)


class ReActFinalSynthesizer(Protocol):
    """从 ReAct observations 汇总最终回答的中立协议。"""

    def synthesize(self, context: ReActSynthesisContext) -> ReActSynthesisResult:
        """Return the final answer for a completed ReAct run."""


class ObservationSummarySynthesizer:
    """默认最终汇总器：按观察摘要和 citations 生成可审计回答。"""

    def synthesize(self, context: ReActSynthesisContext) -> ReActSynthesisResult:
        successful = [observation for observation in context.observations if observation.success]
        if successful:
            summaries = [
                observation.result_summary or f"{observation.tool_name} succeeded."
                for observation in successful
            ]
            final_answer = "\n".join(summaries)
        else:
            final_answer = "No successful tool observations were collected."
        citations = _deduplicate_citations(context.citations)
        return ReActSynthesisResult(
            final_answer=final_answer,
            result_summary=f"Synthesized {len(successful)} successful observation(s).",
            citations=citations,
            knowledge_used=bool(citations),
            metadata={"observation_count": len(context.observations)},
        )


def collect_citations(observations: list[ToolObservation]) -> list[dict]:
    citations: list[dict] = []
    for observation in observations:
        if observation.success:
            citations.extend(observation.citations)
    return _deduplicate_citations(citations)


def _deduplicate_citations(citations: list[dict]) -> list[dict]:
    deduplicated: list[dict] = []
    seen: set[str] = set()
    for citation in citations:
        citation_id = str(citation.get("citation_id") or citation)
        if citation_id in seen:
            continue
        seen.add(citation_id)
        deduplicated.append(dict(citation))
    return deduplicated

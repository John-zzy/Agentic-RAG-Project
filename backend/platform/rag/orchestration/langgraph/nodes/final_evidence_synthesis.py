from __future__ import annotations

from typing import Any

from backend.platform.rag.orchestration.langgraph.config import AgenticRagGraphDependencies
from backend.platform.rag.orchestration.langgraph.state import (
    AgenticRagGraphState,
    build_decision_log_entry,
    build_document_citations,
    build_round_snapshot,
    build_retrieval_trace_snapshot,
    build_tool_observation_snapshot,
)


def build_final_evidence_synthesis_node(_dependencies: AgenticRagGraphDependencies):
    """在证据被接受时写入最终决策和公开可见的 evidence trace。"""

    def final_evidence_synthesis(state: AgenticRagGraphState) -> dict[str, Any]:
        plan = state["plan"]
        result = state["current_result"]
        decision = state.get("current_decision")
        if result is None or decision is None:
            raise ValueError(
                "Agentic RAG final synthesis node requires a current result and decision."
            )

        success = bool(decision.is_sufficient and result.success)
        citations = list(result.citations) if success else []
        if success and not citations:
            citations = build_document_citations(state["documents"])
        knowledge_used = bool(citations and success)
        final_decision_label = "answer_with_evidence"
        exit_reason = "sufficient" if decision.is_sufficient else "finished_by_judge"
        round_snapshot = build_round_snapshot(
            plan=plan,
            result=result,
            decision=decision,
            results=state["results"],
            documents=state["documents"],
            rewrite=state.get("current_rewrite"),
        )
        decision_log_entry = build_decision_log_entry(
            round_snapshot=round_snapshot,
            exit_reason=exit_reason,
        )
        retrieval_trace = build_retrieval_trace_snapshot(
            query=state["query"],
            final_query=plan.active_query,
            success=success,
            final_decision_label=final_decision_label,
            knowledge_used=knowledge_used,
            citations=citations,
            rounds=[*state["rounds"], round_snapshot],
            decision_log=[*state["decision_log"], decision_log_entry],
            exit_reason=exit_reason,
            follow_up_question=decision.follow_up_question,
            candidate_tools=plan.candidate_tools,
        )
        return {
            "rounds": [*state["rounds"], round_snapshot],
            "decision_log": [*state["decision_log"], decision_log_entry],
            "final_decision": decision,
            "final_decision_label": final_decision_label,
            "follow_up_question": decision.follow_up_question,
            "knowledge_used": knowledge_used,
            "citations": citations,
            "success": success,
            "exit_reason": exit_reason,
            "retrieval_trace": retrieval_trace,
            "tool_observation": build_tool_observation_snapshot(
                tool_name=result.tool_name,
                result=result,
                retrieval_trace=retrieval_trace,
                final_decision_label=final_decision_label,
                knowledge_used=knowledge_used,
                follow_up_question=decision.follow_up_question,
                success=success,
                citations=citations,
            ),
            "final_plan": plan,
        }

    return final_evidence_synthesis

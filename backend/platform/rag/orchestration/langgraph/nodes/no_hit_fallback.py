from __future__ import annotations

from typing import Any

from backend.platform.rag.orchestration.langgraph.config import AgenticRagGraphDependencies
from backend.platform.rag.orchestration.langgraph.state import (
    AgenticRagGraphState,
    build_decision_log_entry,
    build_round_snapshot,
    build_retrieval_trace_snapshot,
    build_tool_observation_snapshot,
)


def build_no_hit_fallback_node(_dependencies: AgenticRagGraphDependencies):
    """统一写入 ask_user、no_evidence、max_rounds_reached 和 retrieval_failed 结论。"""

    def no_hit_fallback(state: AgenticRagGraphState) -> dict[str, Any]:
        plan = state["plan"]
        result = state["current_result"]
        decision = state.get("current_decision")
        if result is None or decision is None:
            raise ValueError("Agentic RAG fallback node requires a current result and decision.")

        final_decision = decision
        exit_reason = "ask_user"
        final_decision_label = "ask_user"
        success = False
        knowledge_used = False
        citations: list[dict[str, Any]] = []

        if plan.round_index >= plan.max_rounds and decision.next_action != "finish":
            final_decision = decision.model_copy(
                update={
                    "is_sufficient": False,
                    "next_action": "ask_user",
                    "reason": (
                        f"{decision.reason} Reached max retrieval rounds ({plan.max_rounds})."
                    ),
                }
            )
            exit_reason = "max_rounds_reached"
            final_decision_label = "max_rounds_reached"
        elif result.error:
            exit_reason = "retrieval_failed"
            final_decision_label = "retrieval_failed"
        elif (
            decision.next_action == "finish"
            and result.success
            and not (result.citations or state["documents"])
        ):
            exit_reason = "no_evidence"
            final_decision_label = "no_evidence"
        elif decision.next_action == "ask_user":
            exit_reason = "ask_user"
            final_decision_label = "ask_user"

        round_snapshot = build_round_snapshot(
            plan=plan,
            result=result,
            decision=final_decision,
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
            follow_up_question=final_decision.follow_up_question,
            candidate_tools=plan.candidate_tools,
        )
        return {
            "rounds": [*state["rounds"], round_snapshot],
            "decision_log": [*state["decision_log"], decision_log_entry],
            "final_decision": final_decision,
            "final_decision_label": final_decision_label,
            "follow_up_question": final_decision.follow_up_question,
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
                follow_up_question=final_decision.follow_up_question,
                success=success,
                citations=citations,
            ),
            "final_plan": plan,
        }

    return no_hit_fallback

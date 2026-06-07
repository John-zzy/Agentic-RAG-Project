from __future__ import annotations

from typing import Any

from backend.platform.rag.orchestration.retrieval_graph.config import AgenticRagGraphDependencies
from backend.platform.rag.contracts import RetrievalContext
from backend.platform.rag.orchestration.retrieval_graph.state import (
    AgenticRagGraphState,
    build_decision_log_entry,
    build_round_snapshot,
)


def build_query_rewrite_node(dependencies: AgenticRagGraphDependencies):
    """执行一次查询改写，并把当前轮次收口进审计轨迹。"""

    retriever = dependencies.retriever

    def query_rewrite(state: AgenticRagGraphState) -> dict[str, Any]:
        plan = state["plan"]
        decision = state["current_decision"]
        if decision is None:
            raise ValueError("Agentic RAG rewrite node requires a sufficiency decision.")
        if retriever._rewrite_already_attempted(plan):
            bounded_decision = retriever._build_rewrite_limit_decision(decision)
            round_snapshot = build_round_snapshot(
                plan=plan,
                result=state["current_result"],
                decision=bounded_decision,
                results=state["results"],
                documents=state["documents"],
            )
            decision_log_entry = build_decision_log_entry(
                round_snapshot=round_snapshot,
                exit_reason="ask_user",
            )
            return {
                "current_decision": bounded_decision,
                "follow_up_question": bounded_decision.follow_up_question,
                "rounds": [*state["rounds"], round_snapshot],
                "decision_log": [*state["decision_log"], decision_log_entry],
            }

        rewrite = retriever._rewrite_query(
            RetrievalContext(
                plan=plan,
                results=list(state["results"]),
                documents=list(state["documents"]),
            ),
            dependencies.run_manager,
        )
        next_plan = plan.create_followup(
            active_query=rewrite.query,
            metadata={
                "rewrite_attempted": True,
                "rewrite_reason": rewrite.reason,
                "rewrite_metadata": rewrite.metadata,
            },
        )
        round_snapshot = build_round_snapshot(
            plan=plan,
            result=state["current_result"],
            decision=decision,
            results=state["results"],
            documents=state["documents"],
            rewrite=rewrite,
        )
        decision_log_entry = build_decision_log_entry(
            round_snapshot=round_snapshot,
            exit_reason="continue",
        )
        return {
            "plan": next_plan,
            "final_plan": next_plan,
            "active_query": rewrite.query,
            "rewritten_query": rewrite.query,
            "current_rewrite": None,
            "current_result": None,
            "current_decision": None,
            "rounds": [*state["rounds"], round_snapshot],
            "decision_log": [*state["decision_log"], decision_log_entry],
            "route_next_action": None,
        }

    return query_rewrite


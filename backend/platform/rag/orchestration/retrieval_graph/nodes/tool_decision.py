from __future__ import annotations

from typing import Any

from backend.platform.rag.orchestration.retrieval_graph.config import AgenticRagGraphDependencies
from backend.platform.rag.orchestration.retrieval_graph.state import (
    AgenticRagGraphState,
    build_decision_log_entry,
    build_round_snapshot,
)


def build_tool_decision_node(dependencies: AgenticRagGraphDependencies):
    """选择初始工具，或在 switch_tool 分支中选下一轮工具。"""

    retriever = dependencies.retriever

    def tool_decision(state: AgenticRagGraphState) -> dict[str, Any]:
        plan = state["plan"]
        decision = state.get("current_decision")

        if decision is not None and decision.next_action == "switch_tool":
            next_tool = retriever._resolve_next_tool(plan, decision)
            resolved_query = str(
                decision.metadata.get("resolved_query") or plan.active_query or state["active_query"]
            )
            next_plan = plan.create_followup(
                active_query=resolved_query,
                selected_tool=next_tool,
            )
            round_snapshot = build_round_snapshot(
                plan=plan,
                result=state["current_result"],
                decision=decision,
                results=state["results"],
                documents=state["documents"],
                rewrite=state.get("current_rewrite"),
            )
            decision_log_entry = build_decision_log_entry(
                round_snapshot=round_snapshot,
                exit_reason="continue",
                extra_metadata={"next_tool": next_tool},
            )
            return {
                "plan": next_plan,
                "final_plan": next_plan,
                "selected_tool": next_tool,
                "active_query": resolved_query,
                "attempted_tools": next_plan.attempted_tools,
                "rounds": [*state["rounds"], round_snapshot],
                "decision_log": [*state["decision_log"], decision_log_entry],
                "current_result": None,
                "current_decision": None,
                "current_rewrite": None,
                "rewritten_query": None,
                "route_next_action": None,
            }

        return {
            "selected_tool": plan.selected_tool,
            "active_query": plan.active_query,
            "attempted_tools": plan.attempted_tools,
        }

    return tool_decision


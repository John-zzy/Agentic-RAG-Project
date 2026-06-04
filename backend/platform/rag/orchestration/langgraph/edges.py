from __future__ import annotations

from backend.platform.rag.orchestration.langgraph.state import AgenticRagGraphState

INITIALIZE_PLAN = "initialize_plan"
TOOL_DECISION = "tool_decision"
RETRIEVAL = "retrieval"
RERANK = "rerank"
SUFFICIENCY_CHECK = "sufficiency_check"
ROUTE_NEXT_ACTION = "route_next_action"
QUERY_REWRITE = "query_rewrite"
NO_HIT_FALLBACK = "no_hit_fallback"
FINAL_EVIDENCE_SYNTHESIS = "final_evidence_synthesis"


def build_route_next_action_edge():
    """只做路由判断，不在节点里夹带分支逻辑。"""

    def route_next_action(state: AgenticRagGraphState) -> str:
        plan = state.get("plan")
        decision = state.get("current_decision")
        if plan is None or decision is None:
            return NO_HIT_FALLBACK

        if plan.round_index >= plan.max_rounds and decision.next_action != "finish":
            return NO_HIT_FALLBACK

        if decision.next_action == "rewrite":
            if bool(plan.metadata.get("rewrite_attempted")):
                return NO_HIT_FALLBACK
            return QUERY_REWRITE

        if decision.next_action == "switch_tool":
            return TOOL_DECISION

        if decision.is_sufficient or decision.next_action == "finish":
            return FINAL_EVIDENCE_SYNTHESIS

        if decision.next_action == "ask_user":
            return NO_HIT_FALLBACK

        return NO_HIT_FALLBACK

    return route_next_action

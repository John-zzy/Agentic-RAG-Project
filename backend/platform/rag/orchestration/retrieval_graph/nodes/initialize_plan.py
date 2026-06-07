from __future__ import annotations

from typing import Any

from backend.platform.rag.contracts import RetrievalPlan
from backend.platform.rag.orchestration.retrieval_graph.config import AgenticRagGraphDependencies
from backend.platform.rag.orchestration.retrieval_graph.state import AgenticRagGraphState


def build_initialize_plan_node(dependencies: AgenticRagGraphDependencies):
    """把检索计划和初始候选工具写入图状态。"""

    retriever = dependencies.retriever

    def initialize_plan(state: AgenticRagGraphState) -> dict[str, Any]:
        retriever._validate_tools()

        candidate_tools = retriever._resolve_candidate_tools(state.get("candidate_tools"))
        selected_tool = retriever._resolve_initial_tool(
            state.get("selected_tool"),
            candidate_tools,
        )
        plan = RetrievalPlan(
            user_query=state["query"],
            active_query=state.get("active_query") or state["query"],
            selected_tool=selected_tool,
            max_rounds=int(state.get("max_rounds") or retriever.max_rounds),
            candidate_tools=candidate_tools,
            attempted_tools=(),
            previous_queries=(),
            filters=dict(state.get("filters") or {}),
            top_k=state.get("top_k"),
            min_relevance_score=state.get("min_relevance_score"),
            recall_strategy=str(state.get("recall_strategy") or "hybrid"),
            rerank_enabled=bool(state.get("rerank_enabled", False)),
            rerank_top_n=state.get("rerank_top_n"),
        )

        return {
            "plan": plan,
            "final_plan": plan,
            "selected_tool": selected_tool,
            "candidate_tools": candidate_tools,
            "active_query": plan.active_query,
            "attempted_tools": (),
            "results": [],
            "documents": [],
            "candidate_docs": [],
            "rounds": [],
            "decision_log": [],
            "current_result": None,
            "current_decision": None,
            "current_rewrite": None,
            "retrieval_trace": {},
            "tool_observation": None,
            "citations": [],
            "knowledge_used": False,
            "final_decision": None,
            "final_decision_label": None,
            "follow_up_question": None,
            "success": False,
            "exit_reason": None,
            "route_next_action": None,
        }

    return initialize_plan


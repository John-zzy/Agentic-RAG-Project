from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.platform.rag.orchestration.retrieval_graph.edges import (
    FINAL_EVIDENCE_SYNTHESIS,
    INITIALIZE_PLAN,
    NO_HIT_FALLBACK,
    QUERY_REWRITE,
    RERANK,
    RETRIEVAL,
    ROUTE_NEXT_ACTION,
    SUFFICIENCY_CHECK,
    TOOL_DECISION,
    build_route_next_action_edge,
)
from backend.platform.rag.orchestration.retrieval_graph.config import (
    AgenticRagGraphContext,
    AgenticRagGraphDependencies,
)
from backend.platform.rag.orchestration.retrieval_graph.nodes import (
    build_final_evidence_synthesis_node,
    build_initialize_plan_node,
    build_no_hit_fallback_node,
    build_query_rewrite_node,
    build_rerank_node,
    build_retrieval_node,
    build_route_next_action_node,
    build_sufficiency_check_node,
    build_tool_decision_node,
)
from backend.platform.rag.orchestration.retrieval_graph.state import AgenticRagGraphState
from backend.platform.workflow.langgraph.guards import register_guarded_node

AGENTIC_RAG_GRAPH_NAME = "agentic_rag_graph"
GUARDED_AGENTIC_RAG_NODES = {
    RETRIEVAL: "retrieval",
    QUERY_REWRITE: "retrieval",
    SUFFICIENCY_CHECK: "retrieval",
    FINAL_EVIDENCE_SYNTHESIS: "retrieval",
}


def build_agentic_rag_graph(
    dependencies: AgenticRagGraphDependencies,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """编排 Agentic RAG retrieval graph 拓扑。"""
    builder = StateGraph(AgenticRagGraphState, context_schema=AgenticRagGraphContext)
    builder.add_node(INITIALIZE_PLAN, build_initialize_plan_node(dependencies))
    builder.add_node(TOOL_DECISION, build_tool_decision_node(dependencies))
    _add_guarded_node(builder, RETRIEVAL, build_retrieval_node(dependencies))
    builder.add_node(RERANK, build_rerank_node(dependencies))
    _add_guarded_node(builder, SUFFICIENCY_CHECK, build_sufficiency_check_node(dependencies))
    builder.add_node(ROUTE_NEXT_ACTION, build_route_next_action_node(dependencies))
    _add_guarded_node(builder, QUERY_REWRITE, build_query_rewrite_node(dependencies))
    builder.add_node(NO_HIT_FALLBACK, build_no_hit_fallback_node(dependencies))
    _add_guarded_node(
        builder,
        FINAL_EVIDENCE_SYNTHESIS,
        build_final_evidence_synthesis_node(dependencies),
    )

    builder.add_edge(START, INITIALIZE_PLAN)
    builder.add_edge(INITIALIZE_PLAN, TOOL_DECISION)
    builder.add_edge(TOOL_DECISION, RETRIEVAL)
    builder.add_edge(RETRIEVAL, RERANK)
    builder.add_edge(RERANK, SUFFICIENCY_CHECK)
    builder.add_edge(SUFFICIENCY_CHECK, ROUTE_NEXT_ACTION)
    builder.add_conditional_edges(ROUTE_NEXT_ACTION, build_route_next_action_edge())
    builder.add_edge(QUERY_REWRITE, RETRIEVAL)
    builder.add_edge(NO_HIT_FALLBACK, END)
    builder.add_edge(FINAL_EVIDENCE_SYNTHESIS, END)

    return builder.compile(checkpointer=checkpointer)


def _add_guarded_node(builder: StateGraph, node_name: str, node: Any) -> None:
    # Agentic RAG 暂无日志 wrapper，这里只接入统一 guard 参数。
    register_guarded_node(
        builder,
        node_name,
        node,
        graph_name=AGENTIC_RAG_GRAPH_NAME,
        source=GUARDED_AGENTIC_RAG_NODES[node_name],
        metadata={"guard_scope": "agentic_rag_graph"},
    )

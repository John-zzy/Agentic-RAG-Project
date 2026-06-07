from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.platform.agent_runtime.chat_graph.graph.config import ChatGraphDependencies
from backend.platform.agent_runtime.chat_graph.graph.edges import (
    PLAN_BRANCH,
    REACT_BRANCH,
    RESOLVE_ANSWER_MODE,
    build_route_mode_edge,
)
from backend.platform.agent_runtime.chat_graph.graph.nodes.final_synthesis import (
    build_final_synthesis_node,
)
from backend.platform.agent_runtime.chat_graph.graph.nodes.maybe_hitl_wait import (
    build_maybe_hitl_wait_node,
)
from backend.platform.agent_runtime.chat_graph.graph.nodes.persist_turn import (
    build_persist_turn_node,
)
from backend.platform.agent_runtime.chat_graph.graph.nodes.plan_branch import (
    build_plan_branch_node,
)
from backend.platform.agent_runtime.chat_graph.graph.nodes.prepare_turn import (
    build_prepare_turn_node,
)
from backend.platform.agent_runtime.chat_graph.graph.nodes.react_branch import (
    build_react_branch_node,
)
from backend.platform.agent_runtime.chat_graph.graph.nodes.resolve_answer_mode import (
    build_resolve_answer_mode_node,
)
from backend.platform.agent_runtime.chat_graph.graph.nodes.route_mode import (
    build_route_mode_node,
)
from backend.platform.agent_runtime.chat_graph.graph.nodes.select_mode import (
    build_select_mode_node,
)
from backend.platform.agent_runtime.graph_logging import (
    wrap_graph_node,
    wrap_graph_route,
)
from backend.platform.workflow.langgraph.state import RuntimeGraphState

CHAT_GRAPH_NAME = "chat_graph"


def build_chat_graph(
    dependencies: ChatGraphDependencies,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """编排顶层 ChatGraph 拓扑。"""
    builder = StateGraph(RuntimeGraphState)

    _add_logged_node(builder, "prepare_turn", build_prepare_turn_node(dependencies))
    _add_logged_node(builder, "select_mode", build_select_mode_node(dependencies))
    _add_logged_node(builder, "route_mode", build_route_mode_node(dependencies))
    _add_logged_node(builder, REACT_BRANCH, build_react_branch_node(dependencies))
    _add_logged_node(builder, PLAN_BRANCH, build_plan_branch_node(dependencies))
    _add_logged_node(
        builder,
        RESOLVE_ANSWER_MODE,
        build_resolve_answer_mode_node(dependencies),
    )
    _add_logged_node(builder, "maybe_hitl_wait", build_maybe_hitl_wait_node(dependencies))
    _add_logged_node(builder, "final_synthesis", build_final_synthesis_node(dependencies))
    _add_logged_node(builder, "persist_turn", build_persist_turn_node(dependencies))

    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "select_mode")
    builder.add_edge("select_mode", "route_mode")
    builder.add_conditional_edges(
        "route_mode",
        wrap_graph_route(
            graph_name=CHAT_GRAPH_NAME,
            route_name="route_mode",
            route=build_route_mode_edge(dependencies),
        ),
    )
    builder.add_edge(REACT_BRANCH, RESOLVE_ANSWER_MODE)
    builder.add_edge(PLAN_BRANCH, RESOLVE_ANSWER_MODE)
    builder.add_edge(RESOLVE_ANSWER_MODE, "maybe_hitl_wait")
    builder.add_conditional_edges(
        "maybe_hitl_wait",
        wrap_graph_route(
            graph_name=CHAT_GRAPH_NAME,
            route_name="maybe_hitl_wait",
            route=lambda state: END
            if state.get("status") == "waiting_user"
            else "final_synthesis",
        ),
    )
    builder.add_edge("final_synthesis", "persist_turn")
    builder.add_edge("persist_turn", END)

    return builder.compile(checkpointer=checkpointer)


def _add_logged_node(builder: StateGraph, node_name: str, node: Any) -> None:
    builder.add_node(
        node_name,
        wrap_graph_node(
            graph_name=CHAT_GRAPH_NAME,
            node_name=node_name,
            node=node,
        ),
    )


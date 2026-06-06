from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.edges import (
    CREATE_PLAN,
    EXECUTE_STEP,
    HANDLE_RETRY,
    HANDLE_WAITING_USER,
    SELECT_NEXT_STEP,
    SYNTHESIZE_PLAN_RESULT,
    build_handle_retry_edge,
)
from backend.platform.agent_runtime.plan.graph.nodes import (
    build_create_plan_node,
    build_execute_step_node,
    build_handle_retry_node,
    build_handle_waiting_user_node,
    build_select_next_step_node,
    build_synthesize_plan_result_node,
    build_synthesize_result_node,
)
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState
from backend.platform.agent_runtime.graph_logging import (
    wrap_graph_node,
    wrap_graph_route,
)

PLAN_GRAPH_NAME = "plan_graph"


def build_plan_graph(
    dependencies: PlanGraphDependencies,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """编排 Plan 子图拓扑。"""

    builder = StateGraph(PlanGraphState)
    _add_logged_node(builder, CREATE_PLAN, build_create_plan_node(dependencies))
    _add_logged_node(builder, SELECT_NEXT_STEP, build_select_next_step_node(dependencies))
    _add_logged_node(builder, EXECUTE_STEP, build_execute_step_node(dependencies))
    _add_logged_node(builder, HANDLE_RETRY, build_handle_retry_node(dependencies))
    _add_logged_node(builder, HANDLE_WAITING_USER, build_handle_waiting_user_node(dependencies))
    _add_logged_node(builder, SYNTHESIZE_PLAN_RESULT, build_synthesize_plan_result_node(dependencies))
    _add_logged_node(builder, "synthesize_result", build_synthesize_result_node(dependencies))

    builder.add_edge(START, CREATE_PLAN)
    builder.add_edge(CREATE_PLAN, SELECT_NEXT_STEP)
    builder.add_edge(SELECT_NEXT_STEP, EXECUTE_STEP)
    builder.add_edge(EXECUTE_STEP, HANDLE_RETRY)
    builder.add_conditional_edges(
        HANDLE_RETRY,
        wrap_graph_route(
            graph_name=PLAN_GRAPH_NAME,
            route_name=HANDLE_RETRY,
            route=build_handle_retry_edge(),
        ),
    )
    builder.add_edge(HANDLE_WAITING_USER, "synthesize_result")
    builder.add_edge(SYNTHESIZE_PLAN_RESULT, "synthesize_result")
    builder.add_edge("synthesize_result", END)

    return builder.compile(checkpointer=checkpointer)


def _add_logged_node(builder: StateGraph, node_name: str, node: Any) -> None:
    builder.add_node(
        node_name,
        wrap_graph_node(
            graph_name=PLAN_GRAPH_NAME,
            node_name=node_name,
            node=node,
        ),
    )

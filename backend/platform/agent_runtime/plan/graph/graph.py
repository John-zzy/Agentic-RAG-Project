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
)
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState


def build_plan_graph(
    dependencies: PlanGraphDependencies,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """编排 Plan 子图拓扑。"""

    builder = StateGraph(PlanGraphState)
    builder.add_node(CREATE_PLAN, build_create_plan_node(dependencies))
    builder.add_node(SELECT_NEXT_STEP, build_select_next_step_node(dependencies))
    builder.add_node(EXECUTE_STEP, build_execute_step_node(dependencies))
    builder.add_node(HANDLE_RETRY, build_handle_retry_node(dependencies))
    builder.add_node(HANDLE_WAITING_USER, build_handle_waiting_user_node(dependencies))
    builder.add_node(SYNTHESIZE_PLAN_RESULT, build_synthesize_plan_result_node(dependencies))

    builder.add_edge(START, CREATE_PLAN)
    builder.add_edge(CREATE_PLAN, SELECT_NEXT_STEP)
    builder.add_edge(SELECT_NEXT_STEP, EXECUTE_STEP)
    builder.add_edge(EXECUTE_STEP, HANDLE_RETRY)
    builder.add_conditional_edges(HANDLE_RETRY, build_handle_retry_edge())
    builder.add_edge(HANDLE_WAITING_USER, END)
    builder.add_edge(SYNTHESIZE_PLAN_RESULT, END)

    return builder.compile(checkpointer=checkpointer)

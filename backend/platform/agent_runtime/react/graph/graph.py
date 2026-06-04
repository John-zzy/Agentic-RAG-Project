from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.edges import (
    ASK_USER,
    EXECUTE_TOOL,
    FINAL_ANSWER,
    LOOP_OR_FINISH,
    RESPOND,
    build_loop_or_finish_edge,
    build_route_action_edge,
)
from backend.platform.agent_runtime.react.graph.nodes import (
    build_ask_user_node,
    build_execute_tool_node,
    build_final_answer_node,
    build_loop_or_finish_node,
    build_record_observation_node,
    build_route_action_node,
    build_respond_node,
    build_select_action_node,
    build_validate_action_node,
    build_waiting_user_node,
)
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_react_graph(
    dependencies: ReActGraphDependencies,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """编排 ReAct 子图拓扑。"""

    builder = StateGraph(ReActGraphState)
    builder.add_node("select_action", build_select_action_node(dependencies))
    builder.add_node("validate_action", build_validate_action_node(dependencies))
    builder.add_node("route_action", build_route_action_node(dependencies))
    builder.add_node(RESPOND, build_respond_node(dependencies))
    builder.add_node(EXECUTE_TOOL, build_execute_tool_node(dependencies))
    builder.add_node(ASK_USER, build_ask_user_node(dependencies))
    builder.add_node(FINAL_ANSWER, build_final_answer_node(dependencies))
    builder.add_node("record_observation", build_record_observation_node(dependencies))
    builder.add_node("waiting_user", build_waiting_user_node(dependencies))
    builder.add_node(LOOP_OR_FINISH, build_loop_or_finish_node(dependencies))

    builder.add_edge(START, "select_action")
    builder.add_edge("select_action", "validate_action")
    builder.add_edge("validate_action", "route_action")
    builder.add_conditional_edges("route_action", build_route_action_edge())
    builder.add_edge(RESPOND, "record_observation")
    builder.add_edge(EXECUTE_TOOL, "record_observation")
    builder.add_edge("record_observation", LOOP_OR_FINISH)
    builder.add_edge(ASK_USER, "waiting_user")
    builder.add_edge("waiting_user", END)
    builder.add_edge(FINAL_ANSWER, END)
    builder.add_conditional_edges(LOOP_OR_FINISH, build_loop_or_finish_edge())

    return builder.compile(checkpointer=checkpointer)

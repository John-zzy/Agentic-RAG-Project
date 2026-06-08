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
    build_initialize_run_node,
    build_loop_or_finish_node,
    build_record_observation_node,
    build_route_action_node,
    build_respond_node,
    build_select_action_node,
    build_synthesize_result_node,
    build_validate_action_node,
    build_waiting_user_node,
)
from backend.platform.agent_runtime.react.graph.state import ReActGraphState
from backend.platform.agent_runtime.graph_logging import (
    wrap_graph_node,
    wrap_graph_route,
)
from backend.platform.workflow.langgraph.guards import register_guarded_node

REACT_GRAPH_NAME = "react_graph"
GUARDED_REACT_NODES = {
    "select_action": "runtime",
    EXECUTE_TOOL: "tool",
    "record_observation": "runtime",
    "synthesize_result": "runtime",
}


def build_react_graph(
    dependencies: ReActGraphDependencies,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """编排 ReAct 子图拓扑。"""

    builder = StateGraph(ReActGraphState)
    _add_logged_node(builder, "initialize_run", build_initialize_run_node(dependencies))
    _add_guarded_logged_node(builder, "select_action", build_select_action_node(dependencies))
    _add_logged_node(builder, "validate_action", build_validate_action_node(dependencies))
    _add_logged_node(builder, "route_action", build_route_action_node(dependencies))
    _add_logged_node(builder, RESPOND, build_respond_node(dependencies))
    _add_guarded_logged_node(builder, EXECUTE_TOOL, build_execute_tool_node(dependencies))
    _add_logged_node(builder, ASK_USER, build_ask_user_node(dependencies))
    _add_logged_node(builder, FINAL_ANSWER, build_final_answer_node(dependencies))
    _add_guarded_logged_node(
        builder,
        "record_observation",
        build_record_observation_node(dependencies),
    )
    _add_logged_node(builder, "waiting_user", build_waiting_user_node(dependencies))
    _add_logged_node(builder, LOOP_OR_FINISH, build_loop_or_finish_node(dependencies))
    _add_guarded_logged_node(
        builder,
        "synthesize_result",
        build_synthesize_result_node(dependencies),
    )

    builder.add_edge(START, "initialize_run")
    builder.add_conditional_edges(
        "initialize_run",
        wrap_graph_route(
            graph_name=REACT_GRAPH_NAME,
            route_name="after_initialize",
            route=_route_after_initialize,
        ),
    )
    builder.add_edge("select_action", "validate_action")
    builder.add_edge("validate_action", "route_action")
    builder.add_conditional_edges(
        "route_action",
        wrap_graph_route(
            graph_name=REACT_GRAPH_NAME,
            route_name="route_action",
            route=build_route_action_edge(),
        ),
    )
    builder.add_edge(RESPOND, "record_observation")
    builder.add_edge(EXECUTE_TOOL, "record_observation")
    builder.add_edge("record_observation", LOOP_OR_FINISH)
    builder.add_edge(ASK_USER, "waiting_user")
    builder.add_edge("waiting_user", "synthesize_result")
    builder.add_edge(FINAL_ANSWER, "synthesize_result")
    builder.add_conditional_edges(
        LOOP_OR_FINISH,
        wrap_graph_route(
            graph_name=REACT_GRAPH_NAME,
            route_name=LOOP_OR_FINISH,
            route=build_loop_or_finish_edge(),
        ),
    )
    builder.add_edge("synthesize_result", END)

    return builder.compile(checkpointer=checkpointer)


def _route_after_initialize(state: ReActGraphState) -> str:
    run = state["run"]
    if run.workflow_status in {"waiting_user", "failed", "cancelled", "succeeded"}:
        return "synthesize_result"
    return "select_action"


def _add_logged_node(builder: StateGraph, node_name: str, node: Any) -> None:
    builder.add_node(
        node_name,
        wrap_graph_node(
            graph_name=REACT_GRAPH_NAME,
            node_name=node_name,
            node=node,
        ),
    )


def _add_guarded_logged_node(builder: StateGraph, node_name: str, node: Any) -> None:
    # 保留 observation retry 语义，仅把节点抛出的框架异常交给统一 guard。
    register_guarded_node(
        builder,
        node_name,
        wrap_graph_node(
            graph_name=REACT_GRAPH_NAME,
            node_name=node_name,
            node=node,
        ),
        graph_name=REACT_GRAPH_NAME,
        source=GUARDED_REACT_NODES[node_name],
        metadata={"guard_scope": "react_graph"},
    )

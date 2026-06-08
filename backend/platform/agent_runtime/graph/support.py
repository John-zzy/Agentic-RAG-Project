from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph

from backend.platform.agent_runtime.observability.graph_logging import wrap_graph_node
from backend.platform.workflow.langgraph.guards import register_guarded_node


def add_logged_node(
    builder: StateGraph,
    *,
    graph_name: str,
    node_name: str,
    node: Any,
) -> None:
    builder.add_node(
        node_name,
        wrap_graph_node(
            graph_name=graph_name,
            node_name=node_name,
            node=node,
        ),
    )


def add_guarded_logged_node(
    builder: StateGraph,
    *,
    graph_name: str,
    node_name: str,
    node: Any,
    source: str,
    guard_scope: str,
) -> None:
    register_guarded_node(
        builder,
        node_name,
        wrap_graph_node(
            graph_name=graph_name,
            node_name=node_name,
            node=node,
        ),
        graph_name=graph_name,
        source=source,
        metadata={"guard_scope": guard_scope},
    )

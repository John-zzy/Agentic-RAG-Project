from __future__ import annotations

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_waiting_user_node(dependencies: ReActGraphDependencies):
    """把 waiting_user 作为显式终止节点保留。"""

    del dependencies

    def waiting_user(state: ReActGraphState):
        return {"run": state["run"], "turn": state.get("turn"), "route": "end"}

    return waiting_user

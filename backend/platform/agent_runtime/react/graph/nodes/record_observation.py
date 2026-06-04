from __future__ import annotations

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_record_observation_node(dependencies: ReActGraphDependencies):
    """把 turn 的可观测结果收口成图上的显式节点。"""

    del dependencies

    def record_observation(state: ReActGraphState):
        turn = state.get("turn")
        observation = turn.observation if turn is not None else None
        return {"turn": turn, "observation": observation}

    return record_observation

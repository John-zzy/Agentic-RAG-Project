from __future__ import annotations

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.edges import ASK_USER, EXECUTE_TOOL, FINAL_ANSWER, LOOP_OR_FINISH, RESPOND
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_route_action_node(dependencies: ReActGraphDependencies):
    """把动作分类收口到固定分支名。"""

    del dependencies

    def route_action(state: ReActGraphState) -> dict[str, str]:
        action = state.get("action")
        if action is None:
            return {"route": LOOP_OR_FINISH}
        if action.action_type == "tool_call":
            return {"route": EXECUTE_TOOL}
        if action.action_type == "ask_user":
            return {"route": ASK_USER}
        if action.action_type == "respond":
            return {"route": RESPOND}
        if action.action_type in {"final_answer", "stop"}:
            return {"route": FINAL_ANSWER}
        return {"route": LOOP_OR_FINISH}

    return route_action

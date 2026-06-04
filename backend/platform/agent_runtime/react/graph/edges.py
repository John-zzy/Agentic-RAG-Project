from __future__ import annotations

from langgraph.graph import END

from backend.platform.agent_runtime.react.graph.state import ReActGraphState

RESPOND = "respond"
EXECUTE_TOOL = "execute_tool"
ASK_USER = "ask_user"
FINAL_ANSWER = "final_answer"
LOOP_OR_FINISH = "loop_or_finish"


def build_route_action_edge():
    """只做动作路由，不把分支判断散落到节点里。"""

    def route_action(state: ReActGraphState) -> str:
        route = state.get("route")
        if isinstance(route, str) and route:
            return route
        action = state.get("action")
        if action is None:
            return LOOP_OR_FINISH
        if action.action_type == "tool_call":
            return EXECUTE_TOOL
        if action.action_type == "ask_user":
            return ASK_USER
        if action.action_type == "respond":
            return RESPOND
        if action.action_type in {"final_answer", "stop"}:
            return FINAL_ANSWER
        return LOOP_OR_FINISH

    return route_action


def build_loop_or_finish_edge():
    """执行完一轮后，统一判断是否继续。"""

    def loop_or_finish(state: ReActGraphState) -> str:
        route = state.get("route")
        if isinstance(route, str) and route:
            if route == "end":
                return END
            return route
        run = state["run"]
        if run.workflow_status in {"waiting_user", "failed", "cancelled", "succeeded"}:
            return END
        if len(run.turns) >= run.max_turns:
            return END
        return "select_action"

    return loop_or_finish

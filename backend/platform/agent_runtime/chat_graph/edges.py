from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState

REACT_BRANCH = "react_branch"
PLAN_BRANCH = "plan_branch"
RESOLVE_ANSWER_MODE = "resolve_answer_mode"


def build_route_mode_edge(dependencies: ChatGraphDependencies):
    """根据已准备的 answer_mode 和 agent_mode 选择下一跳。"""

    del dependencies

    def route_mode(state: RuntimeGraphState) -> str:
        answer_mode = str(state.get("answer_mode") or "")
        if state.get("final_decision") and answer_mode in {"direct_answer", "follow_up", "fallback"}:
            return RESOLVE_ANSWER_MODE
        agent_mode = str(state.get("agent_mode") or "react")
        if agent_mode == "plan":
            return PLAN_BRANCH
        return REACT_BRANCH

    return route_mode



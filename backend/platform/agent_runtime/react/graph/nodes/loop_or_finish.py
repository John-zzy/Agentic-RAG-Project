from __future__ import annotations

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_loop_or_finish_node(dependencies: ReActGraphDependencies):
    """判断是否继续下一轮。"""

    del dependencies

    def loop_or_finish(state: ReActGraphState):
        run = state["run"]
        if run.workflow_status in {"waiting_user", "failed", "cancelled", "succeeded"}:
            return {"route": "end"}
        if len(run.turns) >= run.max_turns:
            return {"route": "end"}
        return {"route": "select_action"}

    return loop_or_finish

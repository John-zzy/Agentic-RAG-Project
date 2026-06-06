from __future__ import annotations

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_loop_or_finish_node(dependencies: ReActGraphDependencies):
    """判断是否继续下一轮。"""

    runtime = dependencies.build_runtime()

    def loop_or_finish(state: ReActGraphState):
        run = state["run"]
        if run.workflow_status in {"waiting_user", "failed", "cancelled", "succeeded"}:
            return {"route": "synthesize_result"}
        if len(run.turns) >= run.max_turns:
            runtime.finish_when_budget_exhausted(run)
            return {"run": run, "route": "synthesize_result"}
        return {"route": "select_action"}

    return loop_or_finish

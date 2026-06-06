from __future__ import annotations

from backend.platform.agent_runtime.contracts import ReActRun
from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.state import ReActGraphState
from backend.platform.agent_runtime.react.state import transition


def build_initialize_run_node(dependencies: ReActGraphDependencies):
    """初始化或复用 ReActRun，让子图拥有 run lifecycle 起点。"""

    def initialize_run(state: ReActGraphState):
        run = state.get("run") or dependencies.initial_run
        if run is None:
            run = ReActRun(
                react_run_id=dependencies.react_run_id
                or f"react-{dependencies.request_id}",
                session_id=dependencies.session_id,
                request_id=dependencies.request_id,
                user_goal=dependencies.user_goal,
                max_turns=dependencies.max_turns,
                workflow_status="created",
            )
        if run.workflow_status == "created":
            transition(run, "run_start")
        return {"run": run}

    return initialize_run

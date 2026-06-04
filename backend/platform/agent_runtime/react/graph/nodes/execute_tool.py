from __future__ import annotations

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_execute_tool_node(dependencies: ReActGraphDependencies):
    """执行工具型 turn。"""

    runtime = dependencies.build_runtime()

    def execute_tool(state: ReActGraphState):
        run = state["run"]
        action = state.get("action")
        if action is None:
            return {}
        turn = runtime.append_turn(
            run=run,
            action=action,
            round_index=len(run.turns) + 1,
        )
        runtime.execute_tool_turn(run=run, turn=turn)
        return {"run": run, "turn": turn}

    return execute_tool

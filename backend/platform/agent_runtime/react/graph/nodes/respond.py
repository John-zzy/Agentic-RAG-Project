from __future__ import annotations

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_respond_node(dependencies: ReActGraphDependencies):
    """把 stop 路径收敛成直接回答。"""

    runtime = dependencies.build_runtime()

    def respond(state: ReActGraphState):
        run = state["run"]
        action = state.get("action")
        if action is None:
            return {}
        turn = runtime.append_turn(
            run=run,
            action=action,
            round_index=len(run.turns) + 1,
        )
        turn.status = "succeeded"
        runtime.synthesize_success(run=run, final_turn=turn)
        return {"run": run, "turn": turn}

    return respond

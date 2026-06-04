from __future__ import annotations

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_ask_user_node(dependencies: ReActGraphDependencies):
    """把 ask_user 分支写成等待态。"""

    runtime = dependencies.build_runtime()

    def ask_user(state: ReActGraphState):
        run = state["run"]
        action = state.get("action")
        if action is None:
            return {}
        turn = runtime.append_turn(
            run=run,
            action=action,
            round_index=len(run.turns) + 1,
        )
        runtime.mark_waiting_user(
            run=run,
            turn=turn,
            user_prompt=action.instruction or "Please provide more information.",
            source="react_action",
        )
        return {"run": run, "turn": turn}

    return ask_user

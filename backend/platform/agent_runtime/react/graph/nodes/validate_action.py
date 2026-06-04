from __future__ import annotations

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_validate_action_node(dependencies: ReActGraphDependencies):
    """对 selector 输出做一次显式校验。"""

    runtime = dependencies.build_runtime()

    def validate_action(state: ReActGraphState):
        action = state.get("action")
        if action is None:
            return {}
        run = state["run"]
        validated = runtime.validate_action(
            action=action,
            run=run,
            round_index=len(run.turns) + 1,
        )
        return {"action": validated}

    return validate_action

from __future__ import annotations

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_select_action_node(dependencies: ReActGraphDependencies):
    """选择下一步 ReAct 动作。"""

    runtime = dependencies.build_runtime()

    def select_action(state: ReActGraphState):
        run = state["run"]
        action = runtime.select_action(run=run, round_index=len(run.turns) + 1)
        return {"action": action}

    return select_action

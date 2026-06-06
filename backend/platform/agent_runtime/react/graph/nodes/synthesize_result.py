from __future__ import annotations

from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.react.graph.state import ReActGraphState


def build_synthesize_result_node(dependencies: ReActGraphDependencies):
    """把 ReActRun 投影为 ChatGraph 可消费的结果字段。"""

    def synthesize_result(state: ReActGraphState):
        run = state["run"]
        projected = (
            dict(dependencies.project_result(run))
            if dependencies.project_result is not None
            else {}
        )
        return {
            "run": run,
            "current_turn_id": run.current_turn_id,
            "current_tool_call": (
                run.current_tool_call.model_dump()
                if run.current_tool_call is not None
                else None
            ),
            **projected,
        }

    return synthesize_result

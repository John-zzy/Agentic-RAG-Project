from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState


def build_synthesize_result_node(dependencies: PlanGraphDependencies):
    """把 PlanRun 投影为 ChatGraph 可消费的结果字段。"""

    def synthesize_result(state: PlanGraphState):
        plan_run = state["plan_run"]
        projected = (
            dict(dependencies.project_result(plan_run))
            if dependencies.project_result is not None
            else {}
        )
        return {
            "plan_run": plan_run,
            "status": plan_run.workflow_status,
            "current_step_id": plan_run.current_step_id,
            "current_tool_call": (
                plan_run.current_tool_call.model_dump()
                if plan_run.current_tool_call is not None
                else None
            ),
            **projected,
        }

    return synthesize_result

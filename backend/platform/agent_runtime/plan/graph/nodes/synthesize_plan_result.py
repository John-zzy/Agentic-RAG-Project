from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState


def build_synthesize_plan_result_node(dependencies: PlanGraphDependencies):
    """保留最终汇总节点。"""

    executor = dependencies.build_executor()

    def synthesize_plan_result(state: PlanGraphState):
        plan_run = state["plan_run"]
        return {"plan_run": executor.synthesize_plan_result(plan_run)}

    return synthesize_plan_result

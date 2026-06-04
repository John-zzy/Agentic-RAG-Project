from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState


def build_create_plan_node(dependencies: PlanGraphDependencies):
    """保留显式计划创建边界。"""

    del dependencies

    def create_plan(state: PlanGraphState):
        return {"plan_run": state["plan_run"]}

    return create_plan

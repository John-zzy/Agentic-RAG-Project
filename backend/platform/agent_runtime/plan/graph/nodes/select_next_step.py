from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState


def build_select_next_step_node(dependencies: PlanGraphDependencies):
    """标记下一步可执行步骤。"""

    executor = dependencies.build_executor()

    def select_next_step(state: PlanGraphState):
        plan_run = state["plan_run"]
        next_step = executor.select_next_step(plan_run.steps)
        return {"step": next_step}

    return select_next_step

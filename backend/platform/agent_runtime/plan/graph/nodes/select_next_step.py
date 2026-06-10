from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState
from backend.platform.agent_runtime.plan.state_ops import (
    all_steps_succeeded,
    has_blocked_steps,
    mark_failed,
    mark_steps_blocked_by_unavailable_dependencies,
    select_next_step as select_plan_next_step,
)


def build_select_next_step_node(dependencies: PlanGraphDependencies):
    """标记下一步可执行步骤。"""

    del dependencies

    def select_next_step(state: PlanGraphState):
        plan_run = state["plan_run"]
        mark_steps_blocked_by_unavailable_dependencies(plan_run.steps)
        next_step = select_plan_next_step(plan_run.steps)
        if next_step is None and not all_steps_succeeded(plan_run.steps):
            error = (
                "Plan has blocked steps because required dependencies did not complete."
                if has_blocked_steps(plan_run.steps)
                else "Plan has pending steps but no executable dependency order."
            )
            mark_failed(
                plan_run=plan_run,
                error=error,
            )
        return {"plan_run": plan_run, "step": next_step}

    return select_next_step

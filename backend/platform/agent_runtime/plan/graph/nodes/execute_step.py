from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState
from backend.platform.agent_runtime.plan.state_ops import (
    execute_step_once,
    select_next_step as select_plan_next_step,
)


def build_execute_step_node(dependencies: PlanGraphDependencies):
    """执行单个 step，让后续分支由 handle_retry 节点判断。"""

    def execute_step(state: PlanGraphState):
        plan_run = state["plan_run"]
        step = state.get("step")
        if step is None:
            step = select_plan_next_step(plan_run.steps)
            if step is None:
                return {"plan_run": plan_run}
        execute_step_once(
            plan_run=plan_run,
            step=step,
            tool_executor=dependencies.tool_executor,
        )
        return {"plan_run": plan_run, "step": step}

    return execute_step

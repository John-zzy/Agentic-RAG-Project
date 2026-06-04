from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState


def build_execute_step_node(dependencies: PlanGraphDependencies):
    """执行单个 step，让后续分支由 handle_retry 节点判断。"""

    executor = dependencies.build_executor()

    def execute_step(state: PlanGraphState):
        plan_run = state["plan_run"]
        step = state.get("step")
        if step is None:
            step = executor.select_next_step(plan_run.steps)
            if step is None:
                return {"plan_run": plan_run}
        executor.execute_step_once(plan_run=plan_run, step=step)
        return {"plan_run": plan_run, "step": step}

    return execute_step

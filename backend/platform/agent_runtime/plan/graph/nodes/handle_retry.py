from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState


def build_handle_retry_node(dependencies: PlanGraphDependencies):
    """执行步骤后的分支收口。"""

    executor = dependencies.build_executor()

    def handle_retry(state: PlanGraphState):
        plan_run = state["plan_run"]
        step = state.get("step")
        if step is None:
            if all(item.status == "succeeded" for item in plan_run.steps) and plan_run.steps:
                return {"plan_run": plan_run, "route": "synthesize_plan_result"}
            return {"plan_run": plan_run, "route": "select_next_step"}

        executor.handle_retry(plan_run=plan_run, step=step)
        if step.status == "waiting_user":
            return {"plan_run": plan_run, "step": step, "route": "handle_waiting_user"}
        if step.status in {"failed", "cancelled"} or plan_run.workflow_status in {"failed", "cancelled"}:
            return {"plan_run": plan_run, "step": step, "route": "synthesize_plan_result"}
        if step.status == "pending":
            return {"plan_run": plan_run, "step": step, "route": "select_next_step"}
        if all(item.status == "succeeded" for item in plan_run.steps) and plan_run.steps:
            return {"plan_run": plan_run, "step": step, "route": "synthesize_plan_result"}
        if any(item.status == "pending" for item in plan_run.steps):
            return {"plan_run": plan_run, "step": step, "route": "select_next_step"}
        return {"plan_run": plan_run, "step": step, "route": "synthesize_plan_result"}

    return handle_retry

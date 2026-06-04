from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState


def build_handle_waiting_user_node(dependencies: PlanGraphDependencies):
    """保留 waiting_user 节点，便于后续展开 HITL 恢复。"""

    del dependencies

    def handle_waiting_user(state: PlanGraphState):
        plan_run = state["plan_run"]
        step = state.get("step")
        if step is None:
            return {"plan_run": plan_run}
        return {"plan_run": plan_run, "step": step}

    return handle_waiting_user

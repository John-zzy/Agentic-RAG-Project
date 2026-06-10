from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState
from backend.platform.agent_runtime.plan.state_ops import synthesize_plan_result as synthesize_run


def build_synthesize_plan_result_node(dependencies: PlanGraphDependencies):
    """保留最终汇总节点。"""

    def synthesize_plan_result(state: PlanGraphState):
        plan_run = state["plan_run"]
        if plan_run.workflow_status in {"waiting_user", "failed", "cancelled"}:
            return {"plan_run": plan_run}
        return {
            "plan_run": synthesize_run(
                plan_run=plan_run,
                final_synthesizer=dependencies.resolved_final_synthesizer(),
                model_call_guard=dependencies.model_call_guard,
            )
        }

    return synthesize_plan_result

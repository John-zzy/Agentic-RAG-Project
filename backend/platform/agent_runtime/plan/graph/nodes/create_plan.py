from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState


def build_create_plan_node(dependencies: PlanGraphDependencies):
    """保留显式计划创建边界。"""

    planner = dependencies.build_planner()

    def create_plan(state: PlanGraphState):
        plan_run = state.get("plan_run")
        if plan_run is None:
            plan_run = planner.create_plan(
                session_id=dependencies.session_id,
                request_id=dependencies.request_id,
                user_goal=dependencies.user_goal,
                mounted_knowledge_sources=dependencies.mounted_knowledge_sources,
                candidate_tools=dependencies.candidate_tools,
                scene_policy=dependencies.scene_policy,
                plan_run_id=f"plan-{dependencies.request_id}",
            )
        return {"plan_run": plan_run}

    return create_plan

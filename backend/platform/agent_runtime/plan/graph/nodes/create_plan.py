from __future__ import annotations

from backend.platform.agent_runtime.core.contracts import PlanRun
from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.state import PlanGraphState
from backend.platform.agent_runtime.plan.state_ops import mark_failed


def build_create_plan_node(dependencies: PlanGraphDependencies):
    """保留显式计划创建边界。"""

    def create_plan(state: PlanGraphState):
        plan_run = state.get("plan_run")
        if plan_run is None:
            try:
                if dependencies.planner is None:
                    raise ValueError("Plan planner is required to create a new PlanRun.")
                plan_run = dependencies.planner.create_plan(
                    session_id=dependencies.session_id,
                    request_id=dependencies.request_id,
                    user_goal=dependencies.user_goal,
                    mounted_knowledge_sources=dependencies.mounted_knowledge_sources,
                    candidate_tools=dependencies.candidate_tools,
                    scene_policy=dependencies.scene_policy,
                    default_tool_inputs=dependencies.default_tool_inputs,
                    complexity=dependencies.complexity,
                    max_plan_steps=dependencies.max_plan_steps,
                    plan_run_id=f"plan-{dependencies.request_id}",
                )
            except Exception as exc:
                plan_run = PlanRun(
                    plan_run_id=f"plan-{dependencies.request_id}",
                    session_id=dependencies.session_id,
                    request_id=dependencies.request_id,
                    user_goal=dependencies.user_goal,
                    workflow_status="running",
                    metadata={
                        "planner": {
                            "name": "langchain_plan_planner",
                            "step_source": "llm_structured_output",
                            "max_plan_steps": dependencies.max_plan_steps,
                            "error": str(exc),
                        }
                    },
                )
                mark_failed(plan_run=plan_run, error=str(exc))
        return {"plan_run": plan_run}

    return create_plan

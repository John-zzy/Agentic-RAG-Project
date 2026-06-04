from __future__ import annotations

from backend.platform.agent_runtime.plan.graph.state import PlanGraphState

CREATE_PLAN = "create_plan"
SELECT_NEXT_STEP = "select_next_step"
EXECUTE_STEP = "execute_step"
HANDLE_RETRY = "handle_retry"
HANDLE_WAITING_USER = "handle_waiting_user"
SYNTHESIZE_PLAN_RESULT = "synthesize_plan_result"


def build_handle_retry_edge():
    """执行步骤后，统一判断重试、等待和收口。"""

    def handle_retry(state: PlanGraphState) -> str:
        route = state.get("route")
        if route == HANDLE_WAITING_USER:
            return HANDLE_WAITING_USER
        if route == SELECT_NEXT_STEP:
            return SELECT_NEXT_STEP
        if route == SYNTHESIZE_PLAN_RESULT:
            return SYNTHESIZE_PLAN_RESULT
        return SYNTHESIZE_PLAN_RESULT

    return handle_retry

from __future__ import annotations

from typing import NotRequired, TypedDict

from backend.platform.agent_runtime.contracts import PlanRun, PlanStep


class PlanGraphState(TypedDict, total=False):
    """Plan 图在单次执行中交换的最小状态。"""

    plan_run: PlanRun
    step: NotRequired[PlanStep | None]
    error: NotRequired[str | None]
    route: NotRequired[str | None]

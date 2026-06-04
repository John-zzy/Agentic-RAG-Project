from __future__ import annotations

from typing import NotRequired, TypedDict

from backend.platform.agent_runtime.contracts import ReActAction, ReActRun, ReActTurn, ToolObservation


class ReActGraphState(TypedDict, total=False):
    """ReAct 图在单次执行中交换的最小状态。"""

    run: ReActRun
    action: NotRequired[ReActAction | None]
    turn: NotRequired[ReActTurn | None]
    observation: NotRequired[ToolObservation | None]
    error: NotRequired[str | None]
    route: NotRequired[str | None]

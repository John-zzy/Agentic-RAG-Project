from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from backend.platform.agent_runtime.contracts import ReActAction, ReActRun, ReActTurn, ToolObservation


class ReActGraphState(TypedDict, total=False):
    """ReAct 图在单次执行中交换的最小状态。"""

    run: ReActRun
    action: NotRequired[ReActAction | None]
    turn: NotRequired[ReActTurn | None]
    observation: NotRequired[ToolObservation | None]
    error: NotRequired[str | None]
    route: NotRequired[str | None]
    documents: NotRequired[list[Any]]
    citations: NotRequired[list[dict[str, Any]]]
    retrieval_trace: NotRequired[dict[str, Any]]
    final_decision: NotRequired[str | None]
    answer_mode: NotRequired[str | None]
    knowledge_used: NotRequired[bool]
    follow_up_question: NotRequired[str | None]
    tool_event: NotRequired[dict[str, Any] | None]
    current_turn_id: NotRequired[str | None]
    current_tool_call: NotRequired[dict[str, Any] | None]
    tool_observation: NotRequired[dict[str, Any] | None]

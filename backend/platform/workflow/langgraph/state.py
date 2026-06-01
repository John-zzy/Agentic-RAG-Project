from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


RuntimeGraphStatus = Literal["running", "waiting_user", "succeeded", "failed"]


class RuntimeHitlSuggestedResponse(TypedDict, total=False):
    """用户需要补充信息时，后端给前端展示的一个建议选项。"""

    suggestion_id: str
    label: str
    value: str
    description: str
    metadata: dict[str, Any]


class RuntimeHitlResumePayload(TypedDict, total=False):
    """用户处理等待任务时提交的内容，比如拒绝原因或补充回答。"""

    action: str
    reason: str | None
    edited_args: dict[str, Any] | None
    response: str | None
    source: str | None
    suggestion_id: str | None
    metadata: dict[str, Any]


class RuntimeHitlState(TypedDict, total=False):
    """会话暂停等待用户时，需要保存到 checkpoint 里的信息。"""

    interrupt_id: str
    thread_id: str
    reason: str
    pending_action: str
    proposed_tool_call: dict[str, Any] | None
    allowed_actions: list[str]
    suggested_responses: list[RuntimeHitlSuggestedResponse]
    allow_freeform_response: bool
    resume_payload: RuntimeHitlResumePayload | None


class RuntimeGraphState(TypedDict):
    """LangGraph 每轮聊天要保存的状态。"""

    session_id: str
    request_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    answer: str
    knowledge_used: bool
    citations: list[dict[str, Any]]
    retrieval_trace: dict[str, Any]
    metadata: dict[str, Any]
    status: NotRequired[RuntimeGraphStatus]
    hitl: NotRequired[RuntimeHitlState | None]
    hitl_resume: NotRequired[RuntimeHitlResumePayload | None]


def build_runtime_hitl_state(
    *,
    interrupt_id: str,
    thread_id: str,
    reason: str,
    pending_action: str,
    allowed_actions: Sequence[str],
    proposed_tool_call: Mapping[str, Any] | None = None,
    suggested_responses: Sequence[Mapping[str, Any]] | None = None,
    allow_freeform_response: bool = False,
    resume_payload: Mapping[str, Any] | None = None,
) -> RuntimeHitlState:
    """创建等待用户处理的状态，保证各入口使用同一组字段。"""
    if not interrupt_id:
        raise ValueError("interrupt_id is required for HITL state.")
    if not thread_id:
        raise ValueError("thread_id is required for HITL state.")
    if not pending_action:
        raise ValueError("pending_action is required for HITL state.")

    return {
        "interrupt_id": interrupt_id,
        "thread_id": thread_id,
        "reason": reason,
        "pending_action": pending_action,
        "proposed_tool_call": dict(proposed_tool_call) if proposed_tool_call else None,
        "allowed_actions": list(allowed_actions),
        "suggested_responses": [
            dict(suggested_response)
            for suggested_response in suggested_responses or ()
        ],
        "allow_freeform_response": allow_freeform_response,
        "resume_payload": dict(resume_payload) if resume_payload else None,
    }


def build_runtime_graph_state(
    *,
    session_id: str,
    request_id: str,
    messages: Sequence[AnyMessage] | None = None,
    answer: str = "",
    knowledge_used: bool = False,
    citations: Sequence[Mapping[str, Any]] | None = None,
    retrieval_trace: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    status: RuntimeGraphStatus = "running",
    hitl: Mapping[str, Any] | None = None,
    hitl_resume: Mapping[str, Any] | None = None,
) -> RuntimeGraphState:
    """创建一份可以保存到 checkpoint 的 graph 状态。"""
    if not session_id:
        raise ValueError("session_id is required for runtime graph state.")
    if not request_id:
        raise ValueError("request_id is required for runtime graph state.")

    return {
        "session_id": session_id,
        "request_id": request_id,
        "messages": list(messages or ()),
        "answer": answer,
        "knowledge_used": knowledge_used,
        "citations": [dict(citation) for citation in citations or ()],
        "retrieval_trace": dict(retrieval_trace or {}),
        "metadata": dict(metadata or {}),
        "status": status,
        "hitl": dict(hitl) if hitl else None,
        "hitl_resume": dict(hitl_resume) if hitl_resume else None,
    }

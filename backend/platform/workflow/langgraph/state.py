from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class RuntimeGraphState(TypedDict):
    """聊天运行时进入 LangGraph 的最小 state 契约。"""

    session_id: str
    request_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    answer: str
    knowledge_used: bool
    citations: list[dict[str, Any]]
    retrieval_trace: dict[str, Any]
    metadata: dict[str, Any]


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
) -> RuntimeGraphState:
    """构造可序列化的最小 graph state，避免 application 层直接拼散字段。"""
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
    }

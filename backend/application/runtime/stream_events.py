from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal


BusinessStreamEventName = Literal[
    "start",
    "thinking",
    "chunk",
    "waiting_user",
    "resume",
    "done",
    "error",
]
GraphStreamEventName = Literal[
    "graph_run_created",
    "history_snapshot",
    "retrieval_tool_result",
    "answer_chunk",
    "human_waiting",
    "human_resume",
    "graph_run_succeeded",
    "graph_run_failed",
]


@dataclass(frozen=True)
class ChatStreamEvent:
    """描述一条待编码为 SSE 的聊天业务事件。"""

    event: BusinessStreamEventName
    data: dict[str, Any]


@dataclass(frozen=True)
class GraphRuntimeStreamEvent:
    """描述 graph runtime 内部流事件，不直接暴露给 API 客户端。"""

    event: GraphStreamEventName
    data: dict[str, Any]


class GraphStreamEventMapper:
    """将 graph runtime 内部事件映射为面向界面展示的 SSE 业务协议。"""

    _EVENT_MAP: dict[GraphStreamEventName, BusinessStreamEventName] = {
        "graph_run_created": "start",
        "history_snapshot": "thinking",
        "retrieval_tool_result": "thinking",
        "answer_chunk": "chunk",
        "human_waiting": "waiting_user",
        "human_resume": "resume",
        "graph_run_succeeded": "done",
        "graph_run_failed": "error",
    }

    def map_event(self, event: GraphRuntimeStreamEvent) -> ChatStreamEvent:
        try:
            business_event = self._EVENT_MAP[event.event]
        except KeyError as exc:
            raise ValueError(f"Unsupported graph stream event: {event.event}") from exc
        # SSE 只输出业务事件名，payload 中也不透出 graph 原始事件名。
        if business_event == "thinking":
            return ChatStreamEvent(
                event=business_event,
                data=_safe_thinking_payload(event),
            )
        return ChatStreamEvent(event=business_event, data=dict(event.data))

    def map_events(
        self,
        events: Iterable[GraphRuntimeStreamEvent],
    ) -> Iterator[ChatStreamEvent]:
        for event in events:
            yield self.map_event(event)


def _safe_thinking_payload(event: GraphRuntimeStreamEvent) -> dict[str, Any]:
    """审计型 graph 事件只转成可展示状态，不把历史或工具细节流给界面。"""
    payload = dict(event.data)
    safe: dict[str, Any] = {
        "state": str(payload.get("state") or "running"),
    }
    if payload.get("session_id") is not None:
        safe["session_id"] = payload["session_id"]
    if payload.get("request_id") is not None:
        safe["request_id"] = payload["request_id"]
    if event.event == "history_snapshot":
        safe["message"] = "正在整理对话上下文。"
        safe["stage"] = "history"
    else:
        safe["message"] = "正在整理检索结果。"
        safe["stage"] = "tool"
    return safe


from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal


BusinessStreamEventName = Literal["start", "history", "tool", "chunk", "done", "error"]
GraphStreamEventName = Literal[
    "graph_run_created",
    "history_snapshot",
    "retrieval_tool_result",
    "answer_chunk",
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
    """将 graph runtime 内部事件映射为现有 SSE 业务协议。"""

    _EVENT_MAP: dict[GraphStreamEventName, BusinessStreamEventName] = {
        "graph_run_created": "start",
        "history_snapshot": "history",
        "retrieval_tool_result": "tool",
        "answer_chunk": "chunk",
        "graph_run_succeeded": "done",
        "graph_run_failed": "error",
    }

    def map_event(self, event: GraphRuntimeStreamEvent) -> ChatStreamEvent:
        try:
            business_event = self._EVENT_MAP[event.event]
        except KeyError as exc:
            raise ValueError(f"Unsupported graph stream event: {event.event}") from exc
        # SSE 只输出业务事件名，payload 中也不透出 graph 原始事件名。
        return ChatStreamEvent(event=business_event, data=dict(event.data))

    def map_events(
        self,
        events: Iterable[GraphRuntimeStreamEvent],
    ) -> Iterator[ChatStreamEvent]:
        for event in events:
            yield self.map_event(event)

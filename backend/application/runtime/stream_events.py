from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
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
TypedGraphStreamEventName = Literal[
    "graph_start",
    "model_chunk",
    "safe_thinking",
    "interrupt",
    "resume",
    "graph_output",
    "graph_error",
]


@dataclass(frozen=True)
class ChatStreamEvent:
    """描述一条待编码为 SSE 的聊天业务事件。"""

    event: BusinessStreamEventName
    data: dict[str, Any]


@dataclass(frozen=True)
class TypedGraphStreamEvent:
    """LangGraph typed stream part 的项目内表示。"""

    event: TypedGraphStreamEventName
    data: dict[str, Any]


class GraphStreamEventMapper:
    """将 graph runtime 内部事件映射为面向界面展示的 SSE 业务协议。"""

    _TYPED_EVENT_MAP: dict[TypedGraphStreamEventName, BusinessStreamEventName] = {
        "graph_start": "start",
        "model_chunk": "chunk",
        "safe_thinking": "thinking",
        "interrupt": "waiting_user",
        "resume": "resume",
        "graph_output": "done",
        "graph_error": "error",
    }

    def map_event(self, event: TypedGraphStreamEvent) -> ChatStreamEvent:
        return self._map_typed_event(event)

    def map_events(
        self,
        events: Iterable[TypedGraphStreamEvent],
    ) -> Iterator[ChatStreamEvent]:
        for event in events:
            yield self.map_event(event)

    def map_typed_stream_part(self, part: Any) -> ChatStreamEvent | None:
        """把 LangGraph stream_mode 输出片段转成 UI 事件；未知片段静默丢弃。"""
        event = _typed_event_from_stream_part(part)
        if event is None:
            return None
        return self._map_typed_event(event)

    def _map_typed_event(self, event: TypedGraphStreamEvent) -> ChatStreamEvent:
        try:
            business_event = self._TYPED_EVENT_MAP[event.event]
        except KeyError as exc:
            raise ValueError(f"Unsupported typed graph stream event: {event.event}") from exc
        if business_event == "thinking":
            return ChatStreamEvent(event=business_event, data=_safe_typed_thinking_payload(event))
        return ChatStreamEvent(event=business_event, data=_safe_typed_payload(event))


def _typed_event_from_stream_part(part: Any) -> TypedGraphStreamEvent | None:
    if isinstance(part, TypedGraphStreamEvent):
        return part
    if not isinstance(part, tuple) or not part:
        return None
    mode = str(part[0])
    payload = part[1] if len(part) > 1 else {}
    if mode == "updates":
        return _typed_event_from_update(payload)
    if mode == "messages":
        return _typed_event_from_message(payload)
    if mode == "custom" and isinstance(payload, Mapping):
        event = payload.get("event")
        if isinstance(event, str) and event in GraphStreamEventMapper._TYPED_EVENT_MAP:
            return TypedGraphStreamEvent(event, dict(payload.get("data") or {}))  # type: ignore[arg-type]
    return None


def _typed_event_from_update(payload: Any) -> TypedGraphStreamEvent | None:
    if not isinstance(payload, Mapping):
        return None
    if "__interrupt__" in payload:
        return TypedGraphStreamEvent("interrupt", _interrupt_payload(payload["__interrupt__"]))
    for value in payload.values():
        if isinstance(value, Mapping) and value.get("status") == "waiting_user":
            return TypedGraphStreamEvent(
                "interrupt",
                {"hitl": dict(value.get("hitl") or {}), "workflow_state": "waiting_user"},
            )
    return TypedGraphStreamEvent("safe_thinking", {"stage": "graph", "state": "running"})


def _typed_event_from_message(payload: Any) -> TypedGraphStreamEvent | None:
    message = payload[0] if isinstance(payload, tuple) and payload else payload
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return TypedGraphStreamEvent("model_chunk", {"delta": content})
    return None


def _interrupt_payload(value: Any) -> dict[str, Any]:
    item = value[0] if isinstance(value, list) and value else value
    payload = item.get("value") if isinstance(item, Mapping) else getattr(item, "value", item)
    hitl = payload.get("hitl") if isinstance(payload, Mapping) else None
    return {"hitl": dict(hitl or {}), "workflow_state": "waiting_user"}


def _safe_typed_payload(event: TypedGraphStreamEvent) -> dict[str, Any]:
    return _sanitize_stream_value(event.data)


def _safe_typed_thinking_payload(event: TypedGraphStreamEvent) -> dict[str, Any]:
    data = _safe_typed_payload(event)
    return {
        "state": str(data.get("state") or "running"),
        "stage": str(data.get("stage") or "graph"),
        "message": str(data.get("message") or "正在处理图运行状态。"),
    }


_BLOCKED_STREAM_KEY_PARTS = (
    "api_key",
    "arguments",
    "authorization",
    "checkpoint",
    "chain_of_thought",
    "full_history",
    "hidden_cot",
    "history",
    "input_payload",
    "messages",
    "password",
    "prompt",
    "raw",
    "secret",
    "tool_args",
)

_SAFE_OBSERVABILITY_KEYS = {
    "complexity",
    "error_classification",
    "model_latency_ms",
    "provider",
    "retry_count",
    "tools",
}

_SAFE_TOOL_OBSERVABILITY_KEYS = {"tool_name", "tool_status", "status"}


def _sanitize_stream_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_blocked_stream_key(key_text):
                continue
            if key_text == "observability":
                sanitized[key_text] = _sanitize_observability(item)
                continue
            sanitized[key_text] = _sanitize_stream_value(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_stream_value(item) for item in value]
    return value


def _sanitize_observability(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if key_text not in _SAFE_OBSERVABILITY_KEYS:
            continue
        if key_text == "tools":
            sanitized[key_text] = _sanitize_tool_observability_list(item)
        else:
            sanitized[key_text] = _sanitize_stream_value(item)
    return sanitized


def _sanitize_tool_observability_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    tools: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        tool_payload = {
            str(key): _sanitize_stream_value(tool_value)
            for key, tool_value in item.items()
            if str(key) in _SAFE_TOOL_OBSERVABILITY_KEYS
        }
        if tool_payload:
            tools.append(tool_payload)
    return tools


def _is_blocked_stream_key(key: str) -> bool:
    lowered = key.lower()
    if lowered == "raw_candidates_count":
        return False
    return any(part in lowered for part in _BLOCKED_STREAM_KEY_PARTS)


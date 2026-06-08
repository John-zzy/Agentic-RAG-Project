from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GuardMetadata:
    """节点 guard 的序列化上下文，写入 LangGraph node metadata。"""

    graph_name: str
    node_name: str
    source: str = "runtime"
    request_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    step_id: str | None = None
    tool_name: str | None = None
    metadata: Mapping[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "guard": {
                "graph_name": self.graph_name,
                "node_name": self.node_name,
                "source": self.source,
            }
        }
        guard = payload["guard"]
        for key, value in (
            ("request_id", self.request_id),
            ("run_id", self.run_id),
            ("session_id", self.session_id),
            ("turn_id", self.turn_id),
            ("step_id", self.step_id),
            ("tool_name", self.tool_name),
        ):
            if value:
                guard[key] = value
        if self.metadata:
            guard["metadata"] = dict(self.metadata)
        return payload


def build_guard_metadata(
    *,
    graph_name: str,
    node_name: str,
    source: str = "runtime",
    request_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    step_id: str | None = None,
    tool_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not graph_name:
        raise ValueError("graph_name is required for guard metadata.")
    if not node_name:
        raise ValueError("node_name is required for guard metadata.")
    return GuardMetadata(
        graph_name=graph_name,
        node_name=node_name,
        source=source,
        request_id=request_id,
        run_id=run_id,
        session_id=session_id,
        turn_id=turn_id,
        step_id=step_id,
        tool_name=tool_name,
        metadata=metadata,
    ).to_payload()

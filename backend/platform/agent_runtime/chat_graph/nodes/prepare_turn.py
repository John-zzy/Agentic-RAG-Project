from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_prepare_turn_node(dependencies: ChatGraphDependencies):
    """把当前 turn 的基础字段写入 graph state。"""

    prepared = dependencies.prepared

    def prepare_turn(state: RuntimeGraphState) -> dict[str, Any]:
        metadata = dict(state.get("metadata") or {})
        return {
            "scene": prepared.scene_metadata.scene,
            "answer_mode": prepared.answer_mode,
            "agent_mode": prepared.agent_mode,
            "agent_mode_reason": getattr(prepared, "agent_mode_reason", None),
            "agent_mode_signals": dict(getattr(prepared, "agent_mode_signals", None) or {}),
            "knowledge_used": prepared.knowledge_used,
            "citations": [citation.model_dump() for citation in prepared.citations],
            "retrieval_trace": prepared.retrieval_trace.model_dump(),
            "react_run": getattr(prepared, "react_run", None),
            "plan_run": getattr(prepared, "plan_run", None),
            "current_turn_id": getattr(prepared, "current_turn_id", None),
            "current_step_id": getattr(prepared, "current_step_id", None),
            "current_tool_call": getattr(prepared, "current_tool_call", None),
            "metadata": {
                **metadata,
                "scene": prepared.scene_metadata.scene,
                "agent": prepared.scene_metadata.agent,
            },
        }

    return prepare_turn


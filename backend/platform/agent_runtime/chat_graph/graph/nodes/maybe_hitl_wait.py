from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_maybe_hitl_wait_node(dependencies: ChatGraphDependencies):
    """在最终合成前把需要人工补充的分支转为 waiting_user。"""

    build_hitl_wait_update = dependencies.build_hitl_wait_update
    prepared = dependencies.prepared

    def maybe_hitl_wait(state: RuntimeGraphState) -> dict[str, Any]:
        if build_hitl_wait_update is None:
            return _hitl_decision_update(state=state, enabled=False)
        update = dict(build_hitl_wait_update(prepared, state) or {})
        if update.get("status") == "waiting_user":
            return _merge_hitl_decision(update=update, enabled=True)
        return _hitl_decision_update(state=state, enabled=False)

    return maybe_hitl_wait


def _merge_hitl_decision(*, update: dict[str, Any], enabled: bool) -> dict[str, Any]:
    metadata = dict(update.get("metadata") or {})
    metadata["hitl_wait_enabled"] = enabled
    return {**update, "metadata": metadata}


def _hitl_decision_update(*, state: RuntimeGraphState, enabled: bool) -> dict[str, Any]:
    metadata = dict(state.get("metadata") or {})
    metadata["hitl_wait_enabled"] = enabled
    return {"metadata": metadata}


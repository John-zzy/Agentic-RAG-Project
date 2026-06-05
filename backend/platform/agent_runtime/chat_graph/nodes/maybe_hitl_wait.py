from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.config import ChatGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_maybe_hitl_wait_node(dependencies: ChatGraphDependencies):
    """在最终合成前把需要人工补充的分支转为 waiting_user。"""

    build_hitl_wait_update = dependencies.build_hitl_wait_update
    prepared = dependencies.prepared

    def maybe_hitl_wait(state: RuntimeGraphState) -> dict[str, Any]:
        if build_hitl_wait_update is None:
            return {}
        return dict(build_hitl_wait_update(prepared, state) or {})

    return maybe_hitl_wait


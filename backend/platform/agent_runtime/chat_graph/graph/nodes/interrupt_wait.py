from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from backend.platform.workflow.langgraph.state import RuntimeGraphState
from backend.platform.workflow.langgraph.resume import normalize_interrupt_resume_value


def build_interrupt_wait_node():
    """在 LangGraph 内建立真实 interrupt 等待点，恢复值只写入 hitl.resume_payload。"""

    def interrupt_wait(state: RuntimeGraphState) -> dict[str, Any]:
        hitl = state.get("hitl")
        if state.get("status") != "waiting_user" or not hitl:
            return {}
        resume_payload = interrupt({"hitl": dict(hitl)})
        command = normalize_interrupt_resume_value(resume_payload)
        return {
            "hitl": {**dict(hitl), "resume_payload": command},
            "hitl_resume": command,
        }

    return interrupt_wait

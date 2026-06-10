from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.types import Command


def build_validated_resume_command(resume_payload: Mapping[str, Any]) -> Command:
    """项目校验通过后才构造 LangGraph resume 命令。"""
    if not resume_payload.get("interrupt_id"):
        raise ValueError("interrupt_id is required before building Command(resume=...).")
    if not resume_payload.get("action"):
        raise ValueError("action is required before building Command(resume=...).")
    return Command(resume=dict(resume_payload))


def extract_resume_payload_from_command(command: Command) -> dict[str, Any]:
    """从 Command(resume=...) 取出图内恢复载荷。"""
    payload = getattr(command, "resume", None)
    if not isinstance(payload, Mapping):
        raise ValueError("Command(resume=...) payload must be a mapping.")
    return dict(payload)


def normalize_interrupt_resume_value(value: Any) -> dict[str, Any]:
    """节点内 interrupt() 返回的是 Command.resume 中的业务载荷。"""
    if not isinstance(value, Mapping):
        raise ValueError("interrupt() resume value must be a mapping.")
    return dict(value)

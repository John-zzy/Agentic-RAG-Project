from __future__ import annotations

from typing import Any

from typing_extensions import NotRequired

from langchain.agents.middleware.types import AgentState
from pydantic import Field

from backend.platform.agent_runtime.core.contracts import AgentRuntimeModel
from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext


class ReActInputState(AgentState[Any]):
    """Typed LangChain agent input state consumed by the ReAct provider."""

    react_run_id: NotRequired[str]
    user_goal: NotRequired[str]
    max_turns: NotRequired[int]
    metadata: NotRequired[dict[str, Any]]


class ReActState(ReActInputState):
    """Typed LangChain agent state persisted by the provider graph."""


class ReActContext(AgentRuntimeModel):
    """Typed runtime context passed into LangChain `create_agent`."""

    runtime: AgentRuntimeContext
    react_run_id: str
    user_goal: str
    max_turns: int = 5
    metadata: dict[str, Any] = Field(default_factory=dict)

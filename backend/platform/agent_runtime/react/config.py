from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from backend.platform.agent_runtime.core.contracts import ReActRun
from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext
from backend.platform.agent_runtime.middleware.factory import AgentMiddlewareBundle
from backend.platform.agent_runtime.react.factory import (
    ReActProviderFactory,
)
from backend.platform.agent_runtime.react.runtime import ReActRuntime
from backend.platform.agent_runtime.tooling.executor import ToolExecutor
from backend.platform.models.base.router import TaskComplexity


@dataclass(frozen=True)
class ReActDependencies:
    """Dependencies for the ChatGraph ReAct branch backed by LangChain."""

    tool_executor: ToolExecutor
    provider_factory: ReActProviderFactory
    middleware_bundle: AgentMiddlewareBundle
    runtime_context: AgentRuntimeContext
    session_id: str
    request_id: str
    user_goal: str
    system_prompt: str
    complexity: TaskComplexity = "simple"
    default_tool_inputs: Mapping[str, Mapping[str, Any]] | None = None
    react_run_id: str | None = None
    initial_run: ReActRun | None = None
    project_result: Callable[[ReActRun], Mapping[str, Any]] | None = None
    max_turns: int = 5

    def build_runtime(self) -> ReActRuntime:
        return ReActRuntime(
            tool_executor=self.tool_executor,
            provider_factory=self.provider_factory,
            middleware_bundle=self.middleware_bundle,
            context=self.runtime_context,
            system_prompt=self.system_prompt,
            complexity=self.complexity,
            max_turns=self.max_turns,
            default_tool_inputs=self.default_tool_inputs,
        )

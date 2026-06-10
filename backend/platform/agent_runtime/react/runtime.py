from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from backend.platform.agent_runtime.core.contracts import ReActRun
from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext
from backend.platform.agent_runtime.middleware.dynamic_prompt import DynamicPromptResult
from backend.platform.agent_runtime.middleware.factory import AgentMiddlewareBundle
from backend.platform.agent_runtime.react.factory import (
    ReActProviderFactory,
)
from backend.platform.agent_runtime.react.projection import (
    project_react_agent_output,
)
from backend.platform.agent_runtime.react.state import ReActContext
from backend.platform.agent_runtime.react.tools import build_react_tools
from backend.platform.agent_runtime.tooling.executor import ToolExecutor
from backend.platform.models.base.router import TaskComplexity
from langgraph.types import Command


@dataclass(frozen=True)
class ReActRuntime:
    """Run a single ReAct request through LangChain `create_agent`."""

    tool_executor: ToolExecutor
    provider_factory: ReActProviderFactory
    middleware_bundle: AgentMiddlewareBundle
    context: AgentRuntimeContext
    system_prompt: str
    complexity: TaskComplexity = "simple"
    max_turns: int = 5
    default_tool_inputs: Mapping[str, Mapping[str, Any]] | None = None

    def run(
        self,
        *,
        session_id: str,
        request_id: str,
        user_goal: str,
        react_run_id: str,
        initial_run: ReActRun | None = None,
        metadata: Mapping[str, Any] | None = None,
        resume_command: Command | None = None,
    ) -> ReActRun:
        if initial_run is not None and resume_command is None:
            return initial_run
        prompt = self._build_prompt()
        tools = build_react_tools(
            self.tool_executor,
            default_inputs=self.default_tool_inputs,
        )
        agent = self.provider_factory.build(
            complexity=self.complexity,
            tools=tools,
            system_prompt=prompt.system_prompt,
        )
        checkpoint_config = {
            "configurable": {
                "thread_id": self.context.workflow.thread_id
                or f"{session_id}:react:{request_id}",
                "checkpoint_ns": self.context.workflow.checkpoint_ns
                or f"react:{request_id}",
            }
        }
        input_payload: Mapping[str, Any] | Command = resume_command or {
            "messages": [{"role": "user", "content": user_goal}],
            "react_run_id": react_run_id,
            "user_goal": user_goal,
            "max_turns": self.max_turns,
            "metadata": dict(metadata or {}),
        }

        # 执行 ReAct Agent
        output = agent.invoke(
            input_payload,
            context=ReActContext(
                runtime=self.context,
                react_run_id=react_run_id,
                user_goal=user_goal,
                max_turns=self.max_turns,
                metadata=dict(metadata or {}),
            ),
            config=checkpoint_config,
        )

        # 把 LangChain 输出投影成 ReActRun
        projection = project_react_agent_output(
            output=output,
            session_id=session_id,
            request_id=request_id,
            user_goal=user_goal,
            react_run_id=react_run_id,
            max_turns=self.max_turns,
            trace_events=self.middleware_bundle.trace.events,
        )
        runtime_metadata = dict(projection.run.metadata)
        runtime_metadata["checkpoint"] = dict(checkpoint_config["configurable"])
        runtime_metadata["resume_source"] = "langchain_command" if resume_command else "new_run"
        return projection.run.model_copy(update={"metadata": runtime_metadata})

    def _build_prompt(self) -> DynamicPromptResult:
        scene_prompt = self.system_prompt
        if "Only use tools when" not in scene_prompt:
            scene_prompt = "\n\n".join(
                [
                    scene_prompt.strip(),
                    (
                        "Only use tools when they are needed. Return the final answer "
                        "directly when the request can be answered without tool evidence."
                    ),
                ]
            )
        return self.middleware_bundle.dynamic_prompt.compose_from_parts(
            context=self.context,
            scene_prompt=scene_prompt,
        )

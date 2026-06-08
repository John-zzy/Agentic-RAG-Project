from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain.tools import BaseTool, tool
from pydantic import BaseModel

from backend.platform.agent_runtime.core.contracts import ToolObservation
from backend.platform.agent_runtime.tooling.executor import ToolExecutor


class LangChainToolFactory:
    """Create standard LangChain tools from platform tool definitions."""

    def __init__(
        self,
        *,
        tool_executor: ToolExecutor,
        default_inputs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._tool_executor = tool_executor
        self._default_inputs = {
            str(tool_name): dict(payload)
            for tool_name, payload in dict(default_inputs or {}).items()
        }

    def build_tools(self) -> tuple[BaseTool, ...]:
        """Expose only tools allowed for the current scene and mounted sources."""
        return tuple(
            self.build_tool(tool_name)
            for tool_name in sorted(self._tool_executor.allowed_tools)
            if tool_name in self._tool_executor.registered_tools
        )

    def build_tool(self, tool_name: str) -> BaseTool:
        platform_tool = self._tool_executor.get_registered_tool(tool_name)
        invoke = _build_tool_callable(
            tool_executor=self._tool_executor,
            tool_name=tool_name,
            default_input=self._default_inputs.get(tool_name),
        )
        return tool(
            tool_name,
            description=_tool_description(platform_tool),
            args_schema=_tool_args_schema(platform_tool),
            response_format="content_and_artifact",
        )(invoke)


def build_langchain_tools_from_executor(
    tool_executor: ToolExecutor,
    *,
    default_inputs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[BaseTool, ...]:
    """Create LangChain tools for the executor's current scene/runtime scope."""
    return LangChainToolFactory(
        tool_executor=tool_executor,
        default_inputs=default_inputs,
    ).build_tools()


def observation_from_langchain_artifact(
    artifact: Mapping[str, Any],
) -> ToolObservation:
    """Recover the neutral observation stored in a LangChain tool artifact."""
    observation = artifact.get("tool_observation")
    if not isinstance(observation, Mapping):
        raise ValueError("LangChain tool artifact does not contain tool_observation.")
    return ToolObservation.model_validate(dict(observation))


def _build_tool_callable(
    *,
    tool_executor: ToolExecutor,
    tool_name: str,
    default_input: Mapping[str, Any] | None = None,
) -> Any:
    def invoke(**kwargs: Any) -> tuple[str, dict[str, Any]]:
        merged_input = dict(default_input or {})
        for key, value in kwargs.items():
            if value is None:
                continue
            if (
                isinstance(value, bool)
                and value is False
                and merged_input.get(key) is True
            ):
                continue
            merged_input[key] = value
        observation = tool_executor.execute(
            tool_name=tool_name,
            input_payload=merged_input,
        )
        return _observation_content(observation), _observation_artifact(observation)

    invoke.__name__ = f"{tool_name}_tool"
    invoke.__doc__ = f"Run {tool_name}."
    return invoke


def _tool_description(platform_tool: Any) -> str:
    description = getattr(platform_tool, "description", None)
    return str(description) if description else "Run a platform tool."


def _tool_args_schema(platform_tool: Any) -> type[BaseModel] | None:
    args_schema = getattr(platform_tool, "args_schema", None)
    if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
        return args_schema
    return None


def _observation_content(observation: ToolObservation) -> str:
    if observation.requires_user and observation.user_prompt:
        return observation.user_prompt
    if observation.result_summary:
        return observation.result_summary
    if observation.error:
        return observation.error
    return f"{observation.tool_name} completed."


def _observation_artifact(observation: ToolObservation) -> dict[str, Any]:
    return {
        "tool_observation": observation.model_dump(mode="json"),
        "tool_name": observation.tool_name,
        "status": _observation_status(observation),
        "citations": list(observation.citations),
        "trace": dict(observation.trace),
        "metadata": dict(observation.metadata),
    }


def _observation_status(observation: ToolObservation) -> str:
    if observation.metadata.get("cancelled"):
        return "cancelled"
    if observation.requires_user:
        return "waiting_user"
    if observation.success:
        return "succeeded"
    if observation.retryable:
        return "retryable"
    return "failed"

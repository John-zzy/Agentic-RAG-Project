from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from backend.platform.agent_runtime.middleware.factory import AgentMiddlewareBundle
from backend.platform.agent_runtime.middleware.tool_observation import observation_status
from backend.platform.agent_runtime.react.state import (
    ReActContext,
    ReActState,
)


class LangChainModelGuardAdapter(
    AgentMiddleware[ReActState, ReActContext, Any]
):
    """Adapter that applies project model guard and safe trace around LangChain calls."""

    state_schema = ReActState

    def __init__(self, *, bundle: AgentMiddlewareBundle) -> None:
        self._bundle = bundle

    def wrap_model_call(
        self,
        request: ModelRequest[ReActContext],
        handler: Callable[[ModelRequest[ReActContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        result = self._bundle.model_guard.invoke(
            lambda: handler(request),
            context=self._bundle.context,
            metadata={
                "provider": self._bundle.context.provider_name,
                "complexity": self._bundle.context.complexity,
            },
        )
        self._bundle.trace.record_model_call(
            context=self._bundle.context,
            latency_ms=result.metadata.latency_ms,
            retry_count=result.metadata.retry_count,
            provider=result.metadata.provider,
            complexity=result.metadata.complexity,
            metadata={
                "fallback_used": result.metadata.fallback_used,
                "error_classification": result.metadata.error_classification,
            },
        )
        if not result.success:
            raise RuntimeError(result.error or "LangChain model invocation failed.")
        if not isinstance(result.output, (ModelResponse, AIMessage, ExtendedModelResponse)):
            raise TypeError("LangChain model guard returned an unexpected response.")
        return result.output


class LangChainToolBoundaryAdapter(
    AgentMiddleware[ReActState, ReActContext, Any]
):
    """Apply project tool policy, HITL gate, observation and trace as LangChain tool middleware."""

    state_schema = ReActState

    def __init__(self, *, bundle: AgentMiddlewareBundle) -> None:
        self._bundle = bundle

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_name = str(request.tool_call.get("name") or "")
        tool_call_id = _tool_call_id(request)
        input_payload = _tool_args(request)
        args_schema = getattr(request.tool, "args_schema", None) if request.tool else None

        decision = self._bundle.tool_policy.validate(
            tool_name=tool_name,
            input_payload=input_payload,
            context=self._bundle.context,
            args_schema=args_schema,
        )
        if not decision.allowed:
            observation = self._bundle.tool_observation.normalize(
                tool_name=tool_name,
                error=ValueError(decision.reason or "Tool policy rejected call."),
                retryable=False,
            )
            return self._tool_message(
                observation=observation,
                tool_call_id=tool_call_id,
            )

        guarded_request = request.override(
            tool_call={**request.tool_call, "args": decision.input_payload}
        )
        try:
            result = handler(guarded_request)
        except Exception as exc:
            observation = self._bundle.tool_observation.normalize(
                tool_name=tool_name,
                error=exc,
                retryable=self._bundle.tool_policy.classify_retry(exc),
            )
            return self._tool_message(
                observation=observation,
                tool_call_id=tool_call_id,
            )

        if isinstance(result, ToolMessage):
            return self._record_tool_message(
                message=result,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )
        return result

    def _record_tool_message(
        self,
        *,
        message: ToolMessage,
        tool_name: str,
        tool_call_id: str,
    ) -> ToolMessage:
        observation = self._bundle.tool_observation.normalize(
            tool_name=tool_name,
            result=_observation_payload(message),
        ).model_copy(update={"tool_call_id": tool_call_id})
        self._bundle.trace.record_tool_call(
            context=self._bundle.context,
            observation=observation,
        )
        return message.model_copy(
            update={
                "tool_call_id": tool_call_id,
                "artifact": _artifact_from_message(message, observation),
                "status": "success" if observation.success else "error",
            }
        )

    def _tool_message(
        self,
        *,
        observation: Any,
        tool_call_id: str,
    ) -> ToolMessage:
        observation = observation.model_copy(update={"tool_call_id": tool_call_id})
        self._bundle.trace.record_tool_call(
            context=self._bundle.context,
            observation=observation,
        )
        return ToolMessage(
            content=_observation_content(observation),
            tool_call_id=tool_call_id,
            name=observation.tool_name,
            artifact=_observation_artifact(observation),
            status="success" if observation.success else "error",
        )


def _tool_call_id(request: ToolCallRequest) -> str:
    return str(request.tool_call.get("id") or f"call:{request.tool_call.get('name') or 'tool'}")


def _tool_args(request: ToolCallRequest) -> dict[str, Any]:
    args = request.tool_call.get("args")
    return dict(args) if isinstance(args, dict) else {}


def _observation_payload(message: ToolMessage) -> Any:
    artifact = message.artifact
    if isinstance(artifact, dict) and isinstance(artifact.get("tool_observation"), dict):
        return artifact["tool_observation"]
    return {
        "tool_name": message.name or "unknown_tool",
        "success": message.status == "success",
        "output": message.content,
        "result_summary": _message_content(message),
        "error": None if message.status == "success" else _message_content(message),
    }


def _artifact_from_message(message: ToolMessage, observation: Any) -> dict[str, Any]:
    artifact = dict(message.artifact) if isinstance(message.artifact, dict) else {}
    artifact["tool_observation"] = observation.model_dump(mode="json")
    artifact["tool_name"] = observation.tool_name
    artifact["status"] = observation_status(observation)
    return artifact


def _observation_content(observation: Any) -> str:
    if observation.requires_user and observation.user_prompt:
        return observation.user_prompt
    if observation.result_summary:
        return observation.result_summary
    if observation.error:
        return observation.error
    return f"{observation.tool_name} completed."


def _observation_artifact(observation: Any) -> dict[str, Any]:
    return {
        "tool_observation": observation.model_dump(mode="json"),
        "tool_name": observation.tool_name,
        "status": observation_status(observation),
        "citations": list(observation.citations),
        "trace": dict(observation.trace),
        "metadata": dict(observation.metadata),
    }


def _message_content(message: ToolMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import Field

from backend.platform.agent_runtime.core.contracts import (
    AgentRuntimeModel,
    ReActAction,
    ReActRun,
    ReActTurn,
    ToolExecutionMetadata,
    ToolObservation,
)
from backend.platform.agent_runtime.tooling.langchain import (
    observation_from_langchain_artifact,
)
from backend.platform.agent_runtime.middleware.trace import sanitize_for_trace


class ReActProjection(AgentRuntimeModel):
    """Provider-neutral projection created from a LangChain agent run."""

    run: ReActRun
    messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def project_react_agent_output(
    *,
    output: Mapping[str, Any],
    session_id: str,
    request_id: str,
    user_goal: str,
    react_run_id: str,
    max_turns: int,
    trace_events: Sequence[Any] = (),
) -> ReActProjection:
    """Project LangChain messages into the stable ReActRun contract."""

    interrupt_projection = _project_interrupt_output(
        output=output,
        session_id=session_id,
        request_id=request_id,
        user_goal=user_goal,
        react_run_id=react_run_id,
        max_turns=max_turns,
        trace_events=trace_events,
    )
    if interrupt_projection is not None:
        return interrupt_projection

    messages = _coerce_messages(output.get("messages"))
    turn_builder = _ReActTurnBuilder(
        user_goal=user_goal,
        max_turns=max_turns,
    )
    for message in messages:
        turn_builder.consume(message)

    run = ReActRun(
        react_run_id=react_run_id,
        session_id=session_id,
        request_id=request_id,
        user_goal=user_goal,
        max_turns=max_turns,
        turns=turn_builder.turns,
        observations=turn_builder.observations,
        workflow_status=turn_builder.workflow_status,
        current_turn_id=turn_builder.current_turn_id,
        current_tool_call=turn_builder.current_tool_call,
        final_answer=turn_builder.final_answer,
        result_summary=turn_builder.result_summary,
        error=turn_builder.error,
        metadata={
            "provider": "langchain_create_agent",
            "rationale_summary": turn_builder.rationale_summary,
            "trace_events": [_dump_trace_event(event) for event in trace_events],
        },
    )
    return ReActProjection(
        run=run,
        messages=[_safe_message_dump(message) for message in messages],
        metadata={"message_count": len(messages)},
    )


def _project_interrupt_output(
    *,
    output: Mapping[str, Any],
    session_id: str,
    request_id: str,
    user_goal: str,
    react_run_id: str,
    max_turns: int,
    trace_events: Sequence[Any],
) -> ReActProjection | None:
    interrupts = output.get("__interrupt__")
    if not isinstance(interrupts, Sequence) or isinstance(interrupts, (str, bytes)):
        return None
    first_interrupt = next(iter(interrupts), None)
    value = getattr(first_interrupt, "value", first_interrupt)
    if not isinstance(value, Mapping):
        return None
    action_requests = value.get("action_requests")
    review_configs = value.get("review_configs")
    if not isinstance(action_requests, list) or not action_requests:
        return None
    action_request = action_requests[0]
    if not isinstance(action_request, Mapping):
        return None
    review_config = review_configs[0] if isinstance(review_configs, list) and review_configs else {}
    if not isinstance(review_config, Mapping):
        review_config = {}
    tool_name = str(action_request.get("name") or "")
    tool_args = dict(action_request.get("args") or {})
    tool_call_id = str(action_request.get("id") or f"hitl:{request_id}:{tool_name}:1")
    allowed_decisions = [
        str(decision)
        for decision in list(review_config.get("allowed_decisions") or ["approve", "reject"])
    ]
    description = str(
        action_request.get("description")
        or f"Tool execution requires approval: {tool_name}."
    )
    interrupt_id = f"hitl:{request_id}:{tool_name}:{tool_call_id}"
    hitl_metadata = {
        "mode": "react",
        "react_run_id": react_run_id,
        "current_turn_id": "turn-1",
        "user_goal": user_goal,
        "source": "langchain_human_in_the_loop",
        "langchain": {
            "interrupt_id": interrupt_id,
            "action_requests": [dict(item) for item in action_requests if isinstance(item, Mapping)],
            "review_configs": [dict(item) for item in review_configs or () if isinstance(item, Mapping)],
        },
    }
    turn = ReActTurn(
        turn_id="turn-1",
        round_index=1,
        goal=user_goal,
        action=ReActAction(
            action_type="tool_call",
            tool_name=tool_name or None,
            input=tool_args,
            rationale_summary="Tool call is waiting for LangChain human review.",
            metadata={"tool_call_id": tool_call_id},
        ),
        status="waiting_user",
        input=tool_args,
        tool_name=tool_name or None,
        observation_summary=description,
        result_summary=description,
        metadata={"hitl": hitl_metadata},
    )
    run = ReActRun(
        react_run_id=react_run_id,
        session_id=session_id,
        request_id=request_id,
        user_goal=user_goal,
        workflow_status="waiting_user",
        max_turns=max_turns,
        turns=[turn],
        observations=[],
        current_turn_id=turn.turn_id,
        current_tool_call=ToolExecutionMetadata(
            tool_name=tool_name or "unknown_tool",
            tool_call_id=tool_call_id,
            metadata={"args": tool_args},
        ),
        result_summary=description,
        metadata={
            "provider": "langchain_create_agent",
            "hitl": hitl_metadata,
            "trace_events": [_dump_trace_event(event) for event in trace_events],
        },
    )
    return ReActProjection(
        run=run,
        messages=[],
        metadata={"interrupt": dict(value), "message_count": 0},
    )


class _ReActTurnBuilder:
    def __init__(self, *, user_goal: str, max_turns: int) -> None:
        self._user_goal = user_goal
        self._max_turns = max_turns
        self.turns: list[ReActTurn] = []
        self.observations: list[ToolObservation] = []
        self.workflow_status = "succeeded"
        self.current_turn_id: str | None = None
        self.current_tool_call = None
        self.final_answer: str | None = None
        self.result_summary = ""
        self.error: str | None = None
        self.rationale_summary = ""

    def consume(self, message: BaseMessage) -> None:
        if isinstance(message, AIMessage):
            self._consume_ai_message(message)
            return
        if isinstance(message, ToolMessage):
            self._consume_tool_message(message)

    def _consume_ai_message(self, message: AIMessage) -> None:
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if len(self.turns) >= self._max_turns:
                    self._mark_failed("LangChain ReAct provider exceeded max_turns.")
                    return
                self._append_tool_turn(tool_call)
            return
        content = _message_text(message)
        if content:
            self._append_final_turn(content)

    def _consume_tool_message(self, message: ToolMessage) -> None:
        turn = self._find_turn_by_tool_call_id(message.tool_call_id)
        if turn is None:
            return
        observation = _observation_from_tool_message(message, tool_name=turn.tool_name)
        turn.observation = observation
        turn.observation_summary = observation.result_summary
        turn.result_summary = observation.result_summary
        turn.status = "waiting_user" if observation.requires_user else (
            "succeeded" if observation.success else "failed"
        )
        self.observations.append(observation)
        if observation.requires_user:
            self.workflow_status = "waiting_user"
            self.current_turn_id = turn.turn_id
            self.current_tool_call = observation.execution
        elif not observation.success:
            self._mark_failed(observation.error or observation.result_summary)

    def _append_tool_turn(self, tool_call: Mapping[str, Any]) -> None:
        round_index = len(self.turns) + 1
        tool_name = str(tool_call.get("name") or "")
        tool_call_id = str(tool_call.get("id") or f"call-{round_index}")
        turn = ReActTurn(
            turn_id=f"turn-{round_index}",
            round_index=round_index,
            goal=self._user_goal,
            action=ReActAction(
                action_type="tool_call",
                tool_name=tool_name,
                input=dict(tool_call.get("args") or {}),
                rationale_summary="tool selected by LangChain provider",
                metadata={"tool_call_id": tool_call_id},
            ),
            status="running",
            input=dict(tool_call.get("args") or {}),
            metadata={"tool_call_id": tool_call_id},
        )
        self.turns.append(turn)
        self.current_turn_id = turn.turn_id

    def _append_final_turn(self, content: str) -> None:
        if len(self.turns) >= self._max_turns:
            self._mark_failed("LangChain ReAct provider exceeded max_turns.")
            return
        round_index = len(self.turns) + 1
        turn = ReActTurn(
            turn_id=f"turn-{round_index}",
            round_index=round_index,
            goal=self._user_goal,
            action=ReActAction(
                action_type="final_answer",
                instruction=content,
                rationale_summary="final answer projected from LangChain provider",
            ),
            status="succeeded",
            result_summary=content,
        )
        self.turns.append(turn)
        if self.workflow_status != "failed":
            self.workflow_status = "succeeded"
            self.current_turn_id = None
            self.current_tool_call = None
            self.final_answer = content
            self.result_summary = content

    def _find_turn_by_tool_call_id(self, tool_call_id: str) -> ReActTurn | None:
        for turn in reversed(self.turns):
            if turn.metadata.get("tool_call_id") == tool_call_id:
                return turn
        return None

    def _mark_failed(self, error: str) -> None:
        self.workflow_status = "failed"
        self.error = error
        self.result_summary = error
        self.current_tool_call = None


def _coerce_messages(value: Any) -> list[BaseMessage]:
    if not isinstance(value, list):
        return []
    return [message for message in value if isinstance(message, BaseMessage)]


def _observation_from_tool_message(
    message: ToolMessage,
    *,
    tool_name: str | None,
) -> ToolObservation:
    artifact = message.artifact
    if isinstance(artifact, Mapping):
        try:
            return observation_from_langchain_artifact(artifact)
        except ValueError:
            pass
    return ToolObservation(
        tool_name=tool_name or message.name or "unknown_tool",
        success=message.status == "success",
        tool_call_id=message.tool_call_id,
        output=message.content,
        result_summary=_message_text(message),
        error=None if message.status == "success" else _message_text(message),
    )


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        return "\n".join(part.strip() for part in parts if part.strip())
    return ""


def _safe_message_dump(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": message.type}
    if isinstance(message, AIMessage):
        payload["has_tool_calls"] = bool(message.tool_calls)
        payload["tool_names"] = [str(call.get("name")) for call in message.tool_calls]
        if not message.tool_calls:
            payload["content"] = _message_text(message)
    elif isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
        payload["status"] = message.status
        payload["content"] = _message_text(message)
    return payload


def _dump_trace_event(event: Any) -> dict[str, Any]:
    if hasattr(event, "model_dump"):
        return dict(sanitize_for_trace(event.model_dump()))
    if isinstance(event, Mapping):
        return dict(sanitize_for_trace(event))
    return dict(sanitize_for_trace({"event": str(event)}))

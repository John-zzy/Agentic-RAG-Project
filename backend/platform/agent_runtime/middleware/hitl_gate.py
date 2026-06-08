from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from backend.platform.agent_runtime.core.contracts import AgentRuntimeModel, ToolExecutionMetadata
from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext
from backend.platform.agent_runtime.middleware.tool_observation import ToolObservationMiddleware
from backend.platform.agent_runtime.middleware.tool_policy import ToolPolicyDecision
from backend.platform.workflow.langgraph.state import RuntimeHitlState, build_runtime_hitl_state


HitlGateStatus = Literal["execute", "waiting_user", "rejected"]


class HitlGateDecision(AgentRuntimeModel):
    status: HitlGateStatus
    tool_name: str
    interrupt_id: str | None = None
    wait_state: dict[str, Any] | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HitlGatePolicy:
    approval_required_tools: set[str] = field(default_factory=set)
    pending_action: str = "tool_approval"
    allowed_actions: tuple[str, ...] = ("approve", "reject")


class HitlGateMiddleware:
    """Create waiting_user state before approval-required side-effect tools run."""

    def __init__(
        self,
        *,
        policy: HitlGatePolicy | None = None,
        observation_middleware: ToolObservationMiddleware | None = None,
    ) -> None:
        self._policy = policy or HitlGatePolicy()
        self._observations = observation_middleware or ToolObservationMiddleware()

    def evaluate(
        self,
        *,
        context: AgentRuntimeContext,
        decision: ToolPolicyDecision,
        input_payload: Mapping[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> HitlGateDecision:
        if not decision.allowed:
            return HitlGateDecision(
                status="rejected",
                tool_name=decision.tool_name,
                reason=decision.reason,
                metadata={"policy": decision.metadata},
            )
        if not self._requires_approval(decision):
            return HitlGateDecision(status="execute", tool_name=decision.tool_name)

        interrupt_id = _interrupt_id(context=context, tool_name=decision.tool_name, tool_call_id=tool_call_id)
        wait_state: RuntimeHitlState = build_runtime_hitl_state(
            interrupt_id=interrupt_id,
            thread_id=context.workflow.thread_id or context.session_id,
            reason=f"Tool requires approval: {decision.tool_name}.",
            pending_action=self._policy.pending_action,
            allowed_actions=self._policy.allowed_actions,
            proposed_tool_call={
                "tool_name": decision.tool_name,
                "tool_call_id": tool_call_id,
                "input_payload": dict(input_payload or decision.input_payload),
                "risk_level": decision.risk_level,
            },
            metadata={
                "session_id": context.session_id,
                "request_id": context.request_id,
                "risk_level": decision.risk_level,
            },
        )
        return HitlGateDecision(
            status="waiting_user",
            tool_name=decision.tool_name,
            interrupt_id=interrupt_id,
            wait_state=dict(wait_state),
            reason=wait_state["reason"],
            metadata={"pending_action": self._policy.pending_action},
        )

    def run_or_wait(
        self,
        *,
        context: AgentRuntimeContext,
        decision: ToolPolicyDecision,
        invoke: Callable[[], Any],
        input_payload: Mapping[str, Any] | None = None,
        tool_call_id: str | None = None,
        resume_accepted: bool = False,
    ) -> tuple[HitlGateDecision, Any | None]:
        gate_decision = self.evaluate(
            context=context,
            decision=decision,
            input_payload=input_payload,
            tool_call_id=tool_call_id,
        )
        if gate_decision.status == "waiting_user" and not resume_accepted:
            return gate_decision, None
        if gate_decision.status == "rejected":
            return gate_decision, None
        try:
            result = invoke()
        except Exception as exc:
            execution = ToolExecutionMetadata(tool_name=decision.tool_name, tool_call_id=tool_call_id)
            result = self._observations.normalize(
                tool_name=decision.tool_name,
                error=exc,
                execution=execution,
            )
        return HitlGateDecision(status="execute", tool_name=decision.tool_name), result

    def _requires_approval(self, decision: ToolPolicyDecision) -> bool:
        return (
            decision.risk_level == "high"
            or decision.tool_name in self._policy.approval_required_tools
        )


def _interrupt_id(
    *,
    context: AgentRuntimeContext,
    tool_name: str,
    tool_call_id: str | None,
) -> str:
    if context.workflow.interrupt_id:
        return context.workflow.interrupt_id
    stable_part = tool_call_id or str(uuid4())
    return f"hitl:{context.request_id}:{tool_name}:{stable_part}"

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext
from backend.platform.agent_runtime.middleware.dynamic_prompt import DynamicPromptMiddleware
from backend.platform.agent_runtime.middleware.hitl_gate import (
    HitlGateMiddleware,
    HitlGatePolicy,
)
from backend.platform.agent_runtime.middleware.model_guard import (
    ModelGuardMiddleware,
    ModelGuardPolicy,
)
from backend.platform.agent_runtime.middleware.tool_observation import ToolObservationMiddleware
from backend.platform.agent_runtime.middleware.tool_policy import (
    ToolPolicyMiddleware,
    ToolRiskLevel,
    build_tool_policy_config,
)
from backend.platform.agent_runtime.middleware.trace import RuntimeTraceMiddleware


@dataclass(frozen=True, slots=True)
class AgentMiddlewareBundle:
    """Ordered runtime middleware components for LangChain ReAct provider wiring."""

    context: AgentRuntimeContext
    dynamic_prompt: DynamicPromptMiddleware
    model_guard: ModelGuardMiddleware
    tool_policy: ToolPolicyMiddleware
    tool_observation: ToolObservationMiddleware
    hitl_gate: HitlGateMiddleware
    trace: RuntimeTraceMiddleware
    hitl_interrupts: dict[str, dict[str, object]]

    @property
    def ordered(self) -> tuple[object, ...]:
        return (
            self.dynamic_prompt,
            self.model_guard,
            self.tool_policy,
            self.tool_observation,
            self.hitl_gate,
            self.trace,
        )


def build_agent_middleware(
    *,
    context: AgentRuntimeContext,
    allowed_tools: Sequence[str],
    tool_source_scope: Mapping[str, Sequence[str]] | None = None,
    high_risk_tools: Sequence[str] = (),
    risk_by_tool: Mapping[str, ToolRiskLevel] | None = None,
    max_calls_per_tool: int | None = None,
    model_policy: ModelGuardPolicy | None = None,
    hitl_policy: HitlGatePolicy | None = None,
) -> AgentMiddlewareBundle:
    tool_observation = ToolObservationMiddleware()
    approval_required_tools = set(high_risk_tools)
    if hitl_policy is not None:
        approval_required_tools.update(hitl_policy.approval_required_tools)
    return AgentMiddlewareBundle(
        context=context,
        dynamic_prompt=DynamicPromptMiddleware(),
        model_guard=ModelGuardMiddleware(policy=model_policy),
        tool_policy=ToolPolicyMiddleware(
            build_tool_policy_config(
                allowed_tools=allowed_tools,
                tool_source_scope=tool_source_scope,
                high_risk_tools=high_risk_tools,
                risk_by_tool=risk_by_tool,
                max_calls_per_tool=max_calls_per_tool,
            )
        ),
        tool_observation=tool_observation,
        hitl_gate=HitlGateMiddleware(
            policy=hitl_policy,
            observation_middleware=tool_observation,
        ),
        trace=RuntimeTraceMiddleware(),
        hitl_interrupts={
            tool_name: {"allowed_decisions": ["approve", "reject", "respond"]}
            for tool_name in sorted(approval_required_tools)
        },
    )

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from backend.platform.agent_runtime import (
    MinimalPlanner,
    PlanExecutor,
    PlanRun,
    PlanStep,
    PlanSynthesisContext,
    PlanSynthesisResult,
    ToolObservation,
)
from backend.platform.agent_runtime.tool_executor import ToolExecutor
from backend.platform.agent_runtime.validation import ToolAccessValidationError
from backend.platform.tools.base import ToolResult


class _QueryArgs(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=1, ge=1)


class _RecordingPlanSynthesizer:
    def __init__(self) -> None:
        self.contexts: list[PlanSynthesisContext] = []

    def synthesize(self, context: PlanSynthesisContext) -> PlanSynthesisResult:
        self.contexts.append(context)
        summaries = [step.result_summary for step in context.steps if step.status == "succeeded"]
        return PlanSynthesisResult(
            final_answer=" / ".join(summaries),
            result_summary=f"finalized {len(summaries)} step(s)",
            citations=list(context.citations),
            knowledge_used=bool(context.citations),
            metadata={"step_ids": [step.step_id for step in context.steps]},
        )


def test_minimal_planner_creates_single_step_and_executor_succeeds() -> None:
    calls: list[dict[str, Any]] = []
    executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=calls,
                records=[{"policy": "travel"}],
                citations=[{"citation_id": "policy-1"}],
            )
        },
        allowed_tools={"lookup_policy"},
    )

    plan = MinimalPlanner(
        tool_executor=executor,
        plan_run_id_factory=lambda: "plan-1",
        step_id_factory=lambda index: f"step-{index}",
    ).create_plan(
        session_id="session-1",
        request_id="request-1",
        user_goal="查差旅报销制度",
        mounted_knowledge_sources=("documents",),
    )
    result = PlanExecutor(tool_executor=executor).execute(plan)

    assert plan.workflow_status == "succeeded"
    assert result.workflow_status == "succeeded"
    assert result.plan_run_id == "plan-1"
    assert result.metadata["planner"]["workflow_transitions"] == [
        {"from": "created", "event": "plan_start", "to": "planning"}
    ]
    assert result.steps[0].step_id == "step-1"
    assert result.steps[0].status == "succeeded"
    assert result.steps[0].input == {"query": "查差旅报销制度", "limit": 1}
    assert result.steps[0].output["records"] == [{"policy": "travel"}]
    assert result.steps[0].result_summary == "lookup_policy succeeded with 1 record(s)."
    assert result.final_answer == "lookup_policy succeeded with 1 record(s)."
    assert result.metadata["citations"] == [{"citation_id": "policy-1"}]
    assert calls == [{"query": "查差旅报销制度", "limit": 1}]


def test_minimal_planner_uses_object_style_plan_tools_policy() -> None:
    executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=[],
                records=[{"policy": "travel"}],
            ),
            "lookup_inventory": _structured_tool(
                name="lookup_inventory",
                calls=[],
                records=[{"sku": "sku-1"}],
            ),
        },
        allowed_tools={"lookup_policy", "lookup_inventory"},
    )

    class _Policy:
        plan_tools = ("lookup_inventory",)

    plan = MinimalPlanner(tool_executor=executor).create_plan(
        session_id="session-1",
        request_id="request-policy",
        user_goal="按策略选择工具",
        scene_policy=_Policy(),
    )

    assert [step.tool_name for step in plan.steps] == ["lookup_inventory"]


def test_plan_executor_runs_multi_step_plan_in_dependency_order() -> None:
    policy_calls: list[dict[str, Any]] = []
    inventory_calls: list[dict[str, Any]] = []
    executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=policy_calls,
                records=[{"policy": "return"}],
            ),
            "lookup_inventory": _structured_tool(
                name="lookup_inventory",
                calls=inventory_calls,
                records=[{"sku": "sku-1", "available": True}],
            ),
        },
        allowed_tools={"lookup_policy", "lookup_inventory"},
    )
    plan = PlanRun(
        plan_run_id="plan-deps",
        session_id="session-1",
        request_id="request-deps",
        user_goal="先查退货规则，再查库存",
        steps=[
            PlanStep(
                step_id="step-1",
                goal="查退货规则",
                tool_name="lookup_policy",
                input={"query": "return"},
            ),
            PlanStep(
                step_id="step-2",
                goal="查库存",
                tool_name="lookup_inventory",
                input={"query": "sku-1"},
                depends_on=["step-1"],
            ),
        ],
    )

    result = PlanExecutor(tool_executor=executor).execute(plan)

    assert result.workflow_status == "succeeded"
    assert result.metadata["execution_order"] == ["step-1", "step-2"]
    assert [step.status for step in result.steps] == ["succeeded", "succeeded"]
    assert policy_calls == [{"query": "return", "limit": 1}]
    assert inventory_calls == [{"query": "sku-1", "limit": 1}]


def test_planner_rejects_invalid_tool_before_execution() -> None:
    executor = ToolExecutor(tools={}, allowed_tools=set())

    with pytest.raises(ToolAccessValidationError, match="not allowed"):
        MinimalPlanner(tool_executor=executor).create_plan(
            session_id="session-1",
            request_id="request-invalid-tool",
            user_goal="调用不可用工具",
            proposed_steps=[
                {
                    "step_id": "step-1",
                    "goal": "非法步骤",
                    "tool_name": "unsafe_tool",
                    "input": {"query": "x"},
                }
            ],
        )


def test_plan_executor_retries_retryable_error_and_then_succeeds() -> None:
    calls: list[dict[str, Any]] = []

    def flaky_lookup(query: str, limit: int = 1) -> ToolResult:
        calls.append({"query": query, "limit": limit})
        if len(calls) == 1:
            raise TimeoutError("temporary timeout")
        return ToolResult.ok(tool_name="lookup_policy", records=[{"policy": "ok"}])

    executor = ToolExecutor(
        tools={"lookup_policy": _tool_from_func("lookup_policy", flaky_lookup)},
        allowed_tools={"lookup_policy"},
    )
    plan = _single_step_plan(tool_name="lookup_policy", query="policy")

    result = PlanExecutor(tool_executor=executor).execute(plan)

    assert result.workflow_status == "succeeded"
    assert result.steps[0].status == "succeeded"
    assert result.steps[0].retry_metadata.attempt == 2
    assert result.steps[0].retry_metadata.last_error is None
    assert [transition["event"] for transition in result.metadata["workflow_transitions"]] == [
        "run_start",
        "tool_error_retryable",
        "retry",
        "success",
    ]
    assert calls == [
        {"query": "policy", "limit": 1},
        {"query": "policy", "limit": 1},
    ]


def test_plan_executor_fails_after_retry_exhaustion() -> None:
    calls: list[dict[str, Any]] = []

    def timeout_lookup(query: str, limit: int = 1) -> ToolResult:
        calls.append({"query": query, "limit": limit})
        raise TimeoutError("timeout exhausted")

    executor = ToolExecutor(
        tools={"lookup_policy": _tool_from_func("lookup_policy", timeout_lookup)},
        allowed_tools={"lookup_policy"},
    )
    plan = _single_step_plan(tool_name="lookup_policy", query="policy")

    result = PlanExecutor(tool_executor=executor).execute(plan)

    assert result.workflow_status == "failed"
    assert result.error == "timeout exhausted"
    assert result.steps[0].status == "failed"
    assert result.steps[0].retry_metadata.attempt == 2
    assert [transition["event"] for transition in result.metadata["workflow_transitions"]] == [
        "run_start",
        "tool_error_retryable",
        "retry",
        "tool_error_retryable",
        "tool_error_final",
    ]
    assert calls == [
        {"query": "policy", "limit": 1},
        {"query": "policy", "limit": 1},
    ]


def test_plan_final_synthesis_uses_successful_step_summaries_and_citations() -> None:
    calls: list[dict[str, Any]] = []
    synthesizer = _RecordingPlanSynthesizer()
    executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=calls,
                records=[{"policy": "expense"}],
                citations=[{"citation_id": "policy-1"}],
            )
        },
        allowed_tools={"lookup_policy"},
    )
    plan = _single_step_plan(tool_name="lookup_policy", query="expense")

    result = PlanExecutor(
        tool_executor=executor,
        final_synthesizer=synthesizer,
    ).execute(plan)

    assert result.workflow_status == "succeeded"
    assert result.final_answer == "lookup_policy succeeded with 1 record(s)."
    assert result.result_summary == "finalized 1 step(s)"
    assert result.metadata["citations"] == [{"citation_id": "policy-1"}]
    assert result.metadata["knowledge_used"] is True
    assert result.metadata["final_synthesis"] == {"step_ids": ["step-1"]}
    assert synthesizer.contexts[0].steps == result.steps
    assert synthesizer.contexts[0].citations == [{"citation_id": "policy-1"}]


def test_plan_executor_tool_observation_requires_user_creates_wait_metadata() -> None:
    observation = ToolObservation(
        tool_name="approval_tool",
        success=False,
        requires_user=True,
        user_prompt="是否批准执行该步骤？",
        result_summary="等待用户批准。",
    )
    tool = _StaticObservationTool(name="approval_tool", observation=observation)
    executor = ToolExecutor(
        tools={"approval_tool": tool},
        allowed_tools={"approval_tool"},
    )
    plan = _single_step_plan(tool_name="approval_tool", query="approval")

    result = PlanExecutor(tool_executor=executor).execute(plan)

    assert result.workflow_status == "waiting_user"
    assert result.current_step_id == "step-1"
    assert result.steps[0].status == "waiting_user"
    assert result.metadata["hitl"] == {
        "mode": "plan",
        "plan_run_id": "plan-1",
        "current_step_id": "step-1",
        "user_prompt": "是否批准执行该步骤？",
        "source": "tool_observation",
    }
    assert result.steps[0].metadata["hitl"] == result.metadata["hitl"]
    assert result.steps[0].observation is not None
    assert result.steps[0].observation.metadata["hitl"] == result.metadata["hitl"]
    assert tool.calls == [{"query": "approval", "limit": 1}]


def _single_step_plan(*, tool_name: str, query: str) -> PlanRun:
    return PlanRun(
        plan_run_id="plan-1",
        session_id="session-1",
        request_id="request-1",
        user_goal=query,
        steps=[
            PlanStep(
                step_id="step-1",
                goal=query,
                tool_name=tool_name,
                input={"query": query},
            )
        ],
    )


def _structured_tool(
    *,
    name: str,
    calls: list[dict[str, Any]],
    records: list[dict[str, Any]],
    citations: list[dict[str, Any]] | None = None,
) -> StructuredTool:
    def invoke_tool(query: str, limit: int = 1) -> ToolResult:
        calls.append({"query": query, "limit": limit})
        return ToolResult.ok(
            tool_name=name,
            records=records,
            citations=citations or [],
        )

    return _tool_from_func(name, invoke_tool)


def _tool_from_func(name: str, func: Any) -> StructuredTool:
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=f"Run {name}.",
        args_schema=_QueryArgs,
    )


class _StaticObservationTool:
    description = "Return a static tool observation."
    args_schema = _QueryArgs

    def __init__(self, *, name: str, observation: ToolObservation) -> None:
        self.name = name
        self.observation = observation
        self.calls: list[dict[str, Any]] = []

    def invoke(self, input_payload: dict[str, Any]) -> ToolObservation:
        self.calls.append(dict(input_payload))
        return self.observation

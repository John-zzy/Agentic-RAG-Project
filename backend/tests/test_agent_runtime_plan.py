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
    ToolExecutionMetadata,
    ToolObservation,
)
from backend.platform.agent_runtime.middleware import RuntimeTraceMiddleware, SharedModelCallGuard
from backend.platform.agent_runtime.plan.graph import build_plan_graph
from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.tooling.executor import ToolExecutor
from backend.platform.agent_runtime.core.validation import ToolAccessValidationError
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


class _RecordingStepSelector:
    def __init__(self, steps: list[dict[str, Any]]) -> None:
        self.steps = steps
        self.contexts: list[Any] = []

    def select_steps(self, context: Any):
        self.contexts.append(context)
        return self.steps


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
    assert result.observations == [result.steps[0].observation]
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
    assert plan.metadata["planner"]["step_source"] == "scene_policy.plan_tools"


def test_minimal_planner_uses_explicit_scene_policy_plan_steps() -> None:
    executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=[],
                records=[{"policy": "travel"}],
            )
        },
        allowed_tools={"lookup_policy"},
    )

    class _Policy:
        plan_steps = (
            {
                "step_id": "policy-step",
                "goal": "查制度",
                "tool_name": "lookup_policy",
                "input": {"query": "travel", "limit": 2},
            },
        )

    plan = MinimalPlanner(tool_executor=executor).create_plan(
        session_id="session-1",
        request_id="request-policy-steps",
        user_goal="按显式步骤查制度",
        scene_policy=_Policy(),
    )

    assert plan.metadata["planner"]["step_source"] == "scene_policy.plan_steps"
    assert plan.steps[0].step_id == "policy-step"
    assert plan.steps[0].input == {"query": "travel", "limit": 2}


def test_minimal_planner_uses_candidate_tools_and_tool_input_defaults() -> None:
    executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=[],
                records=[{"policy": "return"}],
            ),
            "lookup_inventory": _structured_tool(
                name="lookup_inventory",
                calls=[],
                records=[{"sku": "sku-1"}],
            ),
        },
        allowed_tools={"lookup_policy", "lookup_inventory"},
    )

    plan = MinimalPlanner(tool_executor=executor).create_plan(
        session_id="session-1",
        request_id="request-candidates",
        user_goal="先查库存再查规则",
        mounted_knowledge_sources=("documents", "inventory"),
        candidate_tools=("lookup_inventory", "lookup_policy"),
        scene_policy={
            "plan_tool_inputs": {
                "lookup_inventory": {"query": "sku-1", "limit": 1},
                "lookup_policy": {"query": "return", "limit": 1},
            }
        },
    )

    assert plan.metadata["planner"]["step_source"] == "candidate_tools"
    assert [step.tool_name for step in plan.steps] == ["lookup_inventory", "lookup_policy"]
    assert plan.steps[1].depends_on == ["step-1"]
    assert plan.steps[0].input == {"query": "sku-1", "limit": 1}
    assert plan.context_summary


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
    assert result.observations == [
        result.steps[0].observation,
        result.steps[1].observation,
    ]
    assert policy_calls == [{"query": "return", "limit": 1}]
    assert inventory_calls == [{"query": "sku-1", "limit": 1}]


def test_plan_tool_execution_records_middleware_trace_in_dependency_order() -> None:
    trace = RuntimeTraceMiddleware()
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
        trace_middleware=trace,
    )
    plan = PlanRun(
        plan_run_id="plan-trace",
        session_id="session-trace",
        request_id="request-trace",
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

    assert result.metadata["execution_order"] == ["step-1", "step-2"]
    assert [event.metadata["tool_name"] for event in trace.events] == [
        "lookup_policy",
        "lookup_inventory",
    ]
    assert all(event.metadata["tool_status"] == "succeeded" for event in trace.events)


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
    assert len(result.observations) == 2
    assert result.observations[-1] == result.steps[0].observation
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
    assert len(result.observations) == 2
    assert result.observations[-1] == result.steps[0].observation
    assert result.error == result.observations[-1].error
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


def test_plan_executor_blocks_steps_when_dependency_fails() -> None:
    failed_observation = ToolObservation(
        tool_name="lookup_policy",
        success=False,
        retryable=False,
        error="policy lookup failed",
        result_summary="policy lookup failed",
    )
    failing_tool = _StaticObservationTool(name="lookup_policy", observation=failed_observation)
    inventory_calls: list[dict[str, Any]] = []
    executor = ToolExecutor(
        tools={
            "lookup_policy": failing_tool,
            "lookup_inventory": _structured_tool(
                name="lookup_inventory",
                calls=inventory_calls,
                records=[{"sku": "sku-1"}],
            ),
        },
        allowed_tools={"lookup_policy", "lookup_inventory"},
    )
    plan = PlanRun(
        plan_run_id="plan-blocked",
        session_id="session-1",
        request_id="request-blocked",
        user_goal="先查规则再查库存",
        steps=[
            PlanStep(
                step_id="step-1",
                goal="查规则",
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
            PlanStep(
                step_id="step-3",
                goal="复核库存",
                tool_name="lookup_inventory",
                input={"query": "sku-1"},
                depends_on=["step-2"],
            ),
        ],
    )

    result = PlanExecutor(tool_executor=executor).execute(plan)

    assert result.workflow_status == "failed"
    assert [step.status for step in result.steps] == ["failed", "blocked", "blocked"]
    assert result.steps[1].metadata["blocked_by"] == ["step-1"]
    assert result.steps[2].metadata["blocked_by"] == ["step-2"]
    assert failing_tool.calls == [{"query": "return", "limit": 1}]
    assert inventory_calls == []


def test_plan_graph_finishes_when_no_step_is_executable() -> None:
    executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=[],
                records=[{"policy": "travel"}],
            )
        },
        allowed_tools={"lookup_policy"},
    )
    plan = PlanRun(
        plan_run_id="plan-no-executable-step",
        session_id="session-1",
        request_id="request-no-executable-step",
        user_goal="依赖缺失的计划",
        workflow_status="planning",
        steps=[
            PlanStep(
                step_id="step-1",
                goal="查制度",
                tool_name="lookup_policy",
                input={"query": "travel"},
                status="blocked",
            )
        ],
    )
    graph = build_plan_graph(
        PlanGraphDependencies(
            tool_executor=executor,
            project_result=lambda run: {
                "answer_mode": "fallback",
                "final_decision": "retrieval_failed",
            },
        )
    )

    result = graph.invoke({"plan_run": plan}, {"recursion_limit": 10})

    assert result["plan_run"].workflow_status == "failed"
    assert result["plan_run"].error == "Plan has pending steps but no executable dependency order."
    assert result["answer_mode"] == "fallback"
    assert result["final_decision"] == "retrieval_failed"


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
    assert synthesizer.contexts[0].observations == result.observations
    assert synthesizer.contexts[0].citations == [{"citation_id": "policy-1"}]
    assert synthesizer.contexts[0].context_summary == result.context_summary
    assert synthesizer.contexts[0].execution_order == ["step-1"]


def test_plan_model_calls_reuse_shared_guard_without_changing_plan_semantics() -> None:
    calls: list[dict[str, Any]] = []
    trace = RuntimeTraceMiddleware()
    guard = SharedModelCallGuard(trace=trace)
    selector = _RecordingStepSelector(
        [
            {
                "step_id": "guarded-step",
                "goal": "查制度",
                "tool_name": "lookup_policy",
                "input": {"query": "expense"},
            }
        ]
    )
    synthesizer = _RecordingPlanSynthesizer()
    tool_executor = ToolExecutor(
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

    plan = MinimalPlanner(
        tool_executor=tool_executor,
        step_selector=selector,
        model_call_guard=guard,
    ).create_plan(
        session_id="session-guard",
        request_id="request-guard",
        user_goal="查报销制度",
    )
    result = PlanExecutor(
        tool_executor=tool_executor,
        final_synthesizer=synthesizer,
        model_call_guard=guard,
    ).execute(plan)

    assert [step.step_id for step in result.steps] == ["guarded-step"]
    assert result.workflow_status == "succeeded"
    assert result.metadata["execution_order"] == ["guarded-step"]
    assert result.metadata["citations"] == [{"citation_id": "policy-1"}]
    assert [event.metadata["operation"] for event in trace.events] == [
        "plan.step_selection",
        "plan.final_synthesis",
    ]


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
    assert result.observations == [result.steps[0].observation]
    assert result.observations[0].metadata["hitl"] == result.metadata["hitl"]
    assert tool.calls == [{"query": "approval", "limit": 1}]


def test_plan_respond_continuation_resumes_waiting_step_with_user_context() -> None:
    executor = PlanExecutor(tool_executor=ToolExecutor(tools={}, allowed_tools=set()))
    plan = PlanRun(
        plan_run_id="plan-respond",
        session_id="session-respond",
        request_id="request-respond",
        user_goal="补充执行范围。",
        workflow_status="waiting_user",
        current_step_id="step-1",
        steps=[
            PlanStep(
                step_id="step-1",
                goal="补充执行范围。",
                tool_name="approval_tool",
                input={"query": "scope"},
                status="waiting_user",
                result_summary="需要补充执行范围。",
                metadata={
                    "hitl": {
                        "mode": "plan",
                        "plan_run_id": "plan-respond",
                        "current_step_id": "step-1",
                    }
                },
            )
        ],
        metadata={
            "hitl": {
                "mode": "plan",
                "plan_run_id": "plan-respond",
                "current_step_id": "step-1",
            }
        },
    )

    resumed = executor.continue_after_respond(
        run=plan,
        response="仅处理 2026 年制度。",
        source="freeform",
        metadata={"operator": "user-1"},
    )

    assert resumed is plan
    assert plan.workflow_status in {"running", "succeeded"}
    assert plan.steps[0].metadata["continuation"] == {
        "mode": "plan",
        "action": "respond",
        "plan_run_id": "plan-respond",
        "waiting_step_id": "step-1",
        "continued_from_step_id": "step-1",
        "metadata": {"operator": "user-1"},
        "response": "仅处理 2026 年制度。",
        "source": "freeform",
        "suggestion_id": None,
    }
    assert plan.metadata["resume"] == plan.steps[0].metadata["continuation"]


def test_plan_approve_continuation_executes_waiting_step_once() -> None:
    calls: list[dict[str, Any]] = []
    executor = PlanExecutor(
        tool_executor=ToolExecutor(
            tools={
                "approval_tool": _structured_tool(
                    name="approval_tool",
                    calls=calls,
                    records=[{"approved": True}],
                )
            },
            allowed_tools={"approval_tool"},
        )
    )
    plan = PlanRun(
        plan_run_id="plan-approve",
        session_id="session-approve",
        request_id="request-approve",
        user_goal="执行审批步骤。",
        workflow_status="waiting_user",
        current_step_id="step-1",
        steps=[
            PlanStep(
                step_id="step-1",
                goal="执行审批步骤。",
                tool_name="approval_tool",
                input={"query": "approved"},
                status="waiting_user",
            )
        ],
    )

    resumed = executor.continue_after_approve(
        run=plan,
        approval_result={"approved_by": "user-1"},
        pending_tool_call={"tool_name": "approval_tool", "args": {"query": "approved"}},
    )

    assert resumed is plan
    assert calls == [{"query": "approved", "limit": 1}]
    assert plan.steps[0].status == "succeeded"
    assert plan.observations[-1].tool_name == "approval_tool"
    assert plan.metadata["resume"]["action"] == "approve"
    assert plan.metadata["resume"]["pending_tool_call"] == {
        "tool_name": "approval_tool",
        "args": {"query": "approved"},
    }


def test_plan_reject_continuation_cancels_plan_and_skips_pending_side_effect() -> None:
    calls: list[dict[str, Any]] = []
    executor = PlanExecutor(
        tool_executor=ToolExecutor(
            tools={
                "external_call": _structured_tool(
                    name="external_call",
                    calls=calls,
                    records=[{"executed": True}],
                )
            },
            allowed_tools={"external_call"},
        )
    )
    plan = PlanRun(
        plan_run_id="plan-reject",
        session_id="session-reject",
        request_id="request-reject",
        user_goal="调用外部接口。",
        workflow_status="waiting_user",
        current_step_id="step-1",
        current_tool_call=ToolExecutionMetadata(
            tool_name="external_call",
            metadata={"args": {"query": "send"}},
        ),
        steps=[
            PlanStep(
                step_id="step-1",
                goal="调用外部接口。",
                tool_name="external_call",
                input={"query": "send"},
                status="waiting_user",
            )
        ],
    )

    cancelled = executor.continue_after_reject(
        run=plan,
        reason="用户拒绝外部调用。",
        pending_tool_call={"tool_name": "external_call", "args": {"query": "send"}},
    )

    assert cancelled is plan
    assert calls == []
    assert plan.workflow_status == "cancelled"
    assert plan.steps[0].status == "cancelled"
    assert plan.current_step_id is None
    assert plan.current_tool_call is None
    assert plan.metadata["resume"] == {
        "mode": "plan",
        "action": "reject",
        "plan_run_id": "plan-reject",
        "waiting_step_id": "step-1",
        "continued_from_step_id": "step-1",
        "metadata": {},
        "reason": "用户拒绝外部调用。",
        "pending_tool_call": {"tool_name": "external_call", "args": {"query": "send"}},
        "side_effect_executed": False,
    }


def test_plan_runtime_rejects_continue_for_stale_or_terminal_step() -> None:
    executor = PlanExecutor(tool_executor=ToolExecutor(tools={}, allowed_tools=set()))
    stale_plan = PlanRun(
        plan_run_id="plan-stale",
        session_id="session-stale",
        request_id="request-stale",
        user_goal="stale",
        workflow_status="waiting_user",
        current_step_id="step-current",
        steps=[
            PlanStep(
                step_id="step-waiting",
                goal="stale",
                tool_name="approval_tool",
                input={"query": "stale"},
                status="waiting_user",
            )
        ],
    )
    terminal_plan = PlanRun(
        plan_run_id="plan-terminal",
        session_id="session-terminal",
        request_id="request-terminal",
        user_goal="terminal",
        workflow_status="succeeded",
        steps=[],
    )

    with pytest.raises(ValueError, match="current_step_id"):
        executor.continue_after_respond(run=stale_plan, response="continue")
    with pytest.raises(ValueError, match="already terminal"):
        executor.continue_after_respond(run=terminal_plan, response="continue")


def test_minimal_planner_requires_explicit_default_retrieval_tool_without_name_guessing() -> None:
    executor = ToolExecutor(
        tools={
            "alpha_lookup": _structured_tool(
                name="alpha_lookup",
                calls=[],
                records=[{"record": "alpha"}],
            ),
            "legacy_search_alias": _structured_tool(
                name="legacy_search_alias",
                calls=[],
                records=[{"record": "legacy"}],
            ),
        },
        allowed_tools={"alpha_lookup", "legacy_search_alias"},
    )

    with pytest.raises(ValueError, match="AGENT_RUNTIME_TOOL_UNAVAILABLE"):
        MinimalPlanner(tool_executor=executor).create_plan(
            session_id="session-policy",
            request_id="request-policy",
            user_goal="需要检索但没有显式默认 retrieval tool。",
            mounted_knowledge_sources=("documents",),
            scene_policy={"candidate_tools": []},
        )


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

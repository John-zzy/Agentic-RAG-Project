from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langgraph.types import Command

from backend.platform.agent_runtime import (
    PlanDraft,
    PlanRun,
    PlanStep,
    PlanStepDraft,
    PlanSynthesisContext,
    PlanSynthesisResult,
    ToolExecutionMetadata,
    ToolObservation,
)
from backend.platform.agent_runtime.middleware import RuntimeTraceMiddleware, SharedModelCallGuard
from backend.platform.agent_runtime.plan.graph import build_plan_graph
from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.graph.resume import (
    PlanHitlResumeGraphDependencies,
    build_plan_hitl_resume_graph,
)
from backend.platform.agent_runtime.plan.planner import LangChainPlanPlanner
from backend.platform.agent_runtime.plan.state_ops import (
    continue_after_approve,
    continue_after_reject,
    continue_after_respond,
)
from backend.platform.agent_runtime.tooling.executor import ToolExecutor
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


class _FakeStructuredModel:
    def __init__(self, draft: PlanDraft | Exception) -> None:
        self.draft = draft
        self.messages: list[Any] = []

    def invoke(self, messages: Any) -> PlanDraft:
        self.messages.append(messages)
        if isinstance(self.draft, Exception):
            raise self.draft
        return self.draft


class _FakeChatModel:
    def __init__(self, draft: PlanDraft | Exception) -> None:
        self.structured = _FakeStructuredModel(draft)

    def with_structured_output(self, schema: Any) -> _FakeStructuredModel:
        assert schema is PlanDraft
        return self.structured


def test_plan_graph_uses_llm_planner_and_executes_single_step() -> None:
    calls: list[dict[str, Any]] = []
    tool_executor = ToolExecutor(
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
    graph = build_plan_graph(
        _deps(
            tool_executor=tool_executor,
            draft=PlanDraft(
                steps=[
                    PlanStepDraft(
                        goal="查差旅制度",
                        tool_name="lookup_policy",
                        input={"query": "差旅制度"},
                    )
                ],
                rationale_summary="single lookup",
            ),
        )
    )

    result = graph.invoke({}, {"recursion_limit": 10})
    run = result["plan_run"]

    assert run.workflow_status == "succeeded"
    assert run.metadata["planner"]["name"] == "langchain_plan_planner"
    assert run.metadata["planner"]["step_source"] == "llm_structured_output"
    assert run.steps[0].step_id == "step-1"
    assert run.steps[0].input == {"query": "差旅制度", "limit": 1}
    assert run.metadata["execution_order"] == ["step-1"]
    assert run.metadata["citations"] == [{"citation_id": "policy-1"}]
    assert calls == [{"query": "差旅制度", "limit": 1}]


def test_plan_graph_executes_multi_step_plan_in_dependency_order() -> None:
    policy_calls: list[dict[str, Any]] = []
    inventory_calls: list[dict[str, Any]] = []
    tool_executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=policy_calls,
                records=[{"policy": "return"}],
            ),
            "lookup_inventory": _structured_tool(
                name="lookup_inventory",
                calls=inventory_calls,
                records=[{"sku": "sku-1"}],
            ),
        },
        allowed_tools={"lookup_policy", "lookup_inventory"},
    )
    graph = build_plan_graph(
        _deps(
            tool_executor=tool_executor,
            draft=PlanDraft(
                steps=[
                    PlanStepDraft(
                        step_id="policy",
                        goal="查规则",
                        tool_name="lookup_policy",
                        input={"query": "return"},
                    ),
                    PlanStepDraft(
                        step_id="inventory",
                        goal="查库存",
                        tool_name="lookup_inventory",
                        input={"query": "sku-1"},
                        depends_on=["policy"],
                    ),
                ]
            ),
        )
    )

    run = graph.invoke({}, {"recursion_limit": 10})["plan_run"]

    assert run.workflow_status == "succeeded"
    assert run.metadata["execution_order"] == ["policy", "inventory"]
    assert [step.status for step in run.steps] == ["succeeded", "succeeded"]
    assert policy_calls == [{"query": "return", "limit": 1}]
    assert inventory_calls == [{"query": "sku-1", "limit": 1}]


def test_plan_graph_fails_when_llm_creates_too_many_steps() -> None:
    tool_executor = _single_tool_executor()
    graph = build_plan_graph(
        _deps(
            tool_executor=tool_executor,
            max_plan_steps=1,
            draft=PlanDraft(
                steps=[
                    PlanStepDraft(goal="one", tool_name="lookup_policy", input={"query": "one"}),
                    PlanStepDraft(goal="two", tool_name="lookup_policy", input={"query": "two"}),
                ]
            ),
        )
    )

    run = graph.invoke({}, {"recursion_limit": 10})["plan_run"]

    assert run.workflow_status == "failed"
    assert "max_plan_steps" in run.error
    assert run.metadata["planner"]["error"] == run.error


@pytest.mark.parametrize(
    ("draft", "error_text"),
    [
        (
            PlanDraft(steps=[PlanStepDraft(goal="bad", tool_name="unsafe", input={"query": "x"})]),
            "Tool is not allowed",
        ),
        (
            PlanDraft(steps=[PlanStepDraft(goal="bad", tool_name="lookup_policy", input={})]),
            "Invalid input",
        ),
        (
            PlanDraft(
                steps=[
                    PlanStepDraft(
                        step_id="step-1",
                        goal="bad",
                        tool_name="lookup_policy",
                        input={"query": "x"},
                        depends_on=["missing"],
                    )
                ]
            ),
            "Unknown dependencies",
        ),
        (
            PlanDraft(
                steps=[
                    PlanStepDraft(
                        step_id="step-1",
                        goal="bad",
                        tool_name="lookup_policy",
                        input={"query": "x"},
                        depends_on=["step-1"],
                    )
                ]
            ),
            "cannot depend on itself",
        ),
    ],
)
def test_plan_graph_fails_controlled_for_invalid_llm_plan(
    draft: PlanDraft,
    error_text: str,
) -> None:
    graph = build_plan_graph(_deps(tool_executor=_single_tool_executor(), draft=draft))

    run = graph.invoke({}, {"recursion_limit": 10})["plan_run"]

    assert run.workflow_status == "failed"
    assert error_text in run.error


def test_plan_graph_retries_retryable_error_and_then_succeeds() -> None:
    calls: list[dict[str, Any]] = []

    def flaky_lookup(query: str, limit: int = 1) -> ToolResult:
        calls.append({"query": query, "limit": limit})
        if len(calls) == 1:
            raise TimeoutError("temporary timeout")
        return ToolResult.ok(tool_name="lookup_policy", records=[{"policy": "ok"}])

    tool_executor = ToolExecutor(
        tools={"lookup_policy": _tool_from_func("lookup_policy", flaky_lookup)},
        allowed_tools={"lookup_policy"},
    )
    run = _invoke_existing_plan(tool_executor=tool_executor, plan=_single_step_plan())

    assert run.workflow_status == "succeeded"
    assert run.steps[0].retry_metadata.attempt == 2
    assert len(run.observations) == 2
    assert [transition["event"] for transition in run.metadata["workflow_transitions"]] == [
        "run_start",
        "tool_error_retryable",
        "retry",
        "success",
    ]


def test_plan_graph_blocks_steps_when_dependency_fails() -> None:
    failed_observation = ToolObservation(
        tool_name="lookup_policy",
        success=False,
        retryable=False,
        error="policy lookup failed",
        result_summary="policy lookup failed",
    )
    failing_tool = _StaticObservationTool(name="lookup_policy", observation=failed_observation)
    inventory_calls: list[dict[str, Any]] = []
    tool_executor = ToolExecutor(
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
        ],
    )

    run = _invoke_existing_plan(tool_executor=tool_executor, plan=plan)

    assert run.workflow_status == "failed"
    assert [step.status for step in run.steps] == ["failed", "blocked"]
    assert run.steps[1].metadata["blocked_by"] == ["step-1"]
    assert inventory_calls == []


def test_plan_graph_waiting_user_records_hitl_metadata() -> None:
    observation = ToolObservation(
        tool_name="approval_tool",
        success=False,
        requires_user=True,
        user_prompt="是否批准执行该步骤？",
        result_summary="等待用户批准。",
    )
    tool = _StaticObservationTool(name="approval_tool", observation=observation)
    tool_executor = ToolExecutor(tools={"approval_tool": tool}, allowed_tools={"approval_tool"})
    plan = _single_step_plan(tool_name="approval_tool", query="approval")

    run = _invoke_existing_plan(tool_executor=tool_executor, plan=plan)

    assert run.workflow_status == "waiting_user"
    assert run.current_step_id == "step-1"
    assert run.steps[0].status == "waiting_user"
    assert run.metadata["hitl"] == {
        "mode": "plan",
        "plan_run_id": "plan-1",
        "current_step_id": "step-1",
        "user_prompt": "是否批准执行该步骤？",
        "source": "tool_observation",
    }
    assert run.steps[0].observation.metadata["hitl"] == run.metadata["hitl"]


def test_plan_final_synthesis_uses_successful_step_summaries_and_citations() -> None:
    calls: list[dict[str, Any]] = []
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

    run = _invoke_existing_plan(
        tool_executor=tool_executor,
        plan=_single_step_plan(query="expense"),
        final_synthesizer=synthesizer,
    )

    assert run.workflow_status == "succeeded"
    assert run.result_summary == "finalized 1 step(s)"
    assert run.metadata["citations"] == [{"citation_id": "policy-1"}]
    assert run.metadata["knowledge_used"] is True
    assert synthesizer.contexts[0].steps == run.steps
    assert synthesizer.contexts[0].observations == run.observations


def test_plan_model_calls_reuse_shared_guard_for_planning_and_synthesis() -> None:
    calls: list[dict[str, Any]] = []
    trace = RuntimeTraceMiddleware()
    guard = SharedModelCallGuard(trace=trace)
    tool_executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=calls,
                records=[{"policy": "expense"}],
            )
        },
        allowed_tools={"lookup_policy"},
    )
    graph = build_plan_graph(
        _deps(
            tool_executor=tool_executor,
            model_call_guard=guard,
            final_synthesizer=_RecordingPlanSynthesizer(),
            draft=PlanDraft(
                steps=[
                    PlanStepDraft(
                        goal="查制度",
                        tool_name="lookup_policy",
                        input={"query": "expense"},
                    )
                ]
            ),
        )
    )

    run = graph.invoke({}, {"recursion_limit": 10})["plan_run"]

    assert run.workflow_status == "succeeded"
    assert [event.metadata["operation"] for event in trace.events] == [
        "plan.create_plan",
        "plan.final_synthesis",
    ]


def test_plan_continuation_helpers_keep_existing_hitl_semantics() -> None:
    calls: list[dict[str, Any]] = []
    tool_executor = ToolExecutor(
        tools={
            "approval_tool": _structured_tool(
                name="approval_tool",
                calls=calls,
                records=[{"approved": True}],
            )
        },
        allowed_tools={"approval_tool"},
    )
    respond_plan = _waiting_plan()

    continued = continue_after_respond(
        run=respond_plan,
        response="仅处理 2026 年制度。",
        source="freeform",
        metadata={"operator": "user-1"},
    )

    assert continued is respond_plan
    assert respond_plan.workflow_status == "running"
    assert respond_plan.steps[0].input["human_response"] == "仅处理 2026 年制度。"
    assert respond_plan.metadata["resume"]["action"] == "respond"

    approve_plan = _waiting_plan()
    approved = continue_after_approve(
        run=approve_plan,
        tool_executor=tool_executor,
        approval_result={"approved_by": "user-1"},
        pending_tool_call={"tool_name": "approval_tool", "args": {"query": "approved"}},
    )
    assert approved.workflow_status == "running"
    assert approved.steps[0].status == "succeeded"
    assert approved.metadata["execution_order"] == ["step-1"]
    assert calls == [{"query": "approved", "limit": 1}]

    reject_plan = _waiting_plan(tool_name="external_call")
    cancelled = continue_after_reject(
        run=reject_plan,
        reason="用户拒绝外部调用。",
        pending_tool_call={"tool_name": "external_call", "args": {"query": "send"}},
    )
    assert cancelled.workflow_status == "cancelled"
    assert cancelled.steps[0].status == "cancelled"
    assert cancelled.metadata["resume"]["side_effect_executed"] is False


def test_plan_hitl_approve_continues_remaining_steps() -> None:
    approval_calls: list[dict[str, Any]] = []
    follow_up_calls: list[dict[str, Any]] = []
    tool_executor = ToolExecutor(
        tools={
            "approval_tool": _structured_tool(
                name="approval_tool",
                calls=approval_calls,
                records=[{"approved": True}],
            ),
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=follow_up_calls,
                records=[{"policy": "approved"}],
            ),
        },
    )
    waiting_plan = _waiting_plan()
    waiting_plan.steps.append(
        PlanStep(
            step_id="step-2",
            goal="审批后继续检索",
            tool_name="lookup_policy",
            input={"query": "after approve"},
            depends_on=["step-1"],
        )
    )
    graph_dependencies = _deps(
        tool_executor=tool_executor,
        draft=PlanDraft(steps=[]),
        final_synthesizer=_RecordingPlanSynthesizer(),
    )
    graph = build_plan_hitl_resume_graph(
        PlanHitlResumeGraphDependencies(
            graph_dependencies=graph_dependencies,
            tool_executor=tool_executor,
        )
    )

    result = graph.invoke(
        {
            "resume_command": Command(
                resume={
                    "plan_run": waiting_plan.model_dump(),
                    "resume_payload": {"action": "approve"},
                    "proposed_tool_call": {
                        "tool_name": "approval_tool",
                        "args": {"query": "approved"},
                    },
                }
            )
        },
        {"recursion_limit": 10},
    )
    run = result["plan_run"]

    assert approval_calls == [{"query": "approved", "limit": 1}]
    assert follow_up_calls == [{"query": "after approve", "limit": 1}]
    assert run.workflow_status == "succeeded"
    assert [step.status for step in run.steps] == ["succeeded", "succeeded"]
    assert run.metadata["execution_order"] == ["step-1", "step-2"]
    assert run.metadata["resume"]["side_effect_executed"] is True


def test_plan_runtime_rejects_continue_for_stale_or_terminal_step() -> None:
    stale_plan = _waiting_plan()
    stale_plan.current_step_id = "missing"
    terminal_plan = PlanRun(
        plan_run_id="plan-terminal",
        session_id="session-terminal",
        request_id="request-terminal",
        user_goal="terminal",
        workflow_status="succeeded",
        steps=[],
    )

    with pytest.raises(ValueError, match="current_step_id"):
        continue_after_respond(run=stale_plan, response="continue")
    with pytest.raises(ValueError, match="already terminal"):
        continue_after_respond(run=terminal_plan, response="continue")


def _deps(
    *,
    tool_executor: ToolExecutor,
    draft: PlanDraft | Exception,
    max_plan_steps: int = 8,
    final_synthesizer: Any | None = None,
    model_call_guard: SharedModelCallGuard | None = None,
) -> PlanGraphDependencies:
    planner = LangChainPlanPlanner(
        model_provider=lambda complexity="moderate": _FakeChatModel(draft),
        tool_executor=tool_executor,
        model_call_guard=model_call_guard,
        plan_run_id_factory=lambda: "plan-1",
        step_id_factory=lambda index: f"step-{index}",
    )
    return PlanGraphDependencies(
        tool_executor=tool_executor,
        planner=planner,
        session_id="session-1",
        request_id="request-1",
        user_goal="查制度",
        mounted_knowledge_sources=("documents",),
        candidate_tools=tuple(tool_executor.allowed_tools),
        default_tool_inputs={
            tool_name: {"query": "查制度"}
            for tool_name in tool_executor.allowed_tools
        },
        max_plan_steps=max_plan_steps,
        final_synthesizer=final_synthesizer,
        model_call_guard=model_call_guard,
    )


def _invoke_existing_plan(
    *,
    tool_executor: ToolExecutor,
    plan: PlanRun,
    final_synthesizer: Any | None = None,
) -> PlanRun:
    graph = build_plan_graph(
        _deps(
            tool_executor=tool_executor,
            draft=PlanDraft(steps=[]),
            final_synthesizer=final_synthesizer,
        )
    )
    return graph.invoke({"plan_run": plan}, {"recursion_limit": 10})["plan_run"]


def _single_tool_executor() -> ToolExecutor:
    return ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=[],
                records=[{"policy": "travel"}],
            )
        },
        allowed_tools={"lookup_policy"},
    )


def _single_step_plan(*, tool_name: str = "lookup_policy", query: str = "policy") -> PlanRun:
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


def _waiting_plan(*, tool_name: str = "approval_tool") -> PlanRun:
    return PlanRun(
        plan_run_id="plan-wait",
        session_id="session-wait",
        request_id="request-wait",
        user_goal="等待用户",
        workflow_status="waiting_user",
        current_step_id="step-1",
        current_tool_call=ToolExecutionMetadata(tool_name=tool_name),
        steps=[
            PlanStep(
                step_id="step-1",
                goal="等待用户",
                tool_name=tool_name,
                input={"query": "scope"},
                status="waiting_user",
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

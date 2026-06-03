from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import Field

from backend.platform.agent_runtime.contracts import (
    AgentRuntimeModel,
    PlanRun,
    PlanStep,
    ToolObservation,
    collect_successful_tool_observations,
)
from backend.platform.agent_runtime.tool_executor import ToolExecutor
from backend.platform.agent_runtime.validation import (
    validate_plan_dependencies,
    validate_plan_tool_allowlist,
)
from backend.platform.workflow.state_machine import validate_transition


class PlanSynthesisContext(AgentRuntimeModel):
    """Plan final synthesizer 的输入，只包含已完成步骤和工具观察。"""

    plan_run_id: str
    session_id: str
    request_id: str
    user_goal: str
    context_summary: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    observations: list[ToolObservation] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanSynthesisResult(AgentRuntimeModel):
    """PlanExecutor 写回 PlanRun 的最终汇总结果。"""

    final_answer: str
    result_summary: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_used: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanFinalSynthesizer(Protocol):
    """从成功 PlanStep 汇总最终回答的中立协议。"""

    def synthesize(self, context: PlanSynthesisContext) -> PlanSynthesisResult:
        """Return the final answer for a completed PlanRun."""


class StepSummarySynthesizer:
    """默认汇总器：按成功 step 的 result_summary 生成最终回答。"""

    def synthesize(self, context: PlanSynthesisContext) -> PlanSynthesisResult:
        summaries = [
            step.result_summary or f"{step.step_id} succeeded."
            for step in context.steps
            if step.status == "succeeded"
        ]
        final_answer = "\n".join(summaries) if summaries else "No successful plan steps were collected."
        citations = _deduplicate_citations(context.citations)
        return PlanSynthesisResult(
            final_answer=final_answer,
            result_summary=f"Synthesized {len(summaries)} successful plan step(s).",
            citations=citations,
            knowledge_used=bool(citations),
            metadata={"step_count": len(context.steps)},
        )


PlanSummarySynthesizer = StepSummarySynthesizer


class PlanExecutor:
    """按依赖顺序串行执行 PlanStep，并维护 run-level workflow 状态。"""

    def __init__(
        self,
        *,
        tool_executor: ToolExecutor,
        final_synthesizer: PlanFinalSynthesizer | None = None,
    ) -> None:
        self._tool_executor = tool_executor
        self._final_synthesizer = final_synthesizer or StepSummarySynthesizer()

    def run(self, plan_run: PlanRun) -> PlanRun:
        self._validate_plan(plan_run.steps)
        if plan_run.workflow_status == "planning":
            _transition(plan_run, "run_start")
        elif plan_run.workflow_status == "created":
            _transition(plan_run, "plan_start")
            _transition(plan_run, "run_start")

        while True:
            _mark_steps_blocked_by_unavailable_dependencies(plan_run.steps)

            eligible_step = _next_eligible_step(plan_run.steps)
            if eligible_step is None:
                if _all_steps_succeeded(plan_run.steps):
                    return self._synthesize_success(plan_run)
                if _has_blocked_steps(plan_run.steps):
                    return self._mark_failed(
                        plan_run=plan_run,
                        error="Plan has blocked steps because required dependencies did not complete.",
                    )
                return self._mark_failed(
                    plan_run=plan_run,
                    error="Plan has pending steps but no executable dependency order.",
                )

            self._execute_step(plan_run=plan_run, step=eligible_step)
            if plan_run.workflow_status in {"waiting_user", "failed", "cancelled"}:
                _mark_steps_blocked_by_unavailable_dependencies(plan_run.steps)
                return plan_run

    def execute(self, plan_run: PlanRun) -> PlanRun:
        """按 task 语义暴露执行入口；内部复用 run 以保持单一流程。"""
        return self.run(plan_run)

    def _validate_plan(self, steps: Sequence[PlanStep]) -> None:
        validate_plan_tool_allowlist(steps, self._tool_executor.allowed_tools)
        validate_plan_dependencies(steps)
        for step in steps:
            self._tool_executor.validate_call(
                tool_name=step.tool_name,
                input_payload=step.input,
            )

    def _execute_step(self, *, plan_run: PlanRun, step: PlanStep) -> None:
        while True:
            step.status = "running"
            _ensure_plan_running(plan_run)
            plan_run.current_step_id = step.step_id

            observation = self._tool_executor.execute(
                tool_name=step.tool_name,
                input_payload=step.input,
                attempt=step.retry_metadata.attempt,
                max_attempts=step.retry_metadata.max_attempts,
            )
            observation = _prepare_observation_for_step(
                plan_run=plan_run,
                step=step,
                observation=observation,
            )
            _persist_step_observation(
                plan_run=plan_run,
                step=step,
                observation=observation,
            )
            step.retry_metadata = step.retry_metadata.model_copy(
                update={
                    "attempt": step.retry_metadata.attempt + 1,
                    "retryable": observation.retryable,
                    "last_error": observation.error,
                }
            )

            if observation.requires_user:
                _mark_step_waiting_on_observation(
                    plan_run=plan_run,
                    step=step,
                    observation=observation,
                )
                _transition(plan_run, "interrupt")
                return

            if observation.success:
                step.status = "succeeded"
                step.error = None
                plan_run.current_step_id = None
                plan_run.current_tool_call = None
                _append_execution_order(plan_run=plan_run, step_id=step.step_id)
                return

            if not observation.retryable:
                step.status = "failed"
                self._mark_failed(
                    plan_run=plan_run,
                    error=_observation_error(observation),
                )
                return

            step.status = "retrying"
            _transition(plan_run, "tool_error_retryable")
            plan_run.error = _observation_error(observation)

            # retrying 是 run 级准备态；还有预算时立即回到 running 再重试同一步。
            if step.retry_metadata.attempt >= step.retry_metadata.max_attempts:
                step.status = "failed"
                self._mark_failed(
                    plan_run=plan_run,
                    error=_observation_error(observation),
                )
                return
            _transition(plan_run, "retry")

    def _synthesize_success(self, plan_run: PlanRun) -> PlanRun:
        observations = collect_successful_tool_observations(plan_run)
        citations = _collect_citations(observations)
        result = self._final_synthesizer.synthesize(
            PlanSynthesisContext(
                plan_run_id=plan_run.plan_run_id,
                session_id=plan_run.session_id,
                request_id=plan_run.request_id,
                user_goal=plan_run.user_goal,
                context_summary=plan_run.context_summary,
                steps=list(plan_run.steps),
                observations=observations,
                citations=citations,
                execution_order=list(plan_run.metadata.get("execution_order") or []),
                metadata={
                    "planner": dict(plan_run.metadata.get("planner") or {}),
                    "context_summary": plan_run.context_summary,
                },
            )
        )
        _transition(plan_run, "success")
        plan_run.final_answer = result.final_answer
        plan_run.result_summary = result.result_summary
        plan_run.error = None
        plan_run.current_step_id = None
        plan_run.current_tool_call = None
        plan_run.metadata["citations"] = result.citations
        plan_run.metadata["knowledge_used"] = result.knowledge_used
        plan_run.metadata["final_synthesis"] = result.metadata
        return plan_run

    def _mark_failed(self, *, plan_run: PlanRun, error: str) -> PlanRun:
        if plan_run.workflow_status == "retrying":
            _transition(plan_run, "tool_error_final")
        elif plan_run.workflow_status not in {"failed", "cancelled"}:
            _transition(plan_run, "tool_error_final")
        plan_run.error = error
        plan_run.result_summary = error
        plan_run.current_tool_call = None
        return plan_run


def _transition(plan_run: PlanRun, event: str) -> None:
    previous_status = plan_run.workflow_status
    next_status = validate_transition(previous_status, event)
    plan_run.workflow_status = next_status
    transitions = list(plan_run.metadata.get("workflow_transitions") or [])
    transitions.append(
        {
            "from": previous_status,
            "event": event,
            "to": next_status,
        }
    )
    plan_run.metadata["workflow_transitions"] = transitions


def _ensure_plan_running(plan_run: PlanRun) -> None:
    """Plan run 只在合法状态下进入 step 执行。"""
    if plan_run.workflow_status == "running":
        return
    if plan_run.workflow_status == "planning":
        _transition(plan_run, "run_start")
        return
    if plan_run.workflow_status == "created":
        _transition(plan_run, "plan_start")
        _transition(plan_run, "run_start")
        return
    raise ValueError(f"Plan run cannot execute from state: {plan_run.workflow_status}.")


def _next_eligible_step(steps: Sequence[PlanStep]) -> PlanStep | None:
    step_by_id = {step.step_id: step for step in steps}
    for step in steps:
        if step.status != "pending":
            continue
        if all(step_by_id[dependency_id].status == "succeeded" for dependency_id in step.depends_on):
            return step
    return None


def _mark_steps_blocked_by_unavailable_dependencies(steps: Sequence[PlanStep]) -> None:
    while True:
        blocked_any = False
        step_by_id = {step.step_id: step for step in steps}
        for step in steps:
            if step.status != "pending":
                continue
            blockers = _blocking_dependency_ids(step=step, step_by_id=step_by_id)
            if not blockers:
                continue
            _mark_step_blocked(step=step, blockers=blockers)
            blocked_any = True
        if not blocked_any:
            return


def _blocking_dependency_ids(
    *,
    step: PlanStep,
    step_by_id: Mapping[str, PlanStep],
) -> list[str]:
    blockers: list[str] = []
    for dependency_id in step.depends_on:
        dependency = step_by_id[dependency_id]
        if dependency.status in {"failed", "cancelled", "blocked"}:
            blockers.append(dependency_id)
    return blockers


def _mark_step_blocked(*, step: PlanStep, blockers: Sequence[str]) -> None:
    blocked_by = list(blockers)
    message = (
        f"Plan step {step.step_id} is blocked by dependency "
        f"{', '.join(blocked_by)}."
    )
    step.status = "blocked"
    step.error = message
    step.result_summary = message
    step.metadata["blocked_by"] = blocked_by


def _has_blocked_steps(steps: Sequence[PlanStep]) -> bool:
    return any(step.status == "blocked" for step in steps)


def _all_steps_succeeded(steps: Sequence[PlanStep]) -> bool:
    return bool(steps) and all(step.status == "succeeded" for step in steps)


def _prepare_observation_for_step(
    *,
    plan_run: PlanRun,
    step: PlanStep,
    observation: ToolObservation,
) -> ToolObservation:
    if not observation.requires_user:
        return observation
    return _attach_observation_hitl_metadata(
        observation=observation,
        hitl_metadata=_build_plan_hitl_metadata(
            plan_run=plan_run,
            step=step,
            user_prompt=observation.user_prompt or observation.result_summary,
            source="tool_observation",
        ),
    )


def _persist_step_observation(
    *,
    plan_run: PlanRun,
    step: PlanStep,
    observation: ToolObservation,
) -> None:
    step.observation = observation
    step.output = observation.output
    step.result_summary = observation.result_summary
    step.error = observation.error
    plan_run.observations.append(observation)
    plan_run.current_tool_call = observation.execution


def _mark_step_waiting_on_observation(
    *,
    plan_run: PlanRun,
    step: PlanStep,
    observation: ToolObservation,
) -> None:
    hitl_metadata = dict(observation.metadata.get("hitl") or {})
    step.status = "waiting_user"
    step.metadata["hitl"] = hitl_metadata
    plan_run.metadata["hitl"] = hitl_metadata


def _append_execution_order(*, plan_run: PlanRun, step_id: str) -> None:
    execution_order = list(plan_run.metadata.get("execution_order") or [])
    execution_order.append(step_id)
    plan_run.metadata["execution_order"] = execution_order


def _collect_citations(observations: Sequence[ToolObservation]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for observation in observations:
        if observation.success:
            citations.extend(observation.citations)
    return _deduplicate_citations(citations)


def _deduplicate_citations(citations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for citation in citations:
        citation_id = str(citation.get("citation_id") or citation)
        if citation_id in seen:
            continue
        seen.add(citation_id)
        deduplicated.append(dict(citation))
    return deduplicated


def _attach_observation_hitl_metadata(
    *,
    observation: ToolObservation,
    hitl_metadata: dict[str, Any],
) -> ToolObservation:
    metadata = dict(observation.metadata)
    metadata["hitl"] = dict(hitl_metadata)
    return observation.model_copy(update={"metadata": metadata})


def _build_plan_hitl_metadata(
    *,
    plan_run: PlanRun,
    step: PlanStep,
    user_prompt: str,
    source: str,
) -> dict[str, Any]:
    return {
        "mode": "plan",
        "plan_run_id": plan_run.plan_run_id,
        "current_step_id": step.step_id,
        "user_prompt": user_prompt,
        "source": source,
    }


def _observation_error(observation: ToolObservation) -> str:
    return observation.error or observation.result_summary

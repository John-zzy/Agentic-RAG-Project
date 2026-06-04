from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from backend.platform.agent_runtime.contracts import (
    PlanRun,
    PlanStep,
    ToolObservation,
    collect_successful_tool_observations,
)
from backend.platform.agent_runtime.plan.synthesis import (
    PlanFinalSynthesizer,
    PlanSynthesisContext,
    PlanSynthesisResult,
    StepSummarySynthesizer,
)
from backend.platform.agent_runtime.tool_executor import ToolExecutor
from backend.platform.agent_runtime.validation import (
    validate_plan_dependencies,
    validate_plan_tool_allowlist,
)
from backend.platform.workflow.state_machine import validate_transition


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
        self.validate_plan(plan_run.steps)
        if plan_run.workflow_status == "planning":
            _transition(plan_run, "run_start")
        elif plan_run.workflow_status == "created":
            _transition(plan_run, "plan_start")
            _transition(plan_run, "run_start")

        while True:
            _mark_steps_blocked_by_unavailable_dependencies(plan_run.steps)

            eligible_step = self.select_next_step(plan_run.steps)
            if eligible_step is None:
                if _all_steps_succeeded(plan_run.steps):
                    return self.synthesize_plan_result(plan_run)
                if _has_blocked_steps(plan_run.steps):
                    return self.mark_failed(
                        plan_run=plan_run,
                        error="Plan has blocked steps because required dependencies did not complete.",
                    )
                return self.mark_failed(
                    plan_run=plan_run,
                    error="Plan has pending steps but no executable dependency order.",
                )

            self.execute_step(plan_run=plan_run, step=eligible_step)
            if plan_run.workflow_status in {"waiting_user", "failed", "cancelled"}:
                _mark_steps_blocked_by_unavailable_dependencies(plan_run.steps)
                return plan_run

    def execute(self, plan_run: PlanRun) -> PlanRun:
        """按 task 语义暴露执行入口；内部复用 run 以保持单一流程。"""
        return self.run(plan_run)

    def validate_plan(self, steps: Sequence[PlanStep]) -> None:
        validate_plan_tool_allowlist(steps, self._tool_executor.allowed_tools)
        validate_plan_dependencies(steps)
        for step in steps:
            self._tool_executor.validate_call(
                tool_name=step.tool_name,
                input_payload=step.input,
            )

    def select_next_step(self, steps: Sequence[PlanStep]) -> PlanStep | None:
        """公开下一步选择边界，供 graph 节点直接复用。"""
        return _next_eligible_step(steps)

    def execute_step(self, *, plan_run: PlanRun, step: PlanStep) -> None:
        """执行单个 step，并在当前调用内完成 retry / HITL / 失败收口。"""
        while True:
            outcome = self.execute_step_once(plan_run=plan_run, step=step)
            if outcome != "retry":
                return

    def execute_step_once(self, *, plan_run: PlanRun, step: PlanStep) -> str | None:
        """执行一次 step；graph 节点使用这个入口，避免把 retry 绑死在节点内部。"""
        step.status = "running"
        self.ensure_running(plan_run)
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
            self.handle_waiting_user(
                plan_run=plan_run,
                step=step,
                observation=observation,
            )
            _transition(plan_run, "interrupt")
            return None

        if observation.success:
            step.status = "succeeded"
            step.error = None
            plan_run.current_step_id = None
            plan_run.current_tool_call = None
            _append_execution_order(plan_run=plan_run, step_id=step.step_id)
            return None

        if not observation.retryable:
            step.status = "failed"
            self.mark_failed(
                plan_run=plan_run,
                error=_observation_error(observation),
            )
            return None

        step.status = "retrying"
        _transition(plan_run, "tool_error_retryable")
        plan_run.error = _observation_error(observation)

        # retrying 是 run 级准备态；下一轮由图的 handle_retry 节点决定是否回到 pending。
        if step.retry_metadata.attempt >= step.retry_metadata.max_attempts:
            step.status = "failed"
            self.mark_failed(
                plan_run=plan_run,
                error=_observation_error(observation),
            )
            return None
        _transition(plan_run, "retry")
        return "retry"

    def synthesize_plan_result(self, plan_run: PlanRun) -> PlanRun:
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

    def mark_failed(self, *, plan_run: PlanRun, error: str) -> PlanRun:
        if plan_run.workflow_status == "retrying":
            _transition(plan_run, "tool_error_final")
        elif plan_run.workflow_status not in {"failed", "cancelled"}:
            _transition(plan_run, "tool_error_final")
        plan_run.error = error
        plan_run.result_summary = error
        plan_run.current_tool_call = None
        return plan_run

    def ensure_running(self, plan_run: PlanRun) -> None:
        """公开 run 状态守卫，避免 graph 节点直接触碰内部函数。"""
        _ensure_plan_running(plan_run)

    def handle_waiting_user(
        self,
        *,
        plan_run: PlanRun,
        step: PlanStep,
        observation: ToolObservation,
    ) -> None:
        """公开 waiting_user 边界；当前仅负责保留 step 级 HITL 事实。"""
        _mark_step_waiting_on_observation(
            plan_run=plan_run,
            step=step,
            observation=observation,
        )

    def handle_retry(self, *, plan_run: PlanRun, step: PlanStep) -> PlanRun:
        """公开 retry 边界；图节点可在此处继续保留统一的失败收口。"""
        if step.status == "retrying" and step.retry_metadata.attempt < step.retry_metadata.max_attempts:
            step.status = "pending"
            return plan_run
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


def _blocking_dependency_ids(*, step: PlanStep, step_by_id: Mapping[str, PlanStep]) -> list[str]:
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

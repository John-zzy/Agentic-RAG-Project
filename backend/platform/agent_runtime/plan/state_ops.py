from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from backend.platform.agent_runtime.core.contracts import (
    PlanRun,
    PlanStep,
    ToolObservation,
    collect_successful_tool_observations,
)
from backend.platform.agent_runtime.middleware.model_call import (
    SharedModelCallGuard,
    default_model_call_context,
)
from backend.platform.agent_runtime.plan.synthesis import (
    PlanFinalSynthesizer,
    PlanSynthesisContext,
    PlanSynthesisResult,
    StepSummarySynthesizer,
)
from backend.platform.agent_runtime.tooling.executor import ToolExecutor
from backend.platform.agent_runtime.tooling.idempotency import ToolExecutionContext
from backend.platform.workflow.state_machine import validate_transition


def transition(plan_run: PlanRun, event: str) -> None:
    previous_status = plan_run.workflow_status
    next_status = validate_transition(previous_status, event)
    plan_run.workflow_status = next_status
    transitions = list(plan_run.metadata.get("workflow_transitions") or [])
    transitions.append({"from": previous_status, "event": event, "to": next_status})
    plan_run.metadata["workflow_transitions"] = transitions


def mark_failed(*, plan_run: PlanRun, error: str) -> PlanRun:
    if plan_run.workflow_status not in {"failed", "cancelled"}:
        transition(plan_run, "tool_error_final")
    plan_run.error = error
    plan_run.result_summary = error
    plan_run.current_tool_call = None
    return plan_run


def ensure_running(plan_run: PlanRun) -> None:
    if plan_run.workflow_status == "running":
        return
    if plan_run.workflow_status == "planning":
        transition(plan_run, "run_start")
        return
    if plan_run.workflow_status == "created":
        transition(plan_run, "plan_start")
        transition(plan_run, "run_start")
        return
    raise ValueError(f"Plan run cannot execute from state: {plan_run.workflow_status}.")


def select_next_step(steps: Sequence[PlanStep]) -> PlanStep | None:
    step_by_id = {step.step_id: step for step in steps}
    for step in steps:
        if step.status != "pending":
            continue
        if all(step_by_id[dependency_id].status == "succeeded" for dependency_id in step.depends_on):
            return step
    return None


def mark_steps_blocked_by_unavailable_dependencies(steps: Sequence[PlanStep]) -> None:
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


def all_steps_succeeded(steps: Sequence[PlanStep]) -> bool:
    return bool(steps) and all(step.status == "succeeded" for step in steps)


def has_blocked_steps(steps: Sequence[PlanStep]) -> bool:
    return any(step.status == "blocked" for step in steps)


def execute_step_once(
    *,
    plan_run: PlanRun,
    step: PlanStep,
    tool_executor: ToolExecutor,
) -> None:
    step.status = "running"
    ensure_running(plan_run)
    plan_run.current_step_id = step.step_id
    observation = tool_executor.execute(
        tool_name=step.tool_name,
        input_payload=step.input,
        attempt=step.retry_metadata.attempt,
        max_attempts=step.retry_metadata.max_attempts,
        execution_context=build_step_execution_context(
            plan_run=plan_run,
            step=step,
            node_name="plan.execute_step",
        ),
    )
    observation = prepare_observation_for_step(
        plan_run=plan_run,
        step=step,
        observation=observation,
    )
    persist_step_observation(plan_run=plan_run, step=step, observation=observation)
    step.retry_metadata = step.retry_metadata.model_copy(
        update={
            "attempt": step.retry_metadata.attempt + 1,
            "retryable": observation.retryable,
            "last_error": observation.error,
        }
    )
    if observation.requires_user:
        mark_step_waiting_on_observation(plan_run=plan_run, step=step, observation=observation)
        transition(plan_run, "interrupt")
        return
    if observation.success:
        step.status = "succeeded"
        step.error = None
        plan_run.current_step_id = None
        plan_run.current_tool_call = None
        append_execution_order(plan_run=plan_run, step_id=step.step_id)
        return
    if not observation.retryable:
        step.status = "failed"
        mark_failed(plan_run=plan_run, error=observation_error(observation))
        return
    step.status = "retrying"
    transition(plan_run, "tool_error_retryable")
    plan_run.error = observation_error(observation)
    if step.retry_metadata.attempt >= step.retry_metadata.max_attempts:
        step.status = "failed"
        mark_failed(plan_run=plan_run, error=observation_error(observation))
        return
    transition(plan_run, "retry")


def prepare_retry_step(*, plan_run: PlanRun, step: PlanStep) -> PlanRun:
    if step.status == "retrying" and step.retry_metadata.attempt < step.retry_metadata.max_attempts:
        step.status = "pending"
    return plan_run


def synthesize_plan_result(
    *,
    plan_run: PlanRun,
    final_synthesizer: PlanFinalSynthesizer | None = None,
    model_call_guard: SharedModelCallGuard | None = None,
) -> PlanRun:
    synthesizer = final_synthesizer or StepSummarySynthesizer()
    observations = collect_successful_tool_observations(plan_run)
    citations = collect_citations(observations)
    context = PlanSynthesisContext(
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
    result = _synthesize_with_guard(
        plan_run=plan_run,
        context=context,
        synthesizer=synthesizer,
        model_call_guard=model_call_guard,
    )
    transition(plan_run, "success")
    plan_run.final_answer = result.final_answer
    plan_run.result_summary = result.result_summary
    plan_run.error = None
    plan_run.current_step_id = None
    plan_run.current_tool_call = None
    plan_run.metadata["citations"] = result.citations
    plan_run.metadata["knowledge_used"] = result.knowledge_used
    plan_run.metadata["final_synthesis"] = result.metadata
    return plan_run


def continue_after_respond(
    *,
    run: PlanRun,
    response: str,
    source: str = "freeform",
    suggestion_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PlanRun:
    step = require_waiting_step(run)
    continuation = record_plan_continuation(
        plan_run=run,
        waiting_step=step,
        action="respond",
        metadata=metadata,
        extra={
            "response": response,
            "source": source,
            "suggestion_id": suggestion_id,
        },
    )
    step.input = {**dict(step.input), "human_response": response}
    step.status = "pending"
    run.workflow_status = validate_transition(run.workflow_status, "resume_respond")
    run.current_step_id = None
    run.current_tool_call = None
    run.metadata["resume"] = continuation
    return run


def continue_after_approve(
    *,
    run: PlanRun,
    tool_executor: ToolExecutor,
    approval_result: Mapping[str, Any] | None = None,
    pending_tool_call: Mapping[str, Any] | None = None,
) -> PlanRun:
    step = require_waiting_step(run)
    tool_call = dict(pending_tool_call or {})
    tool_name = str(tool_call.get("tool_name") or step.tool_name)
    args = tool_call.get("args")
    input_payload = dict(args) if isinstance(args, Mapping) else dict(step.input)
    continuation = record_plan_continuation(
        plan_run=run,
        waiting_step=step,
        action="approve",
        metadata=approval_result,
        extra={
            "pending_tool_call": dict(pending_tool_call or {}),
            "side_effect_executed": True,
        },
    )
    run.workflow_status = validate_transition(run.workflow_status, "resume_approve")
    observation = tool_executor.execute(
        tool_name=tool_name,
        input_payload=input_payload,
        execution_context=build_step_execution_context(
            plan_run=run,
            step=step,
            node_name="plan.resume_approve",
        ),
    )
    persist_step_observation(plan_run=run, step=step, observation=observation)
    step.status = "succeeded" if observation.success else "failed"
    run.current_step_id = None
    run.current_tool_call = None
    run.metadata["resume"] = continuation
    if observation.success:
        append_execution_order(plan_run=run, step_id=step.step_id)
        run.result_summary = observation.result_summary
        run.error = None
    else:
        run.workflow_status = validate_transition(run.workflow_status, "fail")
        run.error = observation.error or observation.result_summary
        run.result_summary = run.error
    return run


def continue_after_reject(
    *,
    run: PlanRun,
    reason: str,
    pending_tool_call: Mapping[str, Any] | None = None,
) -> PlanRun:
    step = require_waiting_step(run)
    continuation = record_plan_continuation(
        plan_run=run,
        waiting_step=step,
        action="reject",
        metadata=None,
        extra={
            "reason": reason,
            "pending_tool_call": dict(pending_tool_call or {}),
            "side_effect_executed": False,
        },
    )
    run.workflow_status = validate_transition(run.workflow_status, "resume_reject")
    step.status = "cancelled"
    step.error = None
    step.result_summary = reason
    run.current_step_id = None
    run.current_tool_call = None
    run.error = None
    run.result_summary = reason
    run.metadata["resume"] = continuation
    return run


def require_waiting_step(plan_run: PlanRun) -> PlanStep:
    if plan_run.workflow_status in {"succeeded", "failed", "cancelled"}:
        raise ValueError(f"Plan run is already terminal: {plan_run.workflow_status}.")
    if plan_run.workflow_status != "waiting_user":
        raise ValueError(f"Plan run is not waiting_user: {plan_run.workflow_status}.")
    if not plan_run.current_step_id:
        raise ValueError("Plan run current_step_id is required for continuation.")
    for step in plan_run.steps:
        if step.step_id == plan_run.current_step_id:
            if step.status != "waiting_user":
                raise ValueError("Plan current step is not waiting_user.")
            return step
    raise ValueError("Plan current_step_id does not match any waiting step.")


def record_plan_continuation(
    *,
    plan_run: PlanRun,
    waiting_step: PlanStep,
    action: str,
    metadata: Mapping[str, Any] | None,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    continuation = {
        "mode": "plan",
        "action": action,
        "plan_run_id": plan_run.plan_run_id,
        "waiting_step_id": waiting_step.step_id,
        "continued_from_step_id": waiting_step.step_id,
        "metadata": dict(metadata or {}),
        **dict(extra),
    }
    history = list(plan_run.metadata.get("continuations") or [])
    history.append(continuation)
    plan_run.metadata["continuations"] = history
    waiting_step.metadata = {**dict(waiting_step.metadata or {}), "continuation": continuation}
    return continuation


def prepare_observation_for_step(
    *,
    plan_run: PlanRun,
    step: PlanStep,
    observation: ToolObservation,
) -> ToolObservation:
    if not observation.requires_user:
        return observation
    return attach_observation_hitl_metadata(
        observation=observation,
        hitl_metadata=build_plan_hitl_metadata(
            plan_run=plan_run,
            step=step,
            user_prompt=observation.user_prompt or observation.result_summary,
            source="tool_observation",
        ),
    )


def persist_step_observation(
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


def mark_step_waiting_on_observation(
    *,
    plan_run: PlanRun,
    step: PlanStep,
    observation: ToolObservation,
) -> None:
    hitl_metadata = dict(observation.metadata.get("hitl") or {})
    step.status = "waiting_user"
    step.metadata["hitl"] = hitl_metadata
    plan_run.metadata["hitl"] = hitl_metadata


def append_execution_order(*, plan_run: PlanRun, step_id: str) -> None:
    execution_order = list(plan_run.metadata.get("execution_order") or [])
    execution_order.append(step_id)
    plan_run.metadata["execution_order"] = execution_order


def collect_citations(observations: Sequence[ToolObservation]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for observation in observations:
        if observation.success:
            citations.extend(observation.citations)
    return _deduplicate_citations(citations)


def attach_observation_hitl_metadata(
    *,
    observation: ToolObservation,
    hitl_metadata: dict[str, Any],
) -> ToolObservation:
    metadata = dict(observation.metadata)
    metadata["hitl"] = dict(hitl_metadata)
    return observation.model_copy(update={"metadata": metadata})


def build_plan_hitl_metadata(
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


def observation_error(observation: ToolObservation) -> str:
    return observation.error or observation.result_summary


def build_step_execution_context(
    *,
    plan_run: PlanRun,
    step: PlanStep,
    node_name: str,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        session_id=plan_run.session_id,
        request_id=plan_run.request_id,
        run_id=plan_run.plan_run_id,
        node_name=node_name,
        step_id=step.step_id,
        metadata={"mode": "plan"},
    )


def _synthesize_with_guard(
    *,
    plan_run: PlanRun,
    context: PlanSynthesisContext,
    synthesizer: PlanFinalSynthesizer,
    model_call_guard: SharedModelCallGuard | None,
) -> PlanSynthesisResult:
    if model_call_guard is None:
        return synthesizer.synthesize(context)
    runtime_context = default_model_call_context(
        session_id=plan_run.session_id,
        request_id=plan_run.request_id,
        scene="platform.plan",
        complexity=str(plan_run.metadata.get("complexity") or "moderate"),
        workflow_metadata={
            "run_id": plan_run.plan_run_id,
            "checkpoint_ns": "plan_graph",
            "metadata": {"node": "synthesize_plan_result"},
        },
        request_metadata={
            "operation": "plan.final_synthesis",
            "step_count": len(plan_run.steps),
        },
    )
    return model_call_guard.invoke(
        lambda: synthesizer.synthesize(context),
        context=runtime_context,
        metadata={"operation": "plan.final_synthesis"},
        output_type=PlanSynthesisResult,
    )


def _blocking_dependency_ids(*, step: PlanStep, step_by_id: Mapping[str, PlanStep]) -> list[str]:
    blockers: list[str] = []
    for dependency_id in step.depends_on:
        dependency = step_by_id[dependency_id]
        if dependency.status in {"failed", "cancelled", "blocked"}:
            blockers.append(dependency_id)
    return blockers


def _mark_step_blocked(*, step: PlanStep, blockers: Sequence[str]) -> None:
    blocked_by = list(blockers)
    message = f"Plan step {step.step_id} is blocked by dependency {', '.join(blocked_by)}."
    step.status = "blocked"
    step.error = message
    step.result_summary = message
    step.metadata["blocked_by"] = blocked_by


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

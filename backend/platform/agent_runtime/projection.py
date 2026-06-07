from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document

from backend.platform.agent_runtime.contracts import (
    PlanRun,
    ReActRun,
    ReActTurn,
    ToolExecutionMetadata,
    ToolObservation,
)

RuntimeFinalDecision = Literal[
    "answer_with_evidence",
    "ask_user",
    "direct_answer",
    "max_rounds_reached",
    "no_evidence",
    "retrieval_failed",
]

AnswerMode = Literal["evidence_answer", "direct_answer", "follow_up", "fallback"]


@dataclass(frozen=True)
class RuntimeRunProjection:
    """Platform-owned facts projected from one ReAct or Plan run."""

    run: ReActRun | PlanRun
    observations: list[ToolObservation]
    successful_observations: list[ToolObservation]
    current_observation: ToolObservation | None
    documents: list[Document]
    final_decision: RuntimeFinalDecision | None
    follow_up_question: str | None
    exit_reason: str
    react_run: dict[str, Any] | None
    plan_run: dict[str, Any] | None
    current_turn_id: str | None
    current_step_id: str | None
    current_tool_call: ToolExecutionMetadata | None
    tool_observation: dict[str, Any] | None


def project_runtime_run(
    *,
    react_run: ReActRun | None = None,
    plan_run: PlanRun | None = None,
) -> RuntimeRunProjection:
    """Project a single Agent runtime run into shared platform fields."""
    run = require_single_agent_run(react_run=react_run, plan_run=plan_run)
    observations = list(run.observations)
    successful_observations = [
        observation for observation in observations if observation.success
    ]
    current_observation = observations[-1] if observations else None
    final_decision = (
        final_decision_from_observations(observations)
        if observations
        else final_decision_from_observationless_run(run)
    )
    follow_up_question = (
        follow_up_question_from_observations(observations)
        if observations
        else follow_up_question_from_observationless_run(run)
    )
    current_tool_call = (
        current_tool_call_from_run(run=run, observation=current_observation)
        if current_observation is not None
        else run.current_tool_call
    )
    return RuntimeRunProjection(
        run=run,
        observations=observations,
        successful_observations=successful_observations,
        current_observation=current_observation,
        documents=deduplicate_documents(
            documents_from_observations(successful_observations)
        ),
        final_decision=final_decision,
        follow_up_question=follow_up_question,
        exit_reason=exit_reason_from_run(run),
        react_run=react_run.model_dump() if react_run is not None else None,
        plan_run=plan_run.model_dump() if plan_run is not None else None,
        current_turn_id=event_turn_id(react_run) if react_run is not None else None,
        current_step_id=event_step_id(plan_run) if plan_run is not None else None,
        current_tool_call=current_tool_call,
        tool_observation=(
            current_observation.model_dump()
            if current_observation is not None
            else None
        ),
    )


def require_single_agent_run(
    *,
    react_run: ReActRun | None,
    plan_run: PlanRun | None,
) -> ReActRun | PlanRun:
    if (react_run is None) == (plan_run is None):
        raise ValueError("Exactly one agent runtime run must be provided.")
    return react_run if react_run is not None else plan_run  # type: ignore[return-value]


def documents_from_observations(
    observations: list[ToolObservation],
) -> list[Document]:
    documents: list[Document] = []
    for observation in observations:
        documents.extend(documents_from_observation(observation))
    return documents


def documents_from_observation(observation: ToolObservation) -> list[Document]:
    output = observation.output if isinstance(observation.output, Mapping) else {}
    raw_documents = output.get("documents") if isinstance(output, Mapping) else None
    documents: list[Document] = []
    if not isinstance(raw_documents, list):
        return documents
    for item in raw_documents:
        if not isinstance(item, Mapping):
            continue
        page_content = str(item.get("page_content") or item.get("content") or "")
        if not page_content:
            continue
        metadata = item.get("metadata")
        documents.append(
            Document(
                page_content=page_content,
                metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
            )
        )
    return documents


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    deduplicated: list[Document] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        key = document_identity(document)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(document)
    return deduplicated


def document_identity(document: Document) -> tuple[str, str]:
    metadata = document.metadata
    namespace = str(metadata.get("namespace", "knowledge"))
    value = (
        metadata.get("chunk_id")
        or metadata.get("citation_id")
        or metadata.get("document_id")
        or metadata.get("source_path")
        or document.page_content[:256]
    )
    return namespace, str(value)


def raw_retrieval_trace(observation: ToolObservation) -> dict[str, Any]:
    trace = observation.trace.get("retrieval_trace")
    return dict(trace) if isinstance(trace, Mapping) else {}


def final_decision_from_observation(
    observation: ToolObservation,
) -> RuntimeFinalDecision | None:
    for source in (
        observation.metadata,
        observation.output if isinstance(observation.output, Mapping) else {},
        raw_retrieval_trace(observation),
    ):
        value = source.get("final_decision") if isinstance(source, Mapping) else None
        if value in {
            "answer_with_evidence",
            "ask_user",
            "max_rounds_reached",
            "no_evidence",
            "retrieval_failed",
        }:
            return value  # type: ignore[return-value]
    if observation.success:
        return "no_evidence"
    return "retrieval_failed"


def final_decision_from_observations(
    observations: list[ToolObservation],
) -> RuntimeFinalDecision | None:
    decisions = [
        decision
        for observation in observations
        if (decision := final_decision_from_observation(observation)) is not None
    ]
    if "answer_with_evidence" in decisions:
        return "answer_with_evidence"
    if decisions:
        return decisions[-1]
    if any(observation.success for observation in observations):
        return "no_evidence"
    return "retrieval_failed"


def follow_up_question_from_observation(observation: ToolObservation) -> str | None:
    if observation.user_prompt:
        return observation.user_prompt
    output = observation.output if isinstance(observation.output, Mapping) else {}
    value = output.get("follow_up_question") if isinstance(output, Mapping) else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    trace = raw_retrieval_trace(observation)
    value = trace.get("follow_up_question")
    return value if isinstance(value, str) and value.strip() else None


def follow_up_question_from_observations(
    observations: list[ToolObservation],
) -> str | None:
    for observation in reversed(observations):
        value = follow_up_question_from_observation(observation)
        if value:
            return value
    return None


def final_decision_from_observationless_run(
    run: ReActRun | PlanRun,
) -> RuntimeFinalDecision:
    if isinstance(run, ReActRun):
        turn = latest_react_turn(run)
        action_type = turn.action.action_type if turn is not None else None
        if run.workflow_status == "waiting_user" or action_type == "ask_user":
            return "ask_user"
        if run.workflow_status == "succeeded" and action_type in {"final_answer", "stop"}:
            return "direct_answer"
    return "no_evidence"


def follow_up_question_from_observationless_run(run: ReActRun | PlanRun) -> str | None:
    if not isinstance(run, ReActRun):
        return None
    turn = latest_react_turn(run)
    if turn is None:
        return None
    if turn.action.instruction:
        return turn.action.instruction
    if turn.result_summary:
        return turn.result_summary
    hitl = turn.metadata.get("hitl") if isinstance(turn.metadata, Mapping) else None
    if isinstance(hitl, Mapping):
        user_prompt = hitl.get("user_prompt")
        if isinstance(user_prompt, str) and user_prompt.strip():
            return user_prompt.strip()
    return None


def exit_reason_from_run(run: ReActRun | PlanRun) -> str:
    if run.workflow_status == "waiting_user":
        return "ask_user"
    if run.workflow_status == "succeeded":
        return "no_tool_observation"
    return str(run.workflow_status)


def latest_react_turn(run: ReActRun) -> ReActTurn | None:
    if run.current_turn_id:
        for turn in reversed(run.turns):
            if turn.turn_id == run.current_turn_id:
                return turn
    return run.turns[-1] if run.turns else None


def current_tool_call_from_run(
    *,
    run: ReActRun | PlanRun,
    observation: ToolObservation,
) -> ToolExecutionMetadata | None:
    return run.current_tool_call or observation.execution


def event_turn_id(react_run: ReActRun | None) -> str | None:
    if react_run is None:
        return None
    if react_run.current_turn_id:
        return react_run.current_turn_id
    for turn in reversed(react_run.turns):
        if turn.observation is not None:
            return turn.turn_id
    if react_run.turns:
        return react_run.turns[-1].turn_id
    return None


def event_step_id(plan_run: PlanRun | None) -> str | None:
    if plan_run is None:
        return None
    if plan_run.current_step_id:
        return plan_run.current_step_id
    if plan_run.steps:
        return plan_run.steps[-1].step_id
    return None

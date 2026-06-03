from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol
from uuid import uuid4

from pydantic import Field

from backend.platform.agent_runtime.contracts import (
    AgentRuntimeModel,
    ReActAction,
    ReActRun,
    ReActTurn,
    ToolObservation,
)
from backend.platform.agent_runtime.tool_executor import ToolExecutor


class ReActActionContext(AgentRuntimeModel):
    """ReAct action selector 可见的审计上下文，不包含隐藏推理链。"""

    react_run_id: str
    session_id: str
    request_id: str
    user_goal: str
    round_index: int = Field(ge=1)
    max_turns: int = Field(ge=1)
    allowed_tools: list[str] = Field(default_factory=list)
    previous_turns: list[ReActTurn] = Field(default_factory=list)


class ReActSynthesisContext(AgentRuntimeModel):
    """最终汇总器输入，只包含已完成的顶层观察和引用。"""

    react_run_id: str
    session_id: str
    request_id: str
    user_goal: str
    turns: list[ReActTurn] = Field(default_factory=list)
    observations: list[ToolObservation] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReActSynthesisResult(AgentRuntimeModel):
    """ReAct 顶层最终回答和可写入 run metadata 的引用信息。"""

    final_answer: str
    result_summary: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_used: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReActActionSelector(Protocol):
    """选择下一步 ReAct 顶层动作的中立协议。"""

    def select_action(self, context: ReActActionContext) -> ReActAction:
        """Return the next auditable ReAct action."""


class ReActFinalSynthesizer(Protocol):
    """从 ReAct observations 汇总最终回答的中立协议。"""

    def synthesize(self, context: ReActSynthesisContext) -> ReActSynthesisResult:
        """Return the final answer for a completed ReAct run."""


class ObservationSummarySynthesizer:
    """默认最终汇总器：按观察摘要和 citations 生成可审计回答。"""

    def synthesize(self, context: ReActSynthesisContext) -> ReActSynthesisResult:
        successful = [observation for observation in context.observations if observation.success]
        if successful:
            summaries = [
                observation.result_summary or f"{observation.tool_name} succeeded."
                for observation in successful
            ]
            final_answer = "\n".join(summaries)
        else:
            final_answer = "No successful tool observations were collected."
        citations = _deduplicate_citations(context.citations)
        return ReActSynthesisResult(
            final_answer=final_answer,
            result_summary=f"Synthesized {len(successful)} successful observation(s).",
            citations=citations,
            knowledge_used=bool(citations),
            metadata={"observation_count": len(context.observations)},
        )


class ReActRuntime:
    """平台层 ReAct loop，负责 action、tool observation、HITL wait 和最终汇总。"""

    def __init__(
        self,
        *,
        tool_executor: ToolExecutor,
        action_selector: ReActActionSelector,
        final_synthesizer: ReActFinalSynthesizer | None = None,
        max_turns: int = 5,
        turn_id_factory: Callable[[int], str] | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be greater than or equal to 1.")
        self._tool_executor = tool_executor
        self._action_selector = action_selector
        self._final_synthesizer = final_synthesizer or ObservationSummarySynthesizer()
        self._max_turns = max_turns
        self._turn_id_factory = turn_id_factory

    def run(
        self,
        *,
        session_id: str,
        request_id: str,
        user_goal: str,
        react_run_id: str | None = None,
    ) -> ReActRun:
        run = ReActRun(
            react_run_id=react_run_id or str(uuid4()),
            session_id=session_id,
            request_id=request_id,
            user_goal=user_goal,
            max_turns=self._max_turns,
            workflow_status="running",
        )

        while len(run.turns) < run.max_turns:
            round_index = len(run.turns) + 1
            action = self._action_selector.select_action(
                ReActActionContext(
                    react_run_id=run.react_run_id,
                    session_id=run.session_id,
                    request_id=run.request_id,
                    user_goal=run.user_goal,
                    round_index=round_index,
                    max_turns=run.max_turns,
                    allowed_tools=sorted(self._tool_executor.allowed_tools),
                    previous_turns=list(run.turns),
                )
            )
            turn = self._new_turn(run=run, action=action, round_index=round_index)
            run.turns.append(turn)
            run.current_turn_id = turn.turn_id

            if action.action_type == "tool_call":
                self._execute_tool_turn(run=run, turn=turn)
                if run.workflow_status in {"waiting_user", "retrying", "failed"}:
                    return run
                continue

            if action.action_type == "ask_user":
                self._mark_waiting_user(
                    run=run,
                    turn=turn,
                    user_prompt=action.instruction or "Please provide more information.",
                    source="react_action",
                )
                return run

            if action.action_type in {"final_answer", "stop"}:
                turn.status = "succeeded"
                return self._synthesize_success(run=run, final_turn=turn)

        run.metadata["max_turns_reached"] = True
        if _successful_observations(run):
            return self._synthesize_success(run=run)
        return self._mark_failed(
            run=run,
            error="ReAct run reached max_turns without a successful observation.",
        )

    def _new_turn(
        self,
        *,
        run: ReActRun,
        action: ReActAction,
        round_index: int,
    ) -> ReActTurn:
        turn_id = (
            self._turn_id_factory(round_index)
            if self._turn_id_factory is not None
            else str(uuid4())
        )
        return ReActTurn(
            turn_id=turn_id,
            round_index=round_index,
            goal=run.user_goal,
            action=action,
            input=dict(action.input),
        )

    def _execute_tool_turn(self, *, run: ReActRun, turn: ReActTurn) -> None:
        turn.status = "running"
        observation = self._tool_executor.execute(
            tool_name=turn.action.tool_name or "",
            input_payload=turn.action.input,
            attempt=turn.retry_metadata.attempt,
            max_attempts=turn.retry_metadata.max_attempts,
        )
        turn.observation = observation
        turn.observation_summary = observation.result_summary
        turn.result_summary = observation.result_summary
        turn.error = observation.error
        turn.retry_metadata = turn.retry_metadata.model_copy(
            update={
                "attempt": turn.retry_metadata.attempt + 1,
                "retryable": observation.retryable,
                "last_error": observation.error,
            }
        )
        run.current_tool_call = observation.execution

        # 工具要求人工输入时，ReAct turn 自身成为 HITL 恢复点。
        if observation.requires_user:
            self._mark_waiting_user(
                run=run,
                turn=turn,
                user_prompt=observation.user_prompt or observation.result_summary,
                source="tool_observation",
            )
            turn.observation = _attach_observation_hitl_metadata(
                observation=observation,
                hitl_metadata=turn.metadata["hitl"],
            )
            return

        if observation.success:
            turn.status = "succeeded"
            run.current_tool_call = None
            return

        turn.status = "retrying" if observation.retryable else "failed"
        if observation.retryable:
            run.workflow_status = "retrying"
            run.error = observation.error or observation.result_summary
            return
        self._mark_failed(run=run, error=observation.error or observation.result_summary)

    def _mark_waiting_user(
        self,
        *,
        run: ReActRun,
        turn: ReActTurn,
        user_prompt: str,
        source: str,
    ) -> None:
        hitl_metadata = {
            "mode": "react",
            "react_run_id": run.react_run_id,
            "current_turn_id": turn.turn_id,
            "user_prompt": user_prompt,
            "source": source,
        }
        turn.status = "waiting_user"
        turn.result_summary = user_prompt
        turn.metadata["hitl"] = hitl_metadata
        run.workflow_status = "waiting_user"
        run.current_turn_id = turn.turn_id
        run.metadata["hitl"] = hitl_metadata

    def _synthesize_success(
        self,
        *,
        run: ReActRun,
        final_turn: ReActTurn | None = None,
    ) -> ReActRun:
        observations = _collected_observations(run)
        citations = _collect_citations(observations)
        result = self._final_synthesizer.synthesize(
            ReActSynthesisContext(
                react_run_id=run.react_run_id,
                session_id=run.session_id,
                request_id=run.request_id,
                user_goal=run.user_goal,
                turns=list(run.turns),
                observations=observations,
                citations=citations,
                metadata={"max_turns_reached": bool(run.metadata.get("max_turns_reached"))},
            )
        )
        run.workflow_status = "succeeded"
        run.final_answer = result.final_answer
        run.result_summary = result.result_summary
        run.error = None
        run.current_turn_id = None
        run.current_tool_call = None
        run.metadata["citations"] = result.citations
        run.metadata["knowledge_used"] = result.knowledge_used
        run.metadata["final_synthesis"] = result.metadata
        if final_turn is not None:
            final_turn.result_summary = result.result_summary
        return run

    def _mark_failed(self, *, run: ReActRun, error: str) -> ReActRun:
        run.workflow_status = "failed"
        run.error = error
        run.result_summary = error
        run.current_tool_call = None
        return run


def _collected_observations(run: ReActRun) -> list[ToolObservation]:
    return [turn.observation for turn in run.turns if turn.observation is not None]


def _successful_observations(run: ReActRun) -> list[ToolObservation]:
    return [observation for observation in _collected_observations(run) if observation.success]


def _collect_citations(observations: list[ToolObservation]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for observation in observations:
        if observation.success:
            citations.extend(observation.citations)
    return _deduplicate_citations(citations)


def _deduplicate_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

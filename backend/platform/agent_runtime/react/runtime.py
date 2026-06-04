from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from backend.platform.agent_runtime.contracts import (
    ReActAction,
    ReActRun,
    ReActTurn,
    ToolObservation,
    collect_successful_tool_observations,
)
from backend.platform.agent_runtime.react.continuation import (
    ReActContinuationInput,
    ReActContinuationManager,
)
from backend.platform.agent_runtime.react.policy import ReActScenePolicy
from backend.platform.agent_runtime.react.selector import (
    ReActActionContext,
    ReActActionSelectionCoordinator,
    ReActActionSelector,
    ReActActionValidator,
)
from backend.platform.agent_runtime.react.state import (
    attempted_tools,
    build_react_hitl_metadata,
    ensure_react_run_can_continue,
    latest_final_decision,
    resume_metadata,
    transition,
)
from backend.platform.agent_runtime.react.synthesis import (
    ObservationSummarySynthesizer,
    ReActFinalSynthesizer,
    ReActSynthesisContext,
    collect_citations,
)
from backend.platform.agent_runtime.react.tool_turns import ReActToolTurnExecutor
from backend.platform.agent_runtime.tool_executor import ToolExecutor


class ReActRuntime:
    """平台层 ReAct loop，负责 action、tool observation、HITL wait 和最终汇总。"""

    def __init__(
        self,
        *,
        tool_executor: ToolExecutor,
        action_selector: ReActActionSelector,
        final_synthesizer: ReActFinalSynthesizer | None = None,
        max_turns: int = 5,
        scene_policy: ReActScenePolicy | None = None,
        selector_retry_budget: int = 1,
        turn_id_factory: Callable[[int], str] | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be greater than or equal to 1.")
        if selector_retry_budget < 0:
            raise ValueError("selector_retry_budget must be greater than or equal to 0.")
        self._tool_executor = tool_executor
        self._final_synthesizer = final_synthesizer or ObservationSummarySynthesizer()
        self._max_turns = max_turns
        self._scene_policy = scene_policy or ReActScenePolicy(max_turns=max_turns)
        self._turn_id_factory = turn_id_factory
        self._tool_turn_executor = ReActToolTurnExecutor(tool_executor=tool_executor)
        self._continuation_manager = ReActContinuationManager()
        self._action_coordinator = ReActActionSelectionCoordinator(
            action_selector=action_selector,
            action_validator=ReActActionValidator(tool_executor=tool_executor),
            scene_policy=self._scene_policy,
            selector_retry_budget=selector_retry_budget,
        )

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
            workflow_status="created",
        )
        transition(run, "run_start")
        return self.continue_run(run)

    def continue_run(self, run: ReActRun) -> ReActRun:
        """继续执行已有 ReAct run；终态和 waiting_user 由上层统一拦截。"""
        ensure_react_run_can_continue(run)
        while self.has_turn_budget(run):
            round_index = len(run.turns) + 1
            action = self.select_action(run=run, round_index=round_index)
            if action is None:
                return self.mark_failed_from_selector(run=run)
            turn = self.append_turn(run=run, action=action, round_index=round_index)
            self.run_turn(run=run, turn=turn)
            if self.must_stop_after_turn(run):
                return run
        return self.finish_when_budget_exhausted(run)

    def select_action(self, *, run: ReActRun, round_index: int) -> ReActAction | None:
        """公开 selector 边界，供 graph 节点直接复用。"""
        context = self.build_action_context(run=run, round_index=round_index)
        return self._action_coordinator.select_next_action(run=run, context=context)

    def validate_action(self, *, action: ReActAction, run: ReActRun, round_index: int) -> ReActAction:
        """公开 action 校验边界，避免 graph 节点触达私有成员。"""
        return self._action_coordinator.validate_action(
            action=action,
            run=run,
            round_index=round_index,
        )

    def continue_after_respond(
        self,
        *,
        run: ReActRun,
        response: str,
        source: str = "freeform",
        suggestion_id: str | None = None,
        metadata: dict | None = None,
    ) -> ReActRun:
        """接收用户补充信息后，在同一个 ReActRun 上继续调度。"""
        self._continuation_manager.apply(
            run=run,
            continuation=ReActContinuationInput(
                action="respond",
                response=response,
                source=source,
                suggestion_id=suggestion_id,
                metadata=dict(metadata or {}),
            ),
        )
        return self.continue_run(run)

    def continue_after_approve(
        self,
        *,
        run: ReActRun,
        approval_result: dict | None = None,
        pending_tool_call: dict | None = None,
        metadata: dict | None = None,
    ) -> ReActRun:
        """记录审批通过结果后，在同一个 ReActRun 上继续调度。"""
        self._continuation_manager.apply(
            run=run,
            continuation=ReActContinuationInput(
                action="approve",
                approval_result=dict(approval_result or {}),
                pending_tool_call=dict(pending_tool_call or {}),
                metadata=dict(metadata or {}),
            ),
        )
        return self.continue_run(run)

    def continue_after_reject(
        self,
        *,
        run: ReActRun,
        reason: str | None = None,
        pending_tool_call: dict | None = None,
        metadata: dict | None = None,
    ) -> ReActRun:
        """拒绝等待项后关闭同一个 ReActRun，不执行待审批副作用。"""
        self._continuation_manager.apply(
            run=run,
            continuation=ReActContinuationInput(
                action="reject",
                reason=reason,
                pending_tool_call=dict(pending_tool_call or {}),
                metadata=dict(metadata or {}),
            ),
        )
        return run

    def build_action_context(self, *, run: ReActRun, round_index: int) -> ReActActionContext:
        return ReActActionContext(
            react_run_id=run.react_run_id,
            session_id=run.session_id,
            request_id=run.request_id,
            user_goal=run.user_goal,
            round_index=round_index,
            max_turns=run.max_turns,
            allowed_tools=sorted(self._tool_executor.allowed_tools),
            previous_turns=list(run.turns),
            run_observations=list(run.observations),
            attempted_tools=attempted_tools(run),
            latest_final_decision=latest_final_decision(run),
            scene_policy=self._scene_policy,
            resume_metadata=resume_metadata(run),
            metadata={"workflow_status": run.workflow_status},
        )

    def run_turn(self, *, run: ReActRun, turn: ReActTurn) -> None:
        if turn.action.action_type == "tool_call":
            self.execute_tool_turn(run=run, turn=turn)
            return
        if turn.action.action_type == "ask_user":
            self.mark_waiting_user(
                run=run,
                turn=turn,
                user_prompt=turn.action.instruction or "Please provide more information.",
                source="react_action",
            )
            return
        if turn.action.action_type in {"final_answer", "stop"}:
            turn.status = "succeeded"
            self.synthesize_success(run=run, final_turn=turn)

    def append_turn(
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
        turn = ReActTurn(
            turn_id=turn_id,
            round_index=round_index,
            goal=run.user_goal,
            action=action,
            input=dict(action.input),
        )
        run.turns.append(turn)
        run.current_turn_id = turn.turn_id
        return turn

    def execute_tool_turn(self, *, run: ReActRun, turn: ReActTurn) -> None:
        error = self._tool_turn_executor.execute(run=run, turn=turn)
        if error is not None:
            self.mark_failed(run=run, error=error)

    def mark_waiting_user(self, *, run: ReActRun, turn: ReActTurn, user_prompt: str, source: str) -> None:
        hitl_metadata = build_react_hitl_metadata(
            run=run,
            turn=turn,
            user_prompt=user_prompt,
            source=source,
        )
        turn.status = "waiting_user"
        turn.result_summary = user_prompt
        turn.metadata["hitl"] = hitl_metadata
        run.current_turn_id = turn.turn_id
        run.metadata["hitl"] = hitl_metadata
        transition(run, "interrupt")

    def synthesize_success(self, *, run: ReActRun, final_turn: ReActTurn | None = None) -> ReActRun:
        observations = _successful_observations(run)
        result = self._final_synthesizer.synthesize(
            ReActSynthesisContext(
                react_run_id=run.react_run_id,
                session_id=run.session_id,
                request_id=run.request_id,
                user_goal=run.user_goal,
                turns=_successful_turns(run),
                observations=observations,
                citations=collect_citations(observations),
                turn_order=[turn.turn_id for turn in run.turns],
                metadata={"max_turns_reached": bool(run.metadata.get("max_turns_reached"))},
            )
        )
        transition(run, "success")
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

    def mark_failed(self, *, run: ReActRun, error: str) -> ReActRun:
        if run.workflow_status == "retrying":
            transition(run, "tool_error_final")
        elif run.workflow_status not in {"failed", "cancelled"}:
            transition(run, "tool_error_final")
        run.error = error
        run.result_summary = error
        run.current_tool_call = None
        return run

    def mark_failed_from_selector(self, *, run: ReActRun) -> ReActRun:
        latest_failure = run.metadata.get("latest_selector_failure")
        if isinstance(latest_failure, dict):
            error = str(latest_failure.get("error") or "ReAct selector failed.")
        else:
            error = "ReAct selector failed."
        return self.mark_failed(run=run, error=error)

    def finish_when_budget_exhausted(self, run: ReActRun) -> ReActRun:
        run.metadata["max_turns_reached"] = True
        if _successful_observations(run):
            return self.synthesize_success(run=run)
        return self.mark_failed(
            run=run,
            error="ReAct run reached max_turns without a successful observation.",
        )

    def has_turn_budget(self, run: ReActRun) -> bool:
        return len(run.turns) < run.max_turns

    def must_stop_after_turn(self, run: ReActRun) -> bool:
        return run.workflow_status in {"waiting_user", "failed", "cancelled", "succeeded"}


def _successful_observations(run: ReActRun) -> list[ToolObservation]:
    return collect_successful_tool_observations(run)


def _successful_turns(run: ReActRun) -> list[ReActTurn]:
    return [turn for turn in run.turns if turn.status == "succeeded"]

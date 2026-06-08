from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from backend.platform.agent_runtime.chat_graph.runtime_parts.agent_state import (
    _coerce_optional_agent_mode,
)
from backend.platform.agent_runtime.chat_graph.contracts import (
    HitlApproveExecutor,
    HitlResumeError,
    HitlResumeInput,
    HitlRuntimeResult,
    HitlWaitInput,
    HitlRespondHandler,
    _HITL_ALLOWED_ACTIONS,
    _HITL_PENDING_ACTIONS,
)
from backend.platform.agent_runtime.contracts import (
    PlanRun,
    ReActRun,
    ToolExecutionMetadata,
    ToolObservation,
)
from backend.platform.agent_runtime.idempotency import (
    ToolExecutionContext,
    attach_idempotency_trace,
    build_input_hash,
    build_tool_idempotency_key,
)
from backend.platform.agent_runtime.plan.executor import PlanExecutor
from backend.platform.agent_runtime.plan.graph import (
    PlanHitlResumeGraphDependencies,
    build_plan_hitl_resume_graph,
)
from backend.platform.agent_runtime.react.graph import (
    ReActHitlResumeGraphDependencies,
    build_react_hitl_resume_graph,
)
from backend.platform.workflow.langgraph.config import build_runtime_graph_config
from backend.platform.workflow.langgraph.state import RuntimeGraphState, RuntimeHitlState, build_runtime_hitl_state
from backend.platform.workflow.state_machine import (
    WorkflowRunEvent,
    WorkflowRunState,
    is_terminal,
    validate_transition,
)


@dataclass(frozen=True)
class _AgentGraphResumeResult:
    answer: str
    status: str
    state_update: dict[str, Any]
    response_result: dict[str, Any]
    tool_result: dict[str, Any] | None = None


class HitlRuntimeMixin:
    def create_hitl_wait(
        self,
        *,
        wait: HitlWaitInput,
        interrupt_id: str,
    ) -> HitlRuntimeResult:
        """把当前会话标记为 waiting_user，让后续请求知道它正在等用户处理。"""
        config = build_runtime_graph_config(
            session_id=wait.session_id,
            request_id=wait.request_id,
            metadata=dict(wait.metadata or {}),
        )
        run = self.lifecycle.create_run(
            thread_id=wait.session_id,
            request_id=wait.request_id,
            metadata=dict(config["metadata"]),
        )
        if (wait.metadata or {}).get("mode") == "plan":
            self.lifecycle.mark_planning(run)
        self.lifecycle.mark_running(run)
        try:
            state = self._load_or_build_thread_state(
                session_id=wait.session_id,
                request_id=wait.request_id,
                config=config,
            )
            output = self._persist_state_update(
                state=state,
                config=config,
                update=self.build_hitl_wait_update(
                    wait=wait,
                    interrupt_id=interrupt_id,
                    state=state,
                    run_id=run.run_id,
                ),
            )
        except Exception as exc:
            self.lifecycle.mark_failed(run, exc)
            raise

        self.lifecycle.mark_waiting_user(run)
        return HitlRuntimeResult(state=output, config=config, run_id=run.run_id)

    def build_hitl_wait_update(
        self,
        *,
        wait: HitlWaitInput,
        interrupt_id: str,
        state: RuntimeGraphState,
        run_id: str,
    ) -> dict[str, Any]:
        """构造 waiting_user checkpoint 更新，供 graph interrupt 和兼容入口复用。"""
        self._validate_hitl_wait(wait)
        hitl = build_runtime_hitl_state(
            interrupt_id=interrupt_id,
            thread_id=wait.session_id,
            reason=wait.reason,
            pending_action=wait.pending_action,
            allowed_actions=wait.allowed_actions,
            proposed_tool_call=wait.proposed_tool_call,
            suggested_responses=wait.suggested_responses,
            allow_freeform_response=wait.allow_freeform_response,
            metadata=wait.metadata,
        )
        orchestration_update = self._build_agent_runtime_wait_update(
            wait=wait,
            hitl=hitl,
        )
        return {
            "request_id": wait.request_id,
            "answer": wait.reason,
            "knowledge_used": False,
            "citations": [],
            "status": "waiting_user",
            "run_id": run_id,
            "state_event": "interrupt",
            "final_state": None,
            "hitl": hitl,
            "metadata": {
                **dict(state.get("metadata") or {}),
                **dict(wait.metadata or {}),
            },
            **orchestration_update,
        }

    def resume_hitl(
        self,
        *,
        resume: HitlResumeInput,
        approve_executor: HitlApproveExecutor | None = None,
        respond_handler: HitlRespondHandler | None = None,
        plan_executor: PlanExecutor | None = None,
    ) -> HitlRuntimeResult:
        """处理用户的 approve、reject、respond 动作，并更新当前会话状态。"""
        with self._resume_lock:
            return self._resume_hitl_locked(
                resume=resume,
                approve_executor=approve_executor,
                respond_handler=respond_handler,
                plan_executor=plan_executor,
            )

    def _resume_hitl_locked(
        self,
        *,
        resume: HitlResumeInput,
        approve_executor: HitlApproveExecutor | None,
        respond_handler: HitlRespondHandler | None,
        plan_executor: PlanExecutor | None,
    ) -> HitlRuntimeResult:
        """真正执行 resume；外层已加锁，避免同一等待点被同时处理两次。"""
        config = build_runtime_graph_config(
            session_id=resume.session_id,
            request_id=resume.request_id,
            metadata=dict(resume.metadata or {}),
        )
        run = self.lifecycle.create_run(
            thread_id=resume.session_id,
            request_id=resume.request_id,
            metadata=dict(config["metadata"]),
        )
        self.lifecycle.mark_running(run)
        accepted_state: RuntimeGraphState | None = None
        try:
            state = self._load_or_build_thread_state(
                session_id=resume.session_id,
                request_id=resume.request_id,
                config=config,
                require_checkpoint=True,
            )
            hitl = self._validate_hitl_resume(state=state, resume=resume)
            resume_payload = self._build_resume_payload(resume=resume, hitl=hitl)
            resume_event = self._event_for_resume_action(resume.action)
            resumed_state = validate_transition(str(state.get("status")), resume_event)
            tool_result: dict[str, Any] | None = None
            response_result: dict[str, Any] = {}
            next_answer = str(state.get("answer") or "")
            next_status: WorkflowRunState = resumed_state
            state_event: WorkflowRunEvent = resume_event
            orchestration_mode = _coerce_optional_agent_mode(
                dict(hitl.get("metadata") or {}).get("mode")
            )
            agent_resume_update: dict[str, Any] = {}

            if orchestration_mode is not None:
                accepted_state = self._persist_resume_acceptance(
                    state=state,
                    config=config,
                    run_id=run.run_id,
                    resume_payload=resume_payload,
                    resumed_state=resumed_state,
                    resume_event=resume_event,
                    update_agent_runtime=False,
                )
                graph_result = self._resume_agent_runtime_graph(
                    mode=orchestration_mode,
                    state=state,
                    accepted_state=accepted_state,
                    hitl=hitl,
                    resume_payload=resume_payload,
                    approve_executor=approve_executor,
                    respond_handler=respond_handler,
                    plan_executor=plan_executor,
                )
                tool_result = graph_result.tool_result
                response_result = graph_result.response_result
                next_answer = graph_result.answer or next_answer
                next_status, state_event = self._resolve_graph_resume_state(
                    current_state=resumed_state,
                    requested_state=graph_result.status,
                    action=resume.action,
                )
                accepted_state = self._persist_state_update(
                    state=accepted_state,
                    config=config,
                    update={
                        **graph_result.state_update,
                        "status": next_status,
                        "state_event": state_event,
                        "final_state": next_status if is_terminal(next_status) else None,
                        "hitl": None,
                        "hitl_resume": resume_payload,
                        "metadata": {
                            **dict(accepted_state.get("metadata") or {}),
                            "hitl_resume": resume_payload,
                            "hitl_tool_result": tool_result,
                        },
                    },
                )
                agent_resume_update = graph_result.state_update

            elif resume.action == "approve":
                accepted_state = self._persist_resume_acceptance(
                    state=state,
                    config=config,
                    run_id=run.run_id,
                    resume_payload=resume_payload,
                    resumed_state=resumed_state,
                    resume_event=resume_event,
                )
                tool_result = self._execute_approved_tool(
                    hitl=hitl,
                    resume_payload=resume_payload,
                    approve_executor=approve_executor,
                )
                next_status = validate_transition(resumed_state, "success")
                state_event = "success"
                next_answer = "已批准并执行待审批操作。"
            elif resume.action == "reject":
                next_status = resumed_state
                state_event = "resume_reject"
                next_answer = "已拒绝该人工等待项，未执行待审批调用。"
            elif resume.action == "respond":
                if respond_handler is None:
                    raise HitlResumeError("respond_handler is required for respond action.")
                accepted_state = self._persist_resume_acceptance(
                    state=state,
                    config=config,
                    run_id=run.run_id,
                    resume_payload=resume_payload,
                    resumed_state=resumed_state,
                    resume_event=resume_event,
                )
                # 先消费等待点后，把已恢复的 running state 交给 handler，避免继续使用旧的 waiting_user 快照。
                response_result = dict(respond_handler(resume_payload, accepted_state) or {})
                next_status, state_event = self._resolve_running_result_state(
                    current_state=resumed_state,
                    requested_state=str(response_result.get("status") or "succeeded"),
                )
                next_answer = str(response_result.get("answer", next_answer))
            elif resume.action == "edit":
                raise HitlResumeError("edit action is not supported yet.")

            response_state_update = self._build_respond_state_update(
                action=resume.action,
                resume_payload=resume_payload,
                answer=next_answer,
                response_result=response_result,
            )
            output = self._persist_state_update(
                state=accepted_state or state,
                config=config,
                update={
                    "request_id": resume.request_id,
                    "answer": next_answer,
                    "status": next_status,
                    "run_id": run.run_id,
                    "state_event": state_event,
                    "final_state": next_status if is_terminal(next_status) else None,
                    "hitl": None,
                    "hitl_resume": resume_payload,
                    "metadata": {
                        **dict((accepted_state or state).get("metadata") or {}),
                        **dict(config["metadata"]),
                        "hitl_resume": resume_payload,
                        "hitl_tool_result": tool_result,
                    },
                    **(
                        agent_resume_update
                        if orchestration_mode is not None
                        else self._build_agent_runtime_resume_update(
                            state=accepted_state or state,
                            action=resume.action,
                            resume_payload=resume_payload,
                            next_status=next_status,
                            final_answer=next_answer,
                            tool_result=tool_result,
                        )
                    ),
                    **response_state_update,
                },
            )
        except Exception as exc:
            if accepted_state is not None:
                self._persist_failed_after_accepted_resume(
                    state=accepted_state,
                    config=config,
                    run_id=run.run_id,
                    error=exc,
                )
            self.lifecycle.mark_failed(run, exc)
            raise

        if next_status == "cancelled":
            self.lifecycle.mark_cancelled(run)
        elif next_status == "failed":
            self.lifecycle.mark_failed(run, next_answer or "resume failed")
        elif next_status == "succeeded":
            self.lifecycle.mark_succeeded(run)
        return HitlRuntimeResult(
            state=output,
            config=config,
            run_id=run.run_id,
            tool_result=tool_result,
        )

    def _resume_agent_runtime_graph(
        self,
        *,
        mode: str,
        state: RuntimeGraphState,
        accepted_state: RuntimeGraphState,
        hitl: RuntimeHitlState,
        resume_payload: Mapping[str, Any],
        approve_executor: HitlApproveExecutor | None,
        respond_handler: HitlRespondHandler | None,
        plan_executor: PlanExecutor | None,
    ) -> _AgentGraphResumeResult:
        """Route accepted orchestration HITL resume through the owning Agent graph."""
        proposed_tool_call = (
            dict(hitl.get("proposed_tool_call") or {})
            if hitl.get("proposed_tool_call") is not None
            else None
        )
        base_input = {
            "resume_payload": dict(resume_payload),
            "proposed_tool_call": proposed_tool_call,
            "accepted_state": dict(accepted_state),
        }
        if mode == "react":
            react_run = state.get("react_run")
            if not isinstance(react_run, Mapping):
                raise HitlResumeError("react_run checkpoint is required for ReAct HITL resume.")
            graph = build_react_hitl_resume_graph(
                ReActHitlResumeGraphDependencies(
                    approve_executor=approve_executor,
                    respond_handler=respond_handler,
                )
            )
            result = graph.invoke(
                {
                    **base_input,
                    "run": ReActRun.model_validate(dict(react_run)),
                }
            )
            run = result.get("run")
            run_payload = run.model_dump() if hasattr(run, "model_dump") else run
            response_result = dict(result.get("response_result") or {})
            return _AgentGraphResumeResult(
                answer=str(result.get("answer") or ""),
                status=str(result.get("status") or "running"),
                tool_result=(
                    dict(result.get("tool_result"))
                    if isinstance(result.get("tool_result"), Mapping)
                    else None
                ),
                response_result=response_result,
                state_update={
                    "agent_mode": "react",
                    "react_run": run_payload,
                    "plan_run": None,
                    "current_turn_id": getattr(run, "current_turn_id", None),
                    "current_step_id": None,
                    "current_tool_call": (
                        run.current_tool_call.model_dump()
                        if getattr(run, "current_tool_call", None) is not None
                        else None
                    ),
                },
            )

        plan_run = state.get("plan_run")
        if not isinstance(plan_run, Mapping):
            raise HitlResumeError("plan_run checkpoint is required for Plan HITL resume.")
        graph = build_plan_hitl_resume_graph(
            PlanHitlResumeGraphDependencies(
                executor=self._require_plan_resume_executor(plan_executor),
            )
        )
        result = graph.invoke(
            {
                **base_input,
                "plan_run": PlanRun.model_validate(dict(plan_run)),
            }
        )
        run = result.get("plan_run")
        run_payload = run.model_dump() if hasattr(run, "model_dump") else run
        response_result = dict(result.get("response_result") or {})
        return _AgentGraphResumeResult(
            answer=str(result.get("answer") or ""),
            status=str(result.get("status") or "running"),
            tool_result=(
                dict(result.get("tool_result"))
                if isinstance(result.get("tool_result"), Mapping)
                else None
            ),
            response_result=response_result,
            state_update={
                "agent_mode": "plan",
                "react_run": None,
                "plan_run": run_payload,
                "current_turn_id": None,
                "current_step_id": getattr(run, "current_step_id", None),
                "current_tool_call": (
                    run.current_tool_call.model_dump()
                    if getattr(run, "current_tool_call", None) is not None
                    else None
                ),
            },
        )

    def _require_plan_resume_executor(
        self,
        plan_executor: PlanExecutor | None,
    ) -> PlanExecutor:
        """Plan HITL resume 必须通过 PlanExecutor 回到图内执行边界。"""
        if plan_executor is None:
            raise HitlResumeError("plan_executor is required for Plan HITL resume.")
        return plan_executor

    def _resolve_graph_resume_state(
        self,
        *,
        current_state: WorkflowRunState,
        requested_state: str,
        action: str,
    ) -> tuple[WorkflowRunState, WorkflowRunEvent]:
        """Normalize graph resume status back through the shared workflow machine."""
        if action == "reject":
            if requested_state != "cancelled":
                raise HitlResumeError("reject resume must cancel the owning Agent run.")
            return "cancelled", "resume_reject"
        return self._resolve_running_result_state(
            current_state=current_state,
            requested_state=requested_state,
        )

    def _validate_hitl_wait(self, wait: HitlWaitInput) -> None:
        """检查等待任务是否合法，避免写出前端不知道怎么展示的状态。"""
        if wait.pending_action not in _HITL_PENDING_ACTIONS:
            raise HitlResumeError("pending_action is not supported for HITL wait.")
        allowed_actions = set(wait.allowed_actions)
        if not allowed_actions:
            raise HitlResumeError("allowed_actions is required for HITL wait.")
        if not allowed_actions.issubset(_HITL_ALLOWED_ACTIONS):
            raise HitlResumeError("allowed_actions contains unsupported action.")

        if wait.pending_action == "clarification":
            if "respond" not in allowed_actions:
                raise HitlResumeError("clarification HITL wait requires respond action.")
            if wait.proposed_tool_call:
                raise HitlResumeError("clarification HITL wait cannot include proposed_tool_call.")
            return

        if "respond" in allowed_actions:
            raise HitlResumeError("approval HITL wait cannot include respond action.")
        if not wait.proposed_tool_call:
            raise HitlResumeError("approval HITL wait requires proposed_tool_call.")
        if wait.suggested_responses:
            raise HitlResumeError("approval HITL wait cannot include suggested_responses.")
        if wait.allow_freeform_response:
            raise HitlResumeError("approval HITL wait cannot allow freeform response.")

    def _load_or_build_thread_state(
        self,
        *,
        session_id: str,
        request_id: str,
        config: dict[str, Any],
        require_checkpoint: bool = False,
    ) -> RuntimeGraphState:
        """读取会话最新保存状态；没有保存过时就创建一个空状态。"""
        checkpoint = self.checkpointer.get_tuple(config)
        if checkpoint is None:
            if require_checkpoint:
                raise HitlResumeError("No checkpoint found for HITL resume.")
            return build_runtime_graph_state(
                session_id=session_id,
                request_id=request_id,
                metadata=dict(config["metadata"]),
            )

        values = dict(checkpoint.checkpoint.get("channel_values") or {})
        return build_runtime_graph_state(
            session_id=str(values.get("session_id") or session_id),
            request_id=str(values.get("request_id") or request_id),
            scene=values.get("scene"),
            messages=list(values.get("messages") or ()),
            answer=str(values.get("answer") or ""),
            knowledge_used=bool(values.get("knowledge_used", False)),
            citations=list(values.get("citations") or ()),
            retrieval_trace=dict(values.get("retrieval_trace") or {}),
            metadata=dict(values.get("metadata") or {}),
            answer_mode=values.get("answer_mode"),
            status=values.get("status", "running"),
            run_id=values.get("run_id"),
            state_event=values.get("state_event"),
            final_state=values.get("final_state"),
            retry_attempt=int(values.get("retry_attempt") or 0),
            retry_metadata=dict(values.get("retry_metadata") or {}),
            hitl=values.get("hitl"),
            hitl_resume=values.get("hitl_resume"),
            agent_mode=values.get("agent_mode"),
            agent_mode_reason=values.get("agent_mode_reason"),
            agent_mode_signals=values.get("agent_mode_signals"),
            react_run=values.get("react_run"),
            plan_run=values.get("plan_run"),
            current_turn_id=values.get("current_turn_id"),
            current_step_id=values.get("current_step_id"),
            current_tool_call=values.get("current_tool_call"),
            documents=values.get("documents"),
            tool_event=values.get("tool_event"),
            final_decision=values.get("final_decision"),
            follow_up_question=values.get("follow_up_question"),
            tool_observation=values.get("tool_observation"),
        )

    def _validate_hitl_resume(
        self,
        *,
        state: RuntimeGraphState,
        resume: HitlResumeInput,
    ) -> RuntimeHitlState:
        """确认用户要恢复的等待点就是当前正在等待的那个。"""
        status = str(state.get("status") or "running")
        if is_terminal(status):
            raise HitlResumeError(
                f"Current workflow is already terminal: {status}."
            )
        if state.get("status") != "waiting_user":
            raise HitlResumeError("Current thread is not waiting for user input.")
        if state.get("session_id") != resume.session_id:
            raise HitlResumeError("session_id does not match current HITL state.")
        hitl = state.get("hitl")
        if not hitl:
            raise HitlResumeError("Current thread has no HITL payload.")
        if hitl.get("thread_id") != resume.session_id:
            raise HitlResumeError("HITL thread_id does not match resume session_id.")
        if hitl.get("interrupt_id") != resume.interrupt_id:
            raise HitlResumeError("interrupt_id does not match current HITL state.")
        allowed_actions = set(hitl.get("allowed_actions") or ())
        if resume.action not in allowed_actions:
            raise HitlResumeError("action is not allowed for current HITL state.")
        self._validate_agent_runtime_resume_point(state=state, hitl=hitl)
        return hitl

    def _validate_agent_runtime_resume_point(
        self,
        *,
        state: RuntimeGraphState,
        hitl: RuntimeHitlState,
    ) -> None:
        """校验顶层 Agent 等待点。"""
        metadata = dict(hitl.get("metadata") or {})
        mode = _coerce_optional_agent_mode(metadata.get("mode"))
        if mode is None:
            return
        if state.get("agent_mode") != mode:
            raise HitlResumeError("HITL orchestration mode does not match checkpoint.")
        if mode == "react":
            expected_run_id = str(metadata.get("react_run_id") or "")
            if not expected_run_id:
                raise HitlResumeError("react_run_id is required for ReAct HITL resume.")
            react_run = state.get("react_run")
            actual_run_id = (
                str(react_run.get("react_run_id") or "")
                if isinstance(react_run, Mapping)
                else ""
            )
            if expected_run_id and actual_run_id != expected_run_id:
                raise HitlResumeError("react_run_id does not match current HITL state.")
            expected_turn_id = str(metadata.get("current_turn_id") or "")
            if not expected_turn_id:
                raise HitlResumeError("current_turn_id is required for ReAct HITL resume.")
            if expected_turn_id and state.get("current_turn_id") != expected_turn_id:
                raise HitlResumeError("current_turn_id does not match current HITL state.")
            return
        expected_run_id = str(metadata.get("plan_run_id") or "")
        if not expected_run_id:
            raise HitlResumeError("plan_run_id is required for Plan HITL resume.")
        plan_run = state.get("plan_run")
        actual_run_id = (
            str(plan_run.get("plan_run_id") or "")
            if isinstance(plan_run, Mapping)
            else ""
        )
        if expected_run_id and actual_run_id != expected_run_id:
            raise HitlResumeError("plan_run_id does not match current HITL state.")
        expected_step_id = str(metadata.get("current_step_id") or "")
        if not expected_step_id:
            raise HitlResumeError("current_step_id is required for Plan HITL resume.")
        if expected_step_id and state.get("current_step_id") != expected_step_id:
            raise HitlResumeError("current_step_id does not match current HITL state.")

    def _persist_resume_acceptance(
        self,
        *,
        state: RuntimeGraphState,
        config: dict[str, Any],
        run_id: str,
        resume_payload: Mapping[str, Any],
        resumed_state: WorkflowRunState,
        resume_event: WorkflowRunEvent,
        update_agent_runtime: bool = True,
    ) -> RuntimeGraphState:
        """先消费等待点再执行副作用，避免 checkpoint 失败后重复 resume。"""
        return self._persist_state_update(
            state=state,
            config=config,
            update={
                "request_id": str(resume_payload.get("request_id") or state["request_id"]),
                "status": resumed_state,
                "run_id": run_id,
                "state_event": resume_event,
                "final_state": None,
                "hitl": None,
                "hitl_resume": dict(resume_payload),
                **(
                    self._build_agent_runtime_resume_update(
                        state=state,
                        action=str(resume_payload.get("action") or ""),
                        resume_payload=resume_payload,
                        next_status=resumed_state,
                        final_answer=None,
                    )
                    if update_agent_runtime
                    else {}
                ),
                "metadata": {
                    **dict(state.get("metadata") or {}),
                    "hitl_resume": dict(resume_payload),
                },
            },
        )

    def _persist_failed_after_accepted_resume(
        self,
        *,
        state: RuntimeGraphState,
        config: dict[str, Any],
        run_id: str,
        error: BaseException,
    ) -> None:
        """已接受 resume 后发生错误时，尽量把 checkpoint 收敛到 failed 终态。"""
        try:
            self._persist_state_update(
                state=state,
                config=config,
                update={
                    "status": "failed",
                    "run_id": run_id,
                    "state_event": "fail",
                    "final_state": "failed",
                    "metadata": {
                        **dict(state.get("metadata") or {}),
                        "error": self._summarize_resume_error(error),
                    },
                },
            )
        except Exception:
            # checkpoint 失败不能覆盖原始副作用或 handler 错误，调用方仍按原错误返回。
            return

    def _summarize_resume_error(self, error: BaseException) -> str:
        """把 resume 副作用错误压缩成可写入 checkpoint metadata 的短文本。"""
        message = str(error)
        return f"{type(error).__name__}: {message}" if message else type(error).__name__

    def _event_for_resume_action(self, action: str) -> WorkflowRunEvent:
        """把用户动作映射为状态机事件，所有 resume 入口共用同一套语义。"""
        if action == "approve":
            return "resume_approve"
        if action == "respond":
            return "resume_respond"
        if action == "reject":
            return "resume_reject"
        if action == "edit":
            raise HitlResumeError("edit action is not supported yet.")
        raise HitlResumeError("resume action is not supported.")

    def _resolve_running_result_state(
        self,
        *,
        current_state: WorkflowRunState,
        requested_state: str,
    ) -> tuple[WorkflowRunState, WorkflowRunEvent]:
        """把继续执行后的结果状态收敛到合法转移，避免 handler 直接写任意状态。"""
        if requested_state == "succeeded":
            return validate_transition(current_state, "success"), "success"
        if requested_state == "failed":
            return validate_transition(current_state, "fail"), "fail"
        if requested_state == "running":
            return current_state, "resume_respond"
        raise HitlResumeError(f"resume handler returned unsupported status: {requested_state}.")

    def _build_resume_payload(
        self,
        *,
        resume: HitlResumeInput,
        hitl: RuntimeHitlState,
    ) -> dict[str, Any]:
        """整理用户恢复时带来的数据，补上会话、动作和请求记录字段。"""
        payload = dict(resume.payload or {})
        payload["action"] = resume.action
        payload["session_id"] = resume.session_id
        payload["interrupt_id"] = resume.interrupt_id
        payload["request_id"] = resume.request_id
        payload["metadata"] = {
            **dict(hitl.get("metadata") or {}),
            **dict(payload.get("metadata") or {}),
        }
        if resume.action == "respond":
            payload.setdefault("source", "freeform")
            payload.setdefault("suggestion_id", None)
            self._validate_respond_payload(payload=payload, hitl=hitl)
        return payload

    def _validate_respond_payload(
        self,
        *,
        payload: dict[str, Any],
        hitl: RuntimeHitlState,
    ) -> None:
        """检查用户补充的信息是否可用，比如不能为空、不能选不存在的建议项。"""
        source = str(payload.get("source") or "freeform")
        if source == "suggested_response":
            self._apply_suggested_response(payload=payload, hitl=hitl)
            return

        if source != "freeform":
            raise HitlResumeError("respond source is not supported.")
        if not bool(hitl.get("allow_freeform_response", False)):
            raise HitlResumeError("freeform response is not allowed for current HITL state.")
        response = str(payload.get("response") or "").strip()
        if not response:
            raise HitlResumeError("respond action requires a non-empty response.")
        payload["response"] = response

    def _apply_suggested_response(
        self,
        *,
        payload: dict[str, Any],
        hitl: RuntimeHitlState,
    ) -> None:
        """用户选择建议项时，把建议项里的文本填到 response 里。"""
        suggestion_id = str(payload.get("suggestion_id") or "")
        if not suggestion_id:
            raise HitlResumeError("suggested_response source requires suggestion_id.")
        suggestions = list(hitl.get("suggested_responses") or ())
        matched = next(
            (
                suggestion
                for suggestion in suggestions
                if suggestion.get("suggestion_id") == suggestion_id
            ),
            None,
        )
        if matched is None:
            raise HitlResumeError("suggestion_id is not allowed for current HITL state.")
        response = str(payload.get("response") or matched.get("value") or "").strip()
        if not response:
            raise HitlResumeError("suggested response value is empty.")
        payload["response"] = response

    def _execute_approved_tool(
        self,
        *,
        hitl: RuntimeHitlState,
        resume_payload: Mapping[str, Any],
        approve_executor: HitlApproveExecutor | None,
    ) -> dict[str, Any]:
        """执行用户已批准的工具调用；没有执行器时不允许继续。"""
        proposed_tool_call = hitl.get("proposed_tool_call")
        if not proposed_tool_call:
            raise HitlResumeError("approve requires a proposed_tool_call.")
        if approve_executor is None:
            raise HitlResumeError("approve_executor is required for approve action.")
        idempotency = self._approved_tool_idempotency(
            hitl=hitl,
            resume_payload=resume_payload,
            proposed_tool_call=proposed_tool_call,
        )
        if idempotency is None:
            result = approve_executor(proposed_tool_call)
            return dict(result or {})

        key, context, tool_name = idempotency
        existing = self.tool_idempotency_store.begin_invocation(
            key=key,
            tool_name=tool_name,
            input_hash=build_input_hash(proposed_tool_call),
            context=context,
            compensation_status="unsupported",
        )
        if existing is not None and existing.observation is not None:
            observation = attach_idempotency_trace(
                observation=existing.observation,
                key=key,
                status=existing.status,
                reused=True,
                compensation_status=existing.compensation_status,
            )
            return observation.model_dump()
        if existing is not None:
            raise HitlResumeError("approved tool invocation is already pending.")

        try:
            result = dict(approve_executor(proposed_tool_call) or {})
        except Exception as exc:
            observation = _approved_tool_exception_to_observation(
                tool_name=tool_name,
                key=key,
                exc=exc,
            )
            self.tool_idempotency_store.mark_failed(
                key=key,
                observation=observation,
                compensation_status="unsupported",
            )
            raise
        observation = _approved_tool_result_to_observation(
            tool_name=tool_name,
            result=result,
        )
        observation = attach_idempotency_trace(
            observation=observation,
            key=key,
            status="succeeded" if observation.success else "failed",
            reused=False,
            compensation_status="unsupported",
        )
        if observation.success:
            self.tool_idempotency_store.mark_succeeded(
                key=key,
                observation=observation,
                compensation_status="unsupported",
            )
        else:
            self.tool_idempotency_store.mark_failed(
                key=key,
                observation=observation,
                compensation_status="unsupported",
            )
        return result

    def _approved_tool_idempotency(
        self,
        *,
        hitl: RuntimeHitlState,
        resume_payload: Mapping[str, Any],
        proposed_tool_call: Mapping[str, Any],
    ) -> tuple[str, ToolExecutionContext, str] | None:
        """为顶层 HITL approve 构造幂等上下文；没有 store 时保持纯执行。"""
        if self.tool_idempotency_store is None:
            return None
        tool_name = str(proposed_tool_call.get("tool_name") or "").strip()
        if not tool_name:
            raise HitlResumeError("proposed_tool_call.tool_name is required.")
        metadata = dict(hitl.get("metadata") or {})
        context = ToolExecutionContext(
            session_id=str(resume_payload.get("session_id") or hitl.get("thread_id") or ""),
            request_id=str(resume_payload.get("request_id") or ""),
            run_id=str(metadata.get("react_run_id") or metadata.get("plan_run_id") or resume_payload.get("run_id") or ""),
            node_name=f"hitl.{hitl.get('pending_action') or 'approve'}",
            turn_id=(
                str(metadata.get("current_turn_id"))
                if metadata.get("current_turn_id") is not None
                else None
            ),
            step_id=(
                str(metadata.get("current_step_id"))
                if metadata.get("current_step_id") is not None
                else None
            ),
            metadata={"mode": metadata.get("mode"), "interrupt_id": hitl.get("interrupt_id")},
        )
        key = build_tool_idempotency_key(
            context=context,
            tool_name=tool_name,
            input_payload=proposed_tool_call,
        )
        return key, context, tool_name

    def _build_respond_state_update(
        self,
        *,
        action: str,
        resume_payload: Mapping[str, Any],
        answer: str,
        response_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """respond 完成后，把继续检索得到的回答、引用和 trace 一起写回 checkpoint。"""
        if action != "respond":
            return {}

        update: dict[str, Any] = {
            "messages": [
                HumanMessage(content=str(resume_payload.get("response") or "")),
                AIMessage(content=answer),
            ],
        }
        if "knowledge_used" in response_result:
            update["knowledge_used"] = bool(response_result["knowledge_used"])
        if "citations" in response_result:
            update["citations"] = list(response_result["citations"] or [])
        if "retrieval_trace" in response_result:
            update["retrieval_trace"] = dict(response_result["retrieval_trace"] or {})
        for key in (
            "agent_mode",
            "react_run",
            "plan_run",
            "current_turn_id",
            "current_step_id",
            "current_tool_call",
            "documents",
            "tool_event",
            "final_decision",
            "follow_up_question",
            "tool_observation",
        ):
            if key in response_result:
                update[key] = response_result[key]
        return update


def _approved_tool_result_to_observation(
    *,
    tool_name: str,
    result: Mapping[str, Any],
) -> ToolObservation:
    """将 approve_executor 返回值收敛为可持久化的统一 observation。"""
    try:
        return ToolObservation.model_validate(dict(result))
    except ValueError:
        return ToolObservation(
            tool_name=tool_name,
            success=True,
            output=dict(result),
            result_summary=f"{tool_name} approved tool executed.",
            metadata=dict(result),
        )


def _approved_tool_exception_to_observation(
    *,
    tool_name: str,
    key: str,
    exc: Exception,
) -> ToolObservation:
    """approve 执行异常也落幂等事实，避免 pending 记录阻塞后续恢复。"""
    observation = ToolObservation(
        tool_name=tool_name,
        success=False,
        result_summary=f"{tool_name} failed after approval: {exc}",
        retryable=False,
        error=str(exc),
        execution=ToolExecutionMetadata(
            tool_name=tool_name,
            idempotency_key=key,
            retryable=False,
        ),
    )
    return attach_idempotency_trace(
        observation=observation,
        key=key,
        status="failed",
        reused=False,
        compensation_status="unsupported",
    )

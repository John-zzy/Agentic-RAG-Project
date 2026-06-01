from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from backend.application.runtime.api.chat.schemas import Citation
from backend.platform.config.settings import AppSettings
from backend.platform.workflow.langgraph.checkpointer import SQLiteLangGraphCheckpointer
from backend.platform.workflow.langgraph.config import build_runtime_graph_config
from backend.platform.workflow.langgraph.lifecycle import GraphRunLifecycleRecorder
from backend.platform.workflow.langgraph.state import (
    RuntimeHitlState,
    RuntimeGraphState,
    build_runtime_graph_state,
    build_runtime_hitl_state,
)


class PreparedGraphTurn(Protocol):
    """ChatService 准备好一轮聊天后，交给 graph runtime 使用的数据。"""

    session_id: str
    request_id: str
    user_message: str
    answer_mode: str
    final_decision: str | None
    knowledge_used: bool
    citations: list[Citation]
    retrieval_trace: Any
    scene_metadata: Any


AnswerBuilder = Callable[[PreparedGraphTurn], tuple[str, list[Citation]]]
HistoryLoader = Callable[[PreparedGraphTurn], Sequence[BaseMessage]]
HitlApproveExecutor = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
HitlRespondHandler = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
_HITL_PENDING_ACTIONS = {"tool_approval", "external_api_approval", "clarification"}
_HITL_ALLOWED_ACTIONS = {"approve", "edit", "reject", "respond"}


class HitlResumeError(ValueError):
    """用户恢复等待任务失败时抛出的错误，例如等待点过期或动作不允许。"""


@dataclass(frozen=True)
class HitlWaitInput:
    """让当前会话暂停并等待用户处理时需要的信息。"""

    session_id: str
    request_id: str
    reason: str
    pending_action: str
    allowed_actions: Sequence[str]
    proposed_tool_call: Mapping[str, Any] | None = None
    suggested_responses: Sequence[Mapping[str, Any]] | None = None
    allow_freeform_response: bool = False
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class HitlResumeInput:
    """用户点击批准、拒绝或补充信息后，恢复会话时带上的信息。"""

    session_id: str
    request_id: str
    interrupt_id: str
    action: str
    payload: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class HitlRuntimeResult:
    """HITL 等待或恢复后，返回最新的 graph 状态和本次运行信息。"""

    state: RuntimeGraphState
    config: dict[str, Any]
    run_id: str
    tool_result: dict[str, Any] | None = None


@dataclass(frozen=True)
class RuntimeGraphResult:
    """一次普通 graph 执行完成后的回答、引用和运行信息。"""

    answer: str
    citations: list[Citation]
    state: RuntimeGraphState
    config: dict[str, Any]
    run_id: str


class ChatGraphRuntime:
    """ChatService 使用的 LangGraph 入口，负责保存和读取会话运行状态。"""

    def __init__(
        self,
        *,
        checkpointer: SQLiteLangGraphCheckpointer,
        lifecycle: GraphRunLifecycleRecorder | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.lifecycle = lifecycle or GraphRunLifecycleRecorder()
        # 批准后可能执行写操作或外部 API，同一进程里先一次只处理一个 resume。
        self._resume_lock = Lock()

    @classmethod
    def from_settings(cls, app_settings: AppSettings) -> "ChatGraphRuntime":
        sqlite_path = Path(app_settings.data_dir) / "langgraph.db"
        return cls(checkpointer=SQLiteLangGraphCheckpointer(sqlite_path))

    def invoke(
        self,
        *,
        prepared: PreparedGraphTurn,
        answer_builder: AnswerBuilder,
        history_loader: HistoryLoader,
    ) -> RuntimeGraphResult:
        config = self.build_config(prepared)
        run = self.lifecycle.create_run(
            thread_id=prepared.session_id,
            request_id=prepared.request_id,
            metadata=dict(config["metadata"]),
        )
        self.lifecycle.mark_running(run)
        graph = self._compile_answer_graph(
            prepared=prepared,
            answer_builder=answer_builder,
        )
        try:
            output = graph.invoke(
                self.build_input_state(
                    prepared=prepared,
                    history_loader=history_loader,
                    config=config,
                ),
                config,
            )
        except Exception as exc:
            self.lifecycle.mark_failed(run, exc)
            raise

        self.lifecycle.mark_succeeded(run)
        return RuntimeGraphResult(
            answer=str(output["answer"]),
            citations=list(prepared.citations if prepared.knowledge_used else []),
            state=output,
            config=config,
            run_id=run.run_id,
        )

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
        self.lifecycle.mark_running(run)
        try:
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
            )
            state = self._load_or_build_thread_state(
                session_id=wait.session_id,
                request_id=wait.request_id,
                config=config,
            )
            output = self._persist_state_update(
                state=state,
                config=config,
                update={
                    "request_id": wait.request_id,
                    "status": "waiting_user",
                    "hitl": hitl,
                    "metadata": {
                        **dict(state.get("metadata") or {}),
                        **dict(config["metadata"]),
                    },
                },
            )
        except Exception as exc:
            self.lifecycle.mark_failed(run, exc)
            raise

        self.lifecycle.mark_succeeded(run)
        return HitlRuntimeResult(state=output, config=config, run_id=run.run_id)

    def resume_hitl(
        self,
        *,
        resume: HitlResumeInput,
        approve_executor: HitlApproveExecutor | None = None,
        respond_handler: HitlRespondHandler | None = None,
    ) -> HitlRuntimeResult:
        """处理用户的 approve、reject、respond 动作，并更新当前会话状态。"""
        with self._resume_lock:
            return self._resume_hitl_locked(
                resume=resume,
                approve_executor=approve_executor,
                respond_handler=respond_handler,
            )

    def _resume_hitl_locked(
        self,
        *,
        resume: HitlResumeInput,
        approve_executor: HitlApproveExecutor | None,
        respond_handler: HitlRespondHandler | None,
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
        try:
            state = self._load_or_build_thread_state(
                session_id=resume.session_id,
                request_id=resume.request_id,
                config=config,
                require_checkpoint=True,
            )
            hitl = self._validate_hitl_resume(state=state, resume=resume)
            resume_payload = self._build_resume_payload(resume=resume, hitl=hitl)
            tool_result: dict[str, Any] | None = None
            response_result: dict[str, Any] = {}
            next_answer = str(state.get("answer") or "")
            next_status = "running"

            if resume.action == "approve":
                tool_result = self._execute_approved_tool(
                    hitl=hitl,
                    approve_executor=approve_executor,
                )
                next_status = "succeeded"
                next_answer = "已批准并执行待审批操作。"
            elif resume.action == "reject":
                next_status = "succeeded"
                next_answer = "已拒绝该人工等待项，未执行待审批调用。"
            elif resume.action == "respond":
                if respond_handler is None:
                    raise HitlResumeError("respond_handler is required for respond action.")
                response_result = dict(respond_handler(resume_payload) or {})
                next_status = str(response_result.get("status", "running"))
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
                state=state,
                config=config,
                update={
                    "request_id": resume.request_id,
                    "answer": next_answer,
                    "status": next_status,
                    "hitl": None,
                    "hitl_resume": resume_payload,
                    "metadata": {
                        **dict(state.get("metadata") or {}),
                        **dict(config["metadata"]),
                        "hitl_resume": resume_payload,
                        "hitl_tool_result": tool_result,
                    },
                    **response_state_update,
                },
            )
        except Exception as exc:
            self.lifecycle.mark_failed(run, exc)
            raise

        self.lifecycle.mark_succeeded(run)
        return HitlRuntimeResult(
            state=output,
            config=config,
            run_id=run.run_id,
            tool_result=tool_result,
        )

    def build_config(self, prepared: PreparedGraphTurn) -> dict[str, Any]:
        scene = getattr(prepared.scene_metadata, "scene", None)
        agent = getattr(prepared.scene_metadata, "agent", None)
        return build_runtime_graph_config(
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            metadata={
                "scene": scene,
                "agent": agent,
                "answer_mode": prepared.answer_mode,
                "final_decision": prepared.final_decision,
            },
        )

    def delete_session_thread(self, session_id: str) -> None:
        """删除这个会话在 LangGraph 里的保存状态。"""
        self.checkpointer.delete_thread(session_id)

    def build_input_state(
        self,
        *,
        prepared: PreparedGraphTurn,
        history_loader: HistoryLoader,
        config: dict[str, Any],
    ) -> RuntimeGraphState:
        history_messages = self._history_seed(
            prepared=prepared,
            history_loader=history_loader,
            config=config,
        )
        return build_runtime_graph_state(
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            messages=[
                *history_messages,
                HumanMessage(content=prepared.user_message),
            ],
            knowledge_used=prepared.knowledge_used,
            citations=[citation.model_dump() for citation in prepared.citations],
            retrieval_trace=prepared.retrieval_trace.model_dump(),
            metadata=dict(config["metadata"]),
        )

    def _compile_answer_graph(
        self,
        *,
        prepared: PreparedGraphTurn,
        answer_builder: AnswerBuilder,
    ) -> Any:
        builder = StateGraph(RuntimeGraphState)

        def answer_node(state: RuntimeGraphState) -> dict[str, Any]:
            del state
            answer, citations = answer_builder(prepared)
            # LangGraph 只记运行状态；聊天记录仍由 ChatService 写入 session 表。
            return {
                "answer": answer,
                "citations": [citation.model_dump() for citation in citations],
                "knowledge_used": prepared.knowledge_used,
                "messages": [AIMessage(content=answer)],
            }

        builder.add_node("answer", answer_node)
        builder.add_edge(START, "answer")
        builder.add_edge("answer", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _persist_state_update(
        self,
        *,
        state: RuntimeGraphState,
        config: dict[str, Any],
        update: Mapping[str, Any],
    ) -> RuntimeGraphState:
        """用一个很小的 graph 写状态，确保仍走 LangGraph 的 checkpoint 保存流程。"""
        builder = StateGraph(RuntimeGraphState)

        def update_node(current_state: RuntimeGraphState) -> dict[str, Any]:
            del current_state
            return dict(update)

        builder.add_node("hitl_state_update", update_node)
        builder.add_edge(START, "hitl_state_update")
        builder.add_edge("hitl_state_update", END)
        graph = builder.compile(checkpointer=self.checkpointer)
        return graph.invoke(state, config)

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
            messages=list(values.get("messages") or ()),
            answer=str(values.get("answer") or ""),
            knowledge_used=bool(values.get("knowledge_used", False)),
            citations=list(values.get("citations") or ()),
            retrieval_trace=dict(values.get("retrieval_trace") or {}),
            metadata=dict(values.get("metadata") or {}),
            status=values.get("status", "running"),
            hitl=values.get("hitl"),
            hitl_resume=values.get("hitl_resume"),
        )

    def _validate_hitl_resume(
        self,
        *,
        state: RuntimeGraphState,
        resume: HitlResumeInput,
    ) -> RuntimeHitlState:
        """确认用户要恢复的等待点就是当前正在等待的那个。"""
        if state.get("status") != "waiting_user":
            raise HitlResumeError("Current thread is not waiting for user input.")
        hitl = state.get("hitl")
        if not hitl:
            raise HitlResumeError("Current thread has no HITL payload.")
        if hitl.get("interrupt_id") != resume.interrupt_id:
            raise HitlResumeError("interrupt_id does not match current HITL state.")
        allowed_actions = set(hitl.get("allowed_actions") or ())
        if resume.action not in allowed_actions:
            raise HitlResumeError("action is not allowed for current HITL state.")
        return hitl

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
        approve_executor: HitlApproveExecutor | None,
    ) -> dict[str, Any]:
        """执行用户已批准的工具调用；没有执行器时不允许继续。"""
        proposed_tool_call = hitl.get("proposed_tool_call")
        if not proposed_tool_call:
            raise HitlResumeError("approve requires a proposed_tool_call.")
        if approve_executor is None:
            raise HitlResumeError("approve_executor is required for approve action.")
        result = approve_executor(proposed_tool_call)
        return dict(result or {})

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
        return update

    def _history_seed(
        self,
        *,
        prepared: PreparedGraphTurn,
        history_loader: HistoryLoader,
        config: dict[str, Any],
    ) -> list[BaseMessage]:
        checkpoint = self.checkpointer.get_tuple(config)
        if checkpoint is not None:
            return []
        # 只有第一次进入 LangGraph 时才带入旧历史，避免后续重复塞历史消息。
        return list(history_loader(prepared))

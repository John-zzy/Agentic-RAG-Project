from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.messages import AIMessage

from backend.application.runtime.api.chat.schemas import Citation
from backend.application.runtime.assembly.runtime_parts.agent_state import AgentRuntimeStateProjectionMixin
from backend.application.runtime.assembly.runtime_parts.answer_graph import AnswerGraphMixin
from backend.platform.agent_runtime.chat_graph.contracts import (
    AnswerBuilder,
    HistoryLoader,
    HitlApproveExecutor,
    HitlRespondHandler,
    HitlResumeError,
    HitlResumeInput,
    HitlRuntimeResult,
    HitlWaitInput,
    PreparedGraphTurn,
    RuntimeGraphResult,
    RuntimeGraphRunHandle,
)
from backend.application.runtime.assembly.runtime_parts.hitl import HitlRuntimeMixin
from backend.application.runtime.assembly.runtime_parts.state_store import RuntimeStateStoreMixin
from backend.platform.config.settings import AppSettings
from backend.platform.workflow.langgraph.checkpointer import SQLiteLangGraphCheckpointer
from backend.platform.workflow.langgraph.lifecycle import GraphRunLifecycleRecorder
from backend.platform.workflow.langgraph.state import RuntimeGraphState


class ChatGraphRuntime(
    RuntimeStateStoreMixin,
    AnswerGraphMixin,
    HitlRuntimeMixin,
    AgentRuntimeStateProjectionMixin,
):
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
        self._mark_agent_run_started(run=run, prepared=prepared)
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
                    run_id=run.run_id,
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

    def start_stream_run(
        self,
        *,
        prepared: PreparedGraphTurn,
        history_loader: HistoryLoader,
    ) -> RuntimeGraphRunHandle:
        """为流式回答创建真实 runtime run，并先写入 running checkpoint。"""
        config = self.build_config(prepared)
        run = self.lifecycle.create_run(
            thread_id=prepared.session_id,
            request_id=prepared.request_id,
            metadata=dict(config["metadata"]),
        )
        self._mark_agent_run_started(run=run, prepared=prepared)
        state = self.build_input_state(
            prepared=prepared,
            history_loader=history_loader,
            config=config,
            run_id=run.run_id,
        )
        output = self._persist_state_update(
            state=state,
            config=config,
            update={
                "request_id": prepared.request_id,
                "status": "running",
                "run_id": run.run_id,
                "state_event": "run_start",
                "final_state": None,
            },
        )
        return RuntimeGraphRunHandle(run=run, state=output, config=config)

    def complete_stream_run(
        self,
        *,
        handle: RuntimeGraphRunHandle,
        answer: str,
        citations: Sequence[Citation],
        knowledge_used: bool,
    ) -> RuntimeGraphState:
        """流式回答成功后写入最终 checkpoint，并记录 succeeded lifecycle。"""
        agent_update = self._build_agent_runtime_success_update(
            state=handle.state,
            answer=answer,
            citations=citations,
            knowledge_used=knowledge_used,
        )
        output = self._persist_state_update(
            state=handle.state,
            config=handle.config,
            update={
                "answer": answer,
                "citations": [citation.model_dump() for citation in citations],
                "knowledge_used": knowledge_used,
                "messages": [AIMessage(content=answer)],
                "status": "succeeded",
                "run_id": handle.run_id,
                "state_event": "success",
                "final_state": "succeeded",
                **agent_update,
            },
        )
        self.lifecycle.mark_succeeded(handle.run)
        return output

    def fail_stream_run(
        self,
        *,
        handle: RuntimeGraphRunHandle,
        error: BaseException | str,
    ) -> None:
        """流式回答失败后尽量写入 failed checkpoint，并记录 failed lifecycle。"""
        self.lifecycle.mark_failed(handle.run, error)
        try:
            self._persist_state_update(
                state=handle.state,
                config=handle.config,
                update={
                    "status": "failed",
                    "run_id": handle.run_id,
                    "state_event": "fail",
                    "final_state": "failed",
                    "metadata": {
                        **dict(handle.state.get("metadata") or {}),
                        "error": self.lifecycle.latest(handle.run).error
                        if self.lifecycle.latest(handle.run)
                        else str(error),
                    },
                },
            )
        except Exception:
            # 失败路径不能掩盖原始模型/流式错误；checkpoint 写失败时 lifecycle 仍保留失败事实。
            return



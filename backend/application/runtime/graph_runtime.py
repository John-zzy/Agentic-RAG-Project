from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from backend.application.runtime.api.chat.schemas import Citation
from backend.platform.config.settings import AppSettings
from backend.platform.workflow.langgraph.checkpointer import SQLiteLangGraphCheckpointer
from backend.platform.workflow.langgraph.config import build_runtime_graph_config
from backend.platform.workflow.langgraph.lifecycle import GraphRunLifecycleRecorder
from backend.platform.workflow.langgraph.state import (
    RuntimeGraphState,
    build_runtime_graph_state,
)


class PreparedGraphTurn(Protocol):
    """Graph runtime 消费的 PreparedChatTurn 最小协议。"""

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


@dataclass(frozen=True)
class RuntimeGraphResult:
    """封装一次 graph 执行后的业务结果和底层观测信息。"""

    answer: str
    citations: list[Citation]
    state: RuntimeGraphState
    config: dict[str, Any]
    run_id: str


class ChatGraphRuntime:
    """application 层 LangGraph facade，负责 graph 输入、配置和生命周期。"""

    def __init__(
        self,
        *,
        checkpointer: SQLiteLangGraphCheckpointer,
        lifecycle: GraphRunLifecycleRecorder | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.lifecycle = lifecycle or GraphRunLifecycleRecorder()

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
        """按会话 ID 清理对应 LangGraph thread 的持久化状态。"""
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
            # LangGraph 只保存运行时状态；对外可查的 chat_messages 仍由 ChatService 统一写入。
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
        # 没有 checkpoint 时才加载旧历史，避免把所有历史会话一次性迁移到 LangGraph。
        return list(history_loader(prepared))

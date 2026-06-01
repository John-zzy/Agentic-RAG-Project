from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from backend.application.runtime.api.chat.schemas import ChatRequest, Citation, RetrievalTrace
from backend.application.runtime.graph_runtime import ChatGraphRuntime
from backend.application.runtime.service import ChatService
from backend.application.runtime.stream_events import (
    GraphRuntimeStreamEvent,
    GraphStreamEventMapper,
)
from backend.platform.config.settings import AppSettings
from backend.platform.memory.base.session_store import SQLiteSessionStore
from backend.platform.memory.chat.prompt_context import PromptContextBuilder
from backend.platform.workflow.langgraph.checkpointer import SQLiteLangGraphCheckpointer
from backend.platform.workflow.langgraph.config import (
    DEFAULT_RUNTIME_CHECKPOINT_NS,
    build_runtime_graph_config,
)
from backend.platform.workflow.langgraph.lifecycle import (
    GraphRunLifecycleRecorder,
)
from backend.platform.workflow.langgraph.state import (
    RuntimeGraphState,
    build_runtime_graph_state,
)
from backend.scenes.base import SceneDefinition, SceneFallbackPolicy, SceneRetrievalPolicy
from backend.tests.test_support import make_test_runtime_dir


@dataclass(frozen=True)
class _SceneMetadata:
    scene: str = "generic_assistant"
    agent: str | None = None


@dataclass(frozen=True)
class _PreparedGraphTurn:
    session_id: str
    request_id: str
    user_message: str
    answer_mode: str
    final_decision: str | None
    knowledge_used: bool
    citations: list[Citation]
    retrieval_trace: RetrievalTrace
    scene_metadata: _SceneMetadata = _SceneMetadata()


class _SearchRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def search(self, query: str, **kwargs: Any) -> list[Document]:
        del query, kwargs
        return self._documents


class _FakeModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.invoke_runnable_calls: list[dict[str, Any]] = []

    def get_runnable(
        self,
        complexity: str = "simple",
        prompt_template: Any | None = None,
        *,
        output_parser: Any | None = None,
    ) -> Any:
        del complexity, output_parser
        answer = RunnableLambda(lambda _: self.answer)
        if prompt_template is None:
            return answer
        return prompt_template | answer

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> str:
        self.invoke_runnable_calls.append({"input": input, "config": config})
        return str(runnable.invoke(input, config=config))


def test_runtime_graph_state_exposes_minimal_chat_execution_fields() -> None:
    assert set(RuntimeGraphState.__annotations__) == {
        "session_id",
        "request_id",
        "messages",
        "answer",
        "knowledge_used",
        "citations",
        "retrieval_trace",
        "metadata",
    }

    state = build_runtime_graph_state(
        session_id="session-state",
        request_id="req-state",
        messages=[HumanMessage(content="hello")],
        answer="world",
        knowledge_used=True,
        citations=[{"document_id": "doc-1", "chunk_id": "chunk-1"}],
        retrieval_trace={"final_decision": "evidence_answer"},
        metadata={"scene": "generic_assistant"},
    )

    assert state["session_id"] == "session-state"
    assert state["request_id"] == "req-state"
    assert state["messages"][0].content == "hello"
    assert state["answer"] == "world"
    assert state["knowledge_used"] is True
    assert state["citations"] == [{"document_id": "doc-1", "chunk_id": "chunk-1"}]
    assert state["retrieval_trace"] == {"final_decision": "evidence_answer"}
    assert state["metadata"] == {"scene": "generic_assistant"}


def test_runtime_graph_state_defaults_keep_non_evidence_branches_representable() -> None:
    state = build_runtime_graph_state(
        session_id="session-no-evidence",
        request_id="req-no-evidence",
        answer="I need more information.",
        retrieval_trace={"final_decision": "ask_user"},
    )

    assert state == {
        "session_id": "session-no-evidence",
        "request_id": "req-no-evidence",
        "messages": [],
        "answer": "I need more information.",
        "knowledge_used": False,
        "citations": [],
        "retrieval_trace": {"final_decision": "ask_user"},
        "metadata": {},
    }


def test_runtime_graph_config_binds_thread_id_to_session_id_and_metadata_to_request() -> None:
    config = build_runtime_graph_config(
        session_id="session-config",
        request_id="req-config",
        metadata={
            "request_id": "stale-request",
            "session_id": "stale-session",
            "scene": "generic_assistant",
        },
    )

    assert config["configurable"]["thread_id"] == "session-config"
    assert config["configurable"]["checkpoint_ns"] == DEFAULT_RUNTIME_CHECKPOINT_NS
    assert config["metadata"] == {
        "request_id": "req-config",
        "session_id": "session-config",
        "scene": "generic_assistant",
    }


def test_graph_run_lifecycle_records_success_path_with_thread_and_request() -> None:
    recorder = GraphRunLifecycleRecorder()

    run = recorder.create_run(
        thread_id="session-success",
        request_id="req-success",
        metadata={"scene": "generic_assistant"},
        run_id="run-success",
    )
    recorder.mark_running(run)
    recorder.mark_succeeded(run)

    events = recorder.events(run.run_id)
    assert tuple(event.status for event in events) == ("created", "running", "succeeded")
    assert all(event.thread_id == "session-success" for event in events)
    assert all(event.request_id == "req-success" for event in events)
    assert all(event.metadata == {"scene": "generic_assistant"} for event in events)
    assert recorder.events_for_thread(
        thread_id="session-success",
        request_id="req-success",
    ) == events
    assert recorder.latest(run) is events[-1]


def test_graph_run_lifecycle_records_failed_path_with_thread_request_and_error() -> None:
    recorder = GraphRunLifecycleRecorder()

    run = recorder.create_run(
        thread_id="session-failed",
        request_id="req-failed",
        metadata={"scene": "generic_assistant"},
        run_id="run-failed",
    )
    recorder.mark_running(run)
    failed = recorder.mark_failed(run, RuntimeError("model invocation failed"))

    events = recorder.events(run.run_id)
    assert tuple(event.status for event in events) == ("created", "running", "failed")
    assert all(event.thread_id == "session-failed" for event in events)
    assert all(event.request_id == "req-failed" for event in events)
    assert failed.error == "RuntimeError: model invocation failed"
    assert recorder.latest(run) == failed


def test_chat_graph_runtime_invokes_answer_graph_and_persists_checkpoint_metadata() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-checkpoint")
    prepared = _prepared_graph_turn(
        session_id="session-graph",
        request_id="req-graph",
        user_message="新问题",
    )

    result = runtime.invoke(
        prepared=prepared,
        answer_builder=lambda turn: ("graph answer", turn.citations),
        history_loader=lambda _: [
            HumanMessage(content="旧问题"),
            AIMessage(content="旧答案"),
        ],
    )
    restored = runtime.checkpointer.get_tuple(
        {
            "configurable": {
                "thread_id": "session-graph",
                "checkpoint_ns": DEFAULT_RUNTIME_CHECKPOINT_NS,
            }
        }
    )

    assert result.config["configurable"]["thread_id"] == "session-graph"
    assert result.config["metadata"]["request_id"] == "req-graph"
    assert result.state["answer"] == "graph answer"
    assert [message.content for message in result.state["messages"]] == [
        "旧问题",
        "旧答案",
        "新问题",
        "graph answer",
    ]
    assert restored is not None
    assert restored.metadata["request_id"] == "req-graph"
    assert restored.metadata["session_id"] == "session-graph"
    assert restored.checkpoint["channel_values"]["answer"] == "graph answer"


def test_chat_graph_runtime_seeds_legacy_history_only_without_checkpoint() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-history-seed")
    loader_calls = 0

    def history_loader(_: _PreparedGraphTurn) -> list[BaseMessage]:
        nonlocal loader_calls
        loader_calls += 1
        return [HumanMessage(content="legacy question")]

    first = _prepared_graph_turn(
        session_id="session-seed",
        request_id="req-seed-1",
        user_message="first graph question",
    )
    second = _prepared_graph_turn(
        session_id="session-seed",
        request_id="req-seed-2",
        user_message="second graph question",
    )

    runtime.invoke(
        prepared=first,
        answer_builder=lambda turn: ("first answer", turn.citations),
        history_loader=history_loader,
    )
    result = runtime.invoke(
        prepared=second,
        answer_builder=lambda turn: ("second answer", turn.citations),
        history_loader=history_loader,
    )

    assert loader_calls == 1
    assert [message.content for message in result.state["messages"]] == [
        "legacy question",
        "first graph question",
        "first answer",
        "second graph question",
        "second answer",
    ]


def test_chat_service_non_streaming_answer_runs_through_graph_runtime() -> None:
    runtime_dir = make_test_runtime_dir("runtime-graph-chat-service")
    sqlite_path = runtime_dir / "sessions.db"
    graph_runtime = _build_chat_graph_runtime("runtime-graph-chat-service")
    service = ChatService(
        scene_definition=_build_scene_definition(
            [
                Document(
                    page_content="Graph runtime 文档证据。",
                    metadata={
                        "document_id": "doc-graph",
                        "source_path": "graph.md",
                        "namespace": "documents",
                        "chunk_id": "chunk-graph",
                        "chunk_index": 0,
                        "score": 0.91,
                    },
                )
            ]
        ),
        app_settings=AppSettings(
            data_dir=runtime_dir,
            session={"sqlite_path": sqlite_path, "window_size": 3},
        ),
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=_FakeModel("这是 graph runtime 生成的回答。"),
        graph_runtime=graph_runtime,
    )

    response = service.chat(ChatRequest(message="请回答 graph runtime"))
    restored = graph_runtime.checkpointer.get_tuple(
        {
            "configurable": {
                "thread_id": response.session_id,
                "checkpoint_ns": DEFAULT_RUNTIME_CHECKPOINT_NS,
            }
        }
    )

    assert response.answer == "这是 graph runtime 生成的回答。\n\n参考来源：[1]"
    assert response.knowledge_used is True
    assert len(response.citations) == 1
    assert restored is not None
    assert restored.metadata["request_id"] == response.request_id
    assert restored.metadata["session_id"] == response.session_id
    assert restored.checkpoint["channel_values"]["answer"] == response.answer
    assert restored.checkpoint["channel_values"]["retrieval_trace"]["final_decision"] == (
        "answer_with_evidence"
    )
    assert service.session_store.count_turns(response.session_id) == 1
    assert service.session_store.count_messages(response.session_id) == 2
    messages, total_messages = service.session_store.get_session_messages(
        session_id=response.session_id,
        limit=10,
    )
    assert total_messages == 2
    assert [message.message_type for message in messages] == ["human", "ai"]
    assert messages[1].content == response.answer
    assert messages[1].citations[0]["citation_id"] == "chunk-graph"


def test_graph_stream_event_mapper_keeps_business_event_protocol() -> None:
    mapper = GraphStreamEventMapper()

    events = list(
        mapper.map_events(
            [
                GraphRuntimeStreamEvent("graph_run_created", {"request_id": "req-stream"}),
                GraphRuntimeStreamEvent("history_snapshot", {"messages": []}),
                GraphRuntimeStreamEvent(
                    "retrieval_tool_result",
                    {"retrieval_trace": {"final_decision": "answer_with_evidence"}},
                ),
                GraphRuntimeStreamEvent("answer_chunk", {"delta": "hello"}),
                GraphRuntimeStreamEvent(
                    "graph_run_succeeded",
                    {"retrieval_trace": {"final_decision": "answer_with_evidence"}},
                ),
                GraphRuntimeStreamEvent(
                    "graph_run_failed",
                    {"code": "MODEL_INVOCATION_FAILED", "request_id": "req-stream"},
                ),
            ]
        )
    )

    assert [event.event for event in events] == [
        "start",
        "history",
        "tool",
        "chunk",
        "done",
        "error",
    ]
    assert all(not event.event.startswith("graph_") for event in events)
    assert all("graph_run_created" not in event.data for event in events)


def _build_chat_graph_runtime(test_name: str) -> ChatGraphRuntime:
    runtime_dir = make_test_runtime_dir(test_name)
    return ChatGraphRuntime(
        checkpointer=SQLiteLangGraphCheckpointer(runtime_dir / "langgraph.db")
    )


def _prepared_graph_turn(
    *,
    session_id: str,
    request_id: str,
    user_message: str,
) -> _PreparedGraphTurn:
    citation = _citation()
    return _PreparedGraphTurn(
        session_id=session_id,
        request_id=request_id,
        user_message=user_message,
        answer_mode="evidence_answer",
        final_decision="answer_with_evidence",
        knowledge_used=True,
        citations=[citation],
        retrieval_trace=_retrieval_trace(citations=[citation]),
    )


def _citation() -> Citation:
    return Citation(
        index=1,
        citation_id="chunk-graph",
        namespace="documents",
        source_kind="document_chunk",
        source_name="graph.md",
        source_path="graph.md",
        document_id="doc-graph",
        chunk_id="chunk-graph",
        chunk_index=0,
        snippet="Graph runtime 文档证据。",
        score=0.91,
        vector_score=0.91,
        keyword_score=None,
        vector_rank=1,
        keyword_rank=None,
        rerank_score=None,
        matched_by=["vector"],
        rank=1,
    )


def _retrieval_trace(*, citations: list[Citation]) -> RetrievalTrace:
    return RetrievalTrace(
        original_query="原始问题",
        final_query="原始问题",
        tool_call_count=1,
        candidate_tools=["knowledge_document_search"],
        exit_reason="sufficient",
        final_decision="answer_with_evidence",
        success=True,
        raw_candidates_count=1,
        filtered_candidates_count=1,
        citations=citations,
        knowledge_used=True,
    )


def _build_scene_definition(documents: list[Document]) -> SceneDefinition:
    return SceneDefinition(
        scene="generic_assistant",
        name="Generic Assistant",
        description="Runtime graph test scene.",
        build_retriever=lambda: _SearchRetriever(documents),
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda mounted: ("knowledge_document_search",),
        system_prompt="你是测试助手。",
        fallback_policy=SceneFallbackPolicy(no_hit_message="没有可用证据。"),
        infer_complexity=lambda message: "simple",
        retrieval_policy=SceneRetrievalPolicy(min_relevance_score=0.0),
    )

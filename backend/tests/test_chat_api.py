import json
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda, RunnableSerializable

from backend.application.runtime.api.app import create_app
from backend.application.runtime import ChatService, SceneChatService, build_default_scene_registry
from backend.platform.config.settings import AppSettings
from backend.platform.knowledge.repositories import VectorStoreFactory
from backend.platform.memory.base.session_store import SQLiteSessionStore
from backend.platform.memory.chat.prompt_context import PromptContextBuilder
from backend.platform.rag.retrieval.documents import DocumentChunkRetrievalResult
from backend.platform.rag.retrieval.documents.service import DocumentRetrievalService
from backend.platform.search_foundation import VectorSearchResult, VectorStoreDocument
from backend.scenes.base import SceneDefinition, SceneFallbackPolicy, SceneRetrievalPolicy
from backend.scenes.ecommerce.knowledge_service import create_knowledge_service
from backend.tests.test_support import make_test_runtime_dir


def _result(
    doc_id: str,
    content: str,
    score: float,
    metadata: dict[str, Any],
) -> VectorSearchResult:
    return VectorSearchResult(
        document=VectorStoreDocument(
            id=doc_id,
            content=content,
            metadata=metadata,
        ),
        score=score,
    )


def _build_document_retrieval_service(app_settings: AppSettings) -> DocumentRetrievalService:
    return DocumentRetrievalService(
        app_settings=app_settings,
        vector_repository=VectorStoreFactory.create_document_chunk_vector_repository(app_settings),
        chunk_source=VectorStoreFactory.create_active_document_chunk_source(app_settings),
    )


class FakeKnowledgeService:
    def __init__(
        self,
        products: list[VectorSearchResult] | None = None,
        reviews: list[VectorSearchResult] | None = None,
        documents: list[VectorSearchResult] | None = None,
    ) -> None:
        self._products = products or []
        self._reviews = reviews or []
        del documents

    def search_products(self, query: str, top_k: int | None = None) -> list[VectorSearchResult]:
        return self._products

    def search_reviews(self, query: str, top_k: int | None = None) -> list[VectorSearchResult]:
        return self._reviews

    def search_orders(self, query: str, top_k: int | None = None) -> list[VectorSearchResult]:
        return []

class FakeDocumentRetrievalService:
    def __init__(self, documents: list[VectorSearchResult] | None = None) -> None:
        self._documents = documents or []
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        minimum_relevance: float | None = None,
        recall_strategy: str = "hybrid",
    ) -> list[DocumentChunkRetrievalResult]:
        del query, namespace
        self.calls.append(
            {
                "top_k": top_k,
                "minimum_relevance": minimum_relevance,
                "recall_strategy": recall_strategy,
            }
        )
        documents = self._documents
        if minimum_relevance is not None:
            documents = [
                result
                for result in documents
                if result.score is None or float(result.score) >= minimum_relevance
            ]
        return [
            DocumentChunkRetrievalResult(
                document=result.document,
                score=result.score,
                vector_score=result.score,
                vector_rank=index,
                matched_by=["vector"],
            )
            for index, result in enumerate(documents[:top_k], start=1)
        ]


class FakeModel:
    def __init__(self, answer: str = "mock-answer", stream_chunks: list[str] | None = None) -> None:
        self.answer = answer
        self.get_runnable_calls: list[str] = []
        self.invoke_runnable_calls: list[dict[str, Any]] = []
        self.stream_runnable_calls: list[dict[str, Any]] = []
        self.history_recorder = HistoryRecorder()
        self.stream_chunks = stream_chunks or [answer]

    def get_runnable(
        self,
        complexity: str = "simple",
        prompt_template: Any | None = None,
        *,
        output_parser: Any | None = None,
    ):
        del output_parser
        self.get_runnable_calls.append(complexity)
        if prompt_template is None:
            return FakeAnswerRunnable(
                answer=self.answer,
                stream_chunks=self.stream_chunks,
                history_recorder=self.history_recorder,
            )
        return prompt_template | FakeAnswerRunnable(
            answer=self.answer,
            stream_chunks=self.stream_chunks,
            history_recorder=self.history_recorder,
        )

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> str:
        self.invoke_runnable_calls.append({"runnable": runnable, "input": input, "config": config})
        return runnable.invoke(input, config=config)

    def stream_runnable(
        self,
        runnable: Any,
        input: Any,
        *,
        config: Any | None = None,
    ):
        self.stream_runnable_calls.append({"runnable": runnable, "input": input, "config": config})
        yield from runnable.stream(input, config=config)

    def build_chat_model_for_complexity(
        self,
        complexity: str = "simple",
    ):
        raise AssertionError("runtime should use get_runnable instead of build_chat_model_for_complexity")

    def invoke_template(self, prompt_template: Any, variables: dict[str, Any], complexity: str = "simple") -> str:
        raise AssertionError("runtime should use invoke_runnable instead of invoke_template")

    def stream_template(
        self,
        prompt_template: Any,
        variables: dict[str, Any],
        complexity: str = "simple",
    ):
        raise AssertionError("runtime should use stream_runnable instead of stream_template")


class FakeAnswerRunnable(RunnableSerializable[Any, str]):
    answer: str
    stream_chunks: list[str]
    history_recorder: Any

    def invoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> str:
        del config, kwargs
        self._record_prompt_messages(input)
        return self.answer

    def stream(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ):
        del config, kwargs
        self._record_prompt_messages(input)
        for chunk in self.stream_chunks:
            yield chunk

    def _record_prompt_messages(self, input: Any) -> None:
        if isinstance(input, ChatPromptValue):
            messages = input.to_messages()
        elif isinstance(input, list) and all(isinstance(message, BaseMessage) for message in input):
            messages = list(input)
        else:
            return

        history_messages = [
            message
            for message in messages[:-1]
            if isinstance(message, BaseMessage) and message.type in {"human", "ai"}
        ]
        self.history_recorder.snapshots.append(history_messages)


class HistoryRecorder:
    def __init__(self) -> None:
        self.snapshots: list[list[BaseMessage]] = []


def _parse_sse_events(raw_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_payload = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            if line.startswith("data: "):
                data_payload = line[len("data: ") :]
        events.append({"event": event_name, "data": data_payload})
    return events


def _build_chat_service(
    test_name: str,
    knowledge_service: FakeKnowledgeService,
    document_retrieval_service: FakeDocumentRetrievalService,
    model: FakeModel,
) -> SceneChatService:
    runtime_dir = make_test_runtime_dir(test_name)
    files_root = runtime_dir / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    for result in document_retrieval_service._documents:
        source_path = result.document.metadata.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            continue
        file_path = files_root / source_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.write_text(result.document.content, encoding="utf-8")
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        app={
            "active_scene": "generic_assistant",
        },
        session={
            "sqlite_path": sqlite_path,
            "window_size": 3,
        }
    )
    return SceneChatService(
        scene_registry=build_default_scene_registry(
            app_settings=app_settings,
            knowledge_service=knowledge_service,  # type: ignore[arg-type]
            document_retrieval_service=document_retrieval_service,  # type: ignore[arg-type]
        ),
        app_settings=app_settings,
        knowledge_service=knowledge_service,  # type: ignore[arg-type]
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=model,
    )


def test_chat_api_success_path() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="P001 手机，续航强，电池 5000mAh。",
                score=0.92,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "doc.txt",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-doc-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="推荐 P001，续航表现较好。")
    service = _build_chat_service("chat-api-success", knowledge, document_retrieval_service, model)
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "推荐续航好的手机"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["session_id"]
    assert payload["request_id"]
    assert payload["answer"] == "推荐 P001，续航表现较好。\n\n参考来源：[1]"
    assert payload["knowledge_used"] is True
    assert payload["scene"] == "generic_assistant"
    assert payload["agent"] is None
    assert len(payload["citations"]) == 1
    assert payload["citations"][0] == {
        "index": 1,
        "citation_id": "chunk-doc-1",
        "namespace": "documents",
        "source_kind": "document_chunk",
        "source_name": "doc.txt",
        "source_path": "doc.txt",
        "document_id": "doc-1",
        "chunk_id": "chunk-doc-1",
        "chunk_index": 0,
        "snippet": "P001 手机，续航强，电池 5000mAh。",
        "score": 0.92,
        "vector_score": 0.92,
        "keyword_score": None,
        "vector_rank": 1,
        "keyword_rank": None,
        "matched_by": ["vector"],
        "rank": 1,
    }
    trace = payload["retrieval_trace"]
    assert trace["original_query"] == "推荐续航好的手机"
    assert trace["final_query"] == "推荐续航好的手机"
    assert trace["rewritten_query"] is None
    assert trace["tool_call_count"] == 1
    assert trace["candidate_tools"] == ["knowledge_document_search"]
    assert trace["exit_reason"] == "sufficient"
    assert trace["knowledge_used"] is True
    assert trace["raw_candidates_count"] == 1
    assert trace["filtered_candidates_count"] == 1
    assert trace["citations"] == payload["citations"]
    assert trace["top_k_chunks"][0]["citation_id"] == "chunk-doc-1"
    assert trace["top_k_chunks"][0]["chunk_id"] == "chunk-doc-1"
    assert trace["top_k_chunks"][0]["score"] == 0.92
    assert trace["rounds"][0]["tool_name"] == "knowledge_document_search"
    assert trace["rounds"][0]["raw_candidates_count"] == 1
    assert trace["rounds"][0]["filtered_candidates_count"] == 1
    saved_session = service.session_store.get_session(payload["session_id"])
    assert saved_session is not None
    assert saved_session.mounted_knowledge_sources == ("documents",)
    assert model.get_runnable_calls == ["simple"]
    assert len(model.invoke_runnable_calls) == 1


def test_chat_api_sse_success_path_returns_structured_events() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="P001 手机，续航强，电池 5000mAh。",
                score=0.92,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "doc.txt",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-doc-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(
        answer="推荐 P001，续航表现较好。[1]",
        stream_chunks=["推荐 P001，", "续航表现较好。[1]"],
    )
    service = _build_chat_service("chat-api-sse-success", knowledge, document_retrieval_service, model)
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "推荐续航好的手机", "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["start", "history", "tool", "chunk", "chunk", "done"]

    start_data = events[0]["data"]
    history_data = events[1]["data"]
    tool_data = events[2]["data"]
    done_data = events[-1]["data"]
    assert start_data
    assert history_data
    assert tool_data
    assert done_data

    start_payload = json.loads(start_data)
    history_payload = json.loads(history_data)
    tool_payload = json.loads(tool_data)
    done_payload = json.loads(done_data)
    assert start_payload["session_id"]
    assert start_payload["request_id"]
    assert start_payload["knowledge_used"] is True
    assert history_payload["message_count"] == 0
    assert history_payload["messages"] == []
    assert tool_payload["knowledge_used"] is True
    assert tool_payload["documents"] == 1
    assert tool_payload["rounds"][0]["tool_name"] == "knowledge_document_search"
    assert tool_payload["retrieval_trace"]["tool_call_count"] == 1
    assert tool_payload["retrieval_trace"]["top_k_chunks"][0]["citation_id"] == "chunk-doc-1"
    assert tool_payload["retrieval_policy"] == {
        "top_k": 5,
        "min_relevance_score": 0.8,
        "recall_strategy": "hybrid",
        "no_hit_strategy": "ask_user",
        "rerank_enabled": False,
        "rerank_top_n": None,
    }
    assert done_payload["answer"] == "推荐 P001，续航表现较好。[1]"
    assert done_payload["knowledge_used"] is True
    assert done_payload["scene"] == "generic_assistant"
    assert len(done_payload["citations"]) == 1
    assert done_payload["retrieval_trace"] == tool_payload["retrieval_trace"]
    assert done_payload["retrieval_trace"]["citations"] == done_payload["citations"]
    saved_turns, total_turns = service.session_store.get_session_detail(
        done_payload["session_id"],
        limit=10,
    )
    assert total_turns == 1
    assert saved_turns[0].assistant_answer == "推荐 P001，续航表现较好。[1]"
    assert model.get_runnable_calls == ["simple"]
    assert len(model.stream_runnable_calls) == 1


def test_chat_api_validation_error_when_message_missing() -> None:
    service = _build_chat_service(
        "chat-api-validation-error",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_api_no_hit_fallback_sets_knowledge_used_false() -> None:
    model = FakeModel(answer="unused")
    service = _build_chat_service("chat-api-no-hit", FakeKnowledgeService(), FakeDocumentRetrievalService(), model)
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "火星基地快递多久到"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert "暂时没有检索到足够相关的文档知识" in payload["answer"]
    assert payload["scene"] == "generic_assistant"
    assert payload["agent"] is None
    assert model.get_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_sse_no_hit_uses_fallback_without_model_streaming() -> None:
    model = FakeModel(answer="unused", stream_chunks=["unused"])
    service = _build_chat_service(
        "chat-api-sse-no-hit",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "火星基地快递多久到", "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["start", "history", "tool", "chunk", "done"]

    history_payload = json.loads(events[1]["data"])
    tool_payload = json.loads(events[2]["data"])
    chunk_payload = json.loads(events[3]["data"])
    done_payload = json.loads(events[4]["data"])
    assert history_payload["messages"] == []
    assert history_payload["message_count"] == 0
    assert tool_payload["knowledge_used"] is False
    assert tool_payload["documents"] == 0
    assert tool_payload["retrieval_trace"]["knowledge_used"] is False
    assert tool_payload["retrieval_trace"]["filtered_candidates_count"] == 0
    assert "暂时没有检索到足够相关的文档知识" in chunk_payload["delta"]
    assert done_payload["knowledge_used"] is False
    assert done_payload["citations"] == []
    assert done_payload["retrieval_trace"] == tool_payload["retrieval_trace"]
    assert model.get_runnable_calls == []
    assert model.stream_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_sse_error_path_keeps_runtime_event_order() -> None:
    class ErrorModel(FakeModel):
        def stream_runnable(
            self,
            runnable: Any,
            input: Any,
            *,
            config: Any | None = None,
        ):
            self.stream_runnable_calls.append({"runnable": runnable, "input": input, "config": config})
            raise ValueError("Model returned empty streaming content")

    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="P001 手机，续航强，电池 5000mAh。",
                score=0.92,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "doc.txt",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-doc-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = ErrorModel(answer="unused", stream_chunks=["unused"])
    service = _build_chat_service("chat-api-sse-error", knowledge, document_retrieval_service, model)
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "推荐续航好的手机", "stream": True})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["start", "history", "tool", "error"]
    error_payload = json.loads(events[-1]["data"])
    assert error_payload["code"] == "MODEL_EMPTY_RESPONSE"


def test_chat_api_non_stream_response_and_session_persistence_do_not_regress() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="AeroPhone X 当前有货，售价 4599 元。",
                score=0.93,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="AeroPhone X 当前有货，售价 4599 元。[1]")
    service = _build_chat_service(
        "chat-api-non-stream-regression",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "AeroPhone X 多少钱", "stream": False})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "AeroPhone X 当前有货，售价 4599 元。[1]"
    assert payload["knowledge_used"] is True
    turns, total_turns = service.session_store.get_session_detail(payload["session_id"], limit=10)
    assert total_turns == 1
    assert turns[0].assistant_answer == "AeroPhone X 当前有货，售价 4599 元。[1]"
    assert turns[0].retrieval_snippets[0]["citation_id"] == "chunk-manual-1"


def test_chat_api_uses_message_history_for_follow_up_turns() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="AeroPhone X 当前有货，售价 4599 元。",
                score=0.93,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="AeroPhone X 当前有货，售价 4599 元。[1]")
    service = _build_chat_service(
        "chat-api-message-history-follow-up",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "AeroPhone X 多少钱"})
        assert first.status_code == 200
        session_id = first.json()["session_id"]

        second = client.post(
            "/chat",
            json={"message": "那它现在有货吗", "session_id": session_id},
        )

    assert second.status_code == 200
    assert len(model.history_recorder.snapshots) >= 2
    second_history = model.history_recorder.snapshots[-1]
    assert len(second_history) == 2
    assert second_history[0].type == "human"
    assert second_history[0].content == "AeroPhone X 多少钱"
    assert second_history[1].type == "ai"
    assert second_history[1].content == "AeroPhone X 当前有货，售价 4599 元。[1]"

    messages = service.session_store.get_messages(session_id)
    assert len(messages) == 4
    assert messages[0].type == "human"
    assert messages[1].type == "ai"
    assert messages[2].type == "human"
    assert messages[3].type == "ai"


def test_chat_api_message_history_respects_window_size() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="AeroPhone X 当前有货，售价 4599 元。",
                score=0.93,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="AeroPhone X 当前有货，售价 4599 元。[1]")
    service = _build_chat_service(
        "chat-api-message-history-window",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        session_id: str | None = None
        for index in range(1, 4):
            response = client.post(
                "/chat",
                json={
                    "message": f"第{index}轮问题",
                    **({"session_id": session_id} if session_id else {}),
                },
            )
            assert response.status_code == 200
            session_id = response.json()["session_id"]

    assert session_id is not None
    latest_history = model.history_recorder.snapshots[-1]
    assert len(latest_history) == 4
    assert [message.content for message in latest_history] == [
        "第1轮问题",
        "AeroPhone X 当前有货，售价 4599 元。[1]",
        "第2轮问题",
        "AeroPhone X 当前有货，售价 4599 元。[1]",
    ]
    messages = service.session_store.get_messages(session_id)
    assert len(messages) == 6


def test_chat_api_sse_history_event_uses_trimmed_message_window() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="AeroPhone X 当前有货，售价 4599 元。",
                score=0.93,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(
        answer="AeroPhone X 当前有货，售价 4599 元。[1]",
        stream_chunks=["AeroPhone X 当前", "有货，售价 4599 元。[1]"],
    )
    service = _build_chat_service(
        "chat-api-sse-history-window",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        session_id: str | None = None
        for index in range(1, 4):
            response = client.post(
                "/chat",
                json={
                    "message": f"第{index}轮问题",
                    "stream": index == 3,
                    **({"session_id": session_id} if session_id else {}),
                },
            )
            assert response.status_code == 200
            if index < 3:
                session_id = response.json()["session_id"]
            else:
                events = _parse_sse_events(response.text)

    assert session_id is not None
    assert [event["event"] for event in events] == ["start", "history", "tool", "chunk", "chunk", "done"]
    history_payload = json.loads(events[1]["data"])
    assert history_payload["message_count"] == 4
    assert [message["content"] for message in history_payload["messages"]] == [
        "第1轮问题",
        "AeroPhone X 当前有货，售价 4599 元。[1]",
        "第2轮问题",
        "AeroPhone X 当前有货，售价 4599 元。[1]",
    ]


def test_session_management_endpoints() -> None:
    service = _build_chat_service(
        "chat-api-session-endpoints",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        create_response = client.post("/sessions")
        assert create_response.status_code == 200
        create_payload = create_response.json()
        session_id = create_payload["session_id"]
        assert session_id
        assert create_payload["scene"] == "generic_assistant"
        assert create_payload["mounted_knowledge_sources"] == ["documents"]
        assert service.session_store.get_session(session_id) is not None

        empty_session_response = client.get(f"/sessions/{session_id}")
        assert empty_session_response.status_code == 200
        assert empty_session_response.json()["scene"] == "generic_assistant"
        assert empty_session_response.json()["mounted_knowledge_sources"] == ["documents"]
        assert empty_session_response.json()["total_messages"] == 0
        assert empty_session_response.json()["messages"] == []

        chat_response = client.post("/chat", json={"message": "你好", "session_id": session_id})
        assert chat_response.status_code == 200

        populated_session_response = client.get(f"/sessions/{session_id}")
        assert populated_session_response.status_code == 200
        payload = populated_session_response.json()
        assert payload["session_id"] == session_id
        assert payload["mounted_knowledge_sources"] == ["documents"]
        assert payload["total_messages"] == 2
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["type"] == "human"
        assert payload["messages"][0]["content"] == "你好"
        assert payload["messages"][0]["knowledge_used"] is None
        assert payload["messages"][1]["type"] == "ai"
        assert payload["messages"][1]["request_id"] == chat_response.json()["request_id"]
        assert payload["messages"][1]["content"] == chat_response.json()["answer"]
        assert payload["messages"][1]["knowledge_used"] is False
        assert payload["messages"][1]["citations"] == []

        delete_response = client.delete(f"/sessions/{session_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["deleted_messages"] == 2

        after_delete_response = client.get(f"/sessions/{session_id}")
        assert after_delete_response.status_code == 200
        assert after_delete_response.json()["total_messages"] == 0


def test_chat_api_rejects_expired_session() -> None:
    service = _build_chat_service(
        "chat-api-expired-session",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    service.session_store.create_session(
        session_id="expired-session",
        now="2026-04-23T00:00:00+00:00",
    )
    service.session_store.cleanup_expired_sessions(
        now="2026-04-23T01:00:00+00:00",
        timeout_minutes=30,
        limit=10,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "你好", "session_id": "expired-session"})

    assert response.status_code == 409
    payload = response.json()
    assert payload["detail"]["code"] == "SESSION_EXPIRED"


def test_chat_api_rejects_unknown_session_id() -> None:
    service = _build_chat_service(
        "chat-api-missing-session",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "你好", "session_id": "missing-session"})

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["code"] == "SESSION_NOT_FOUND"


def test_list_scenes_endpoint_returns_available_scene_metadata() -> None:
    service = _build_chat_service(
        "chat-api-list-scenes",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.get("/scenes")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_scene"] == "generic_assistant"
    assert [scene["scene"] for scene in payload["scenes"]] == [
        "generic_assistant",
        "ecommerce",
    ]
    assert payload["scenes"][0]["is_default"] is True
    assert payload["scenes"][1]["is_default"] is False


def test_create_session_rejects_unknown_scene() -> None:
    service = _build_chat_service(
        "chat-api-unknown-scene",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/sessions", json={"scene": "unknown_scene"})

    assert response.status_code == 400
    payload = response.json()
    assert payload["detail"]["code"] == "UNKNOWN_SCENE"


def test_create_session_accepts_explicit_mounted_knowledge_sources() -> None:
    service = _build_chat_service(
        "chat-api-mounted-sources",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/sessions",
            json={
                "scene": "generic_assistant",
                "mounted_knowledge_sources": ["ecommerce", "documents", "documents"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mounted_knowledge_sources"] == ["documents", "ecommerce"]
    saved_session = service.session_store.get_session(payload["session_id"])
    assert saved_session is not None
    assert saved_session.mounted_knowledge_sources == ("documents", "ecommerce")


def test_create_session_rejects_unknown_mounted_knowledge_source() -> None:
    service = _build_chat_service(
        "chat-api-invalid-mounted-source",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/sessions",
            json={"mounted_knowledge_sources": ["documents", "unknown_source"]},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["detail"]["code"] == "INVALID_MOUNTED_KNOWLEDGE_SOURCES"


def test_session_detail_returns_explicit_mounted_knowledge_sources() -> None:
    service = _build_chat_service(
        "chat-api-session-detail-mounted-sources",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    created = service.create_session(
        scene="generic_assistant",
        mounted_knowledge_sources=["documents", "ecommerce"],
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.get(f"/sessions/{created.session_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mounted_knowledge_sources"] == ["documents", "ecommerce"]


def test_chat_routes_by_session_scene_instead_of_global_default() -> None:
    service = _build_chat_service(
        "chat-api-session-scene-routing",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        create_response = client.post("/sessions", json={"scene": "ecommerce"})
        assert create_response.status_code == 200
        session_id = create_response.json()["session_id"]

        chat_response = client.post(
            "/chat",
            json={"message": "Where is my order?", "session_id": session_id},
        )

    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["scene"] == "ecommerce"
    assert payload["agent"] == "shopping_agent"


def test_chat_only_uses_document_tools_when_session_mounts_documents_only() -> None:
    knowledge = FakeKnowledgeService(
        products=[
            _result(
                doc_id="product-1",
                content="AeroPhone X，库存充足。",
                score=0.95,
                metadata={"product_id": "P005"},
            )
        ]
    )
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="售后 FAQ：库存问题以系统实时状态为准。",
                score=0.91,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "faq.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-faq-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="请以文档说明为准。")
    service = _build_chat_service(
        "chat-api-documents-only-routing",
        knowledge,
        document_retrieval_service,
        model,
    )
    created = service.create_session(scene="generic_assistant", mounted_knowledge_sources=["documents"])
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "AeroPhone X 现在有货吗", "session_id": created.session_id},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is True
    assert payload["citations"][0]["namespace"] == "documents"
    assert payload["citations"][0]["source_kind"] == "document_chunk"
    assert "[1]" in payload["answer"]


def test_chat_can_route_to_ecommerce_tools_when_session_mounts_ecommerce() -> None:
    knowledge = FakeKnowledgeService(
        products=[
            _result(
                doc_id="product-1",
                content="AeroPhone X，库存充足。",
                score=0.95,
                metadata={"product_id": "P005"},
            )
        ]
    )
    model = FakeModel(answer="AeroPhone X 当前有货。")
    service = _build_chat_service(
        "chat-api-ecommerce-routing",
        knowledge,
        FakeDocumentRetrievalService(),
        model,
    )
    created = service.create_session(
        scene="generic_assistant",
        mounted_knowledge_sources=["documents", "ecommerce"],
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "AeroPhone X 现在有货吗", "session_id": created.session_id},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is True
    namespaces = {citation["namespace"] for citation in payload["citations"]}
    assert "documents" not in namespaces
    assert "products" in namespaces or "inventory" in namespaces
    assert "[1]" in payload["answer"]


def test_chat_service_resolves_candidate_tools_from_scene_definition() -> None:
    observed_candidate_tools: list[tuple[str, ...]] = []

    class _TrackingRetriever:
        def retrieve_with_trace(self, query: str, *, candidate_tools: tuple[str, ...], **kwargs: Any):
            del query, kwargs
            observed_candidate_tools.append(candidate_tools)

            class _Outcome:
                documents: list[Document] = []
                exit_reason = "sufficient"
                success = True
                rounds: list[object] = []

            return _Outcome()

    scene_definition = SceneDefinition(
        scene="generic_assistant",
        name="Tracking Scene",
        description="Track candidate tool resolution source.",
        build_retriever=lambda: _TrackingRetriever(),  # type: ignore[return-value]
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda mounted_knowledge_sources: (
            "scene_documents_only",
            *(("scene_ecommerce_tool",) if "ecommerce" in mounted_knowledge_sources else ()),
        ),
        system_prompt="test",
        fallback_policy=SceneFallbackPolicy(no_hit_message="no hit"),
        infer_complexity=lambda _: "simple",
    )
    runtime_dir = make_test_runtime_dir("chat-service-scene-candidate-tools")
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        session={"sqlite_path": sqlite_path, "window_size": 3},
    )
    service = ChatService(
        scene_definition=scene_definition,
        app_settings=app_settings,
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=FakeModel(answer="unused"),
    )
    created = service.session_store.create_session(
        session_id="scene-definition-candidate-tools",
        scene="generic_assistant",
        mounted_knowledge_sources=("documents", "ecommerce"),
    )

    response = service.chat(type("Payload", (), {
        "message": "hello",
        "session_id": created.session_id,
        "stream": False,
    })())

    assert response.knowledge_used is False
    assert observed_candidate_tools == [("scene_documents_only", "scene_ecommerce_tool")]


def test_chat_service_passes_through_scene_defined_custom_candidate_tools() -> None:
    observed_candidate_tools: list[tuple[str, ...]] = []

    class _TrackingRetriever:
        def retrieve_with_trace(self, query: str, *, candidate_tools: tuple[str, ...], **kwargs: Any):
            del query, kwargs
            observed_candidate_tools.append(candidate_tools)

            class _Outcome:
                documents: list[Document] = []
                exit_reason = "sufficient"
                success = True
                rounds: list[object] = []

            return _Outcome()

    scene_definition = SceneDefinition(
        scene="generic_assistant",
        name="Custom Candidate Scene",
        description="Pass through arbitrary candidate tools.",
        build_retriever=lambda: _TrackingRetriever(),  # type: ignore[return-value]
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda mounted_knowledge_sources: (
            "alpha_tool",
            *(("beta_extension_tool",) if "ecommerce" in mounted_knowledge_sources else ()),
        ),
        system_prompt="test",
        fallback_policy=SceneFallbackPolicy(no_hit_message="no hit"),
        infer_complexity=lambda _: "simple",
    )
    runtime_dir = make_test_runtime_dir("chat-service-custom-candidate-tools")
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        session={"sqlite_path": sqlite_path, "window_size": 3},
    )
    service = ChatService(
        scene_definition=scene_definition,
        app_settings=app_settings,
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=FakeModel(answer="unused"),
    )
    created = service.session_store.create_session(
        session_id="custom-candidate-tools",
        scene="generic_assistant",
        mounted_knowledge_sources=("documents", "ecommerce"),
    )

    response = service.chat(type("Payload", (), {
        "message": "hello",
        "session_id": created.session_id,
        "stream": False,
    })())

    assert response.knowledge_used is False
    assert observed_candidate_tools == [("alpha_tool", "beta_extension_tool")]


def test_chat_service_passes_scene_retrieval_policy_to_agentic_retriever() -> None:
    observed_kwargs: list[dict[str, Any]] = []

    class _TrackingRetriever:
        def retrieve_with_trace(
            self,
            query: str,
            *,
            candidate_tools: tuple[str, ...],
            top_k: int | None = None,
            min_relevance_score: float | None = None,
            recall_strategy: str = "hybrid",
            rerank_enabled: bool = False,
            rerank_top_n: int | None = None,
        ):
            del query, candidate_tools
            observed_kwargs.append(
                {
                    "top_k": top_k,
                    "min_relevance_score": min_relevance_score,
                    "recall_strategy": recall_strategy,
                    "rerank_enabled": rerank_enabled,
                    "rerank_top_n": rerank_top_n,
                }
            )

            class _Outcome:
                documents: list[Document] = []
                exit_reason = "ask_user"
                success = False
                rounds: list[object] = []

            return _Outcome()

    scene_definition = SceneDefinition(
        scene="generic_assistant",
        name="Policy Scene",
        description="Track scene retrieval policy propagation.",
        build_retriever=lambda: _TrackingRetriever(),  # type: ignore[return-value]
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda _: ("knowledge_document_search",),
        system_prompt="test",
        fallback_policy=SceneFallbackPolicy(no_hit_message="no hit"),
        infer_complexity=lambda _: "simple",
        retrieval_policy=SceneRetrievalPolicy(
            top_k=2,
            min_relevance_score=0.91,
            recall_strategy="hybrid",
            no_hit_strategy="ask_user",
            rerank_enabled=True,
            rerank_top_n=1,
        ),
    )
    runtime_dir = make_test_runtime_dir("chat-service-scene-policy-propagation")
    sqlite_path = runtime_dir / "chat-sessions.db"
    service = ChatService(
        scene_definition=scene_definition,
        app_settings=AppSettings(
            data_dir=runtime_dir,
            session={"sqlite_path": sqlite_path, "window_size": 3},
        ),
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=FakeModel(answer="unused"),
    )

    response = service.chat(type("Payload", (), {
        "message": "hello",
        "session_id": None,
        "stream": False,
        "top_k": 99,
    })())

    assert response.knowledge_used is False
    assert observed_kwargs == [
        {
            "top_k": 2,
            "min_relevance_score": 0.91,
            "recall_strategy": "hybrid",
            "rerank_enabled": True,
            "rerank_top_n": 1,
        }
    ]


def test_chat_no_hit_strategy_fallback_answer_uses_neutral_message() -> None:
    class _EmptySearchRetriever:
        def search(self, **kwargs: Any) -> list[Document]:
            del kwargs
            return []

    scene_definition = SceneDefinition(
        scene="generic_assistant",
        name="Fallback Answer Scene",
        description="Track no-hit strategy.",
        build_retriever=lambda: _EmptySearchRetriever(),  # type: ignore[return-value]
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda _: ("knowledge_document_search",),
        system_prompt="test",
        fallback_policy=SceneFallbackPolicy(
            no_hit_message="ask user fallback",
            neutral_no_hit_message="neutral no evidence fallback",
        ),
        infer_complexity=lambda _: "simple",
        retrieval_policy=SceneRetrievalPolicy(no_hit_strategy="fallback_answer"),
    )
    runtime_dir = make_test_runtime_dir("chat-no-hit-strategy-fallback-answer")
    sqlite_path = runtime_dir / "chat-sessions.db"
    service = ChatService(
        scene_definition=scene_definition,
        app_settings=AppSettings(
            data_dir=runtime_dir,
            session={"sqlite_path": sqlite_path, "window_size": 3},
        ),
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=FakeModel(answer="unused"),
    )

    response = service.chat(type("Payload", (), {
        "message": "hello",
        "session_id": None,
        "stream": False,
    })())

    assert response.knowledge_used is False
    assert response.citations == []
    assert response.answer == "neutral no evidence fallback"


def test_chat_scene_policy_min_relevance_preserves_no_hit_fallback() -> None:
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-low",
                content="低相关片段不应进入回答。",
                score=0.5,
                metadata={
                    "document_id": "doc-low",
                    "source_path": "low.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-low",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="unused")
    service = _build_chat_service(
        "chat-api-scene-min-relevance-no-hit",
        FakeKnowledgeService(),
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "低相关问题"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert document_retrieval_service.calls[-1]["minimum_relevance"] == 0.8


def test_chat_documents_and_ecommerce_mounts_do_not_force_extension_when_docs_are_sufficient() -> None:
    knowledge = FakeKnowledgeService(
        products=[
            _result(
                doc_id="product-1",
                content="AeroPhone X，库存充足。",
                score=0.95,
                metadata={"product_id": "P005"},
            )
        ]
    )
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="产品手册：AeroPhone X 价格 4599 元，电池 5000mAh。",
                score=0.97,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="根据产品手册，AeroPhone X 售价 4599 元，电池 5000mAh。")
    service = _build_chat_service(
        "chat-api-docs-sufficient-with-ecommerce-mounted",
        knowledge,
        document_retrieval_service,
        model,
    )
    created = service.create_session(
        scene="generic_assistant",
        mounted_knowledge_sources=["documents", "ecommerce"],
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "请根据产品手册说明 AeroPhone X 的价格和电池参数",
                "session_id": created.session_id,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is True
    assert {citation["namespace"] for citation in payload["citations"]} == {"documents"}


def test_session_detail_normalizes_legacy_retrieval_snippets() -> None:
    service = _build_chat_service(
        "chat-api-legacy-retrieval-snippets",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    service.session_store.append_turn(
        session_id="legacy-session",
        request_id="req-legacy",
        user_message="旧问题",
        assistant_answer="旧回答",
        retrieval_snippets=[
            {
                "citation_id": "legacy-doc",
                "namespace": "documents",
                "snippet": "历史文档片段",
                "score": 0.7,
            }
        ],
        timestamp="2026-04-23T00:00:00+00:00",
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.get("/sessions/legacy-session")

    assert response.status_code == 200
    assistant_message = response.json()["messages"][1]
    assert assistant_message["type"] == "ai"
    assert assistant_message["knowledge_used"] is True
    assert assistant_message["citations"] == [
        {
            "index": 1,
            "citation_id": "legacy-doc",
            "namespace": "documents",
            "source_kind": "documents",
            "source_name": "legacy-doc",
            "source_path": None,
            "document_id": None,
            "chunk_id": "legacy-doc",
            "chunk_index": None,
            "snippet": "历史文档片段",
            "score": 0.7,
            "vector_score": None,
            "keyword_score": None,
            "vector_rank": None,
            "keyword_rank": None,
            "matched_by": [],
            "rank": 1,
        }
    ]


def test_session_detail_returns_message_view_with_assistant_metadata() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="AeroPhone X 当前有货，售价 4599 元。",
                score=0.93,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="AeroPhone X 当前有货，售价 4599 元。[1]")
    service = _build_chat_service(
        "chat-api-session-message-view",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        chat_response = client.post("/chat", json={"message": "AeroPhone X 有货吗"})
        assert chat_response.status_code == 200
        chat_payload = chat_response.json()

        session_response = client.get(f"/sessions/{chat_payload['session_id']}")

    assert session_response.status_code == 200
    payload = session_response.json()
    assert payload["total_messages"] == 2
    assert [message["type"] for message in payload["messages"]] == ["human", "ai"]
    assistant_message = payload["messages"][1]
    assert assistant_message["request_id"] == chat_payload["request_id"]
    assert assistant_message["timestamp"]
    assert assistant_message["knowledge_used"] is True
    assert assistant_message["citations"][0]["citation_id"] == "chunk-manual-1"


def test_chat_api_real_runtime_filters_low_relevance_document_hits_for_greeting() -> None:
    runtime_dir = make_test_runtime_dir("chat-api-real-runtime-low-relevance")
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        app={
            "active_scene": "generic_assistant",
        },
        session={
            "sqlite_path": sqlite_path,
            "window_size": 3,
        },
        vector_store={
            "provider": "chroma",
            "chroma": {"persist_directory": runtime_dir / ".chroma"},
        },
    )
    store = VectorStoreFactory.create(app_settings)
    store.ensure_document_indexes()
    store.upsert_document_chunks(
        [
            VectorStoreDocument(
                id="chunk-order-1",
                content=(
                    '{"carrier":"申通快递","status":"已签收","tracking_no":"ST0011223344CN",'
                    '"shipping_address":"重庆市渝中区解放碑步行街9号"}'
                ),
                metadata={
                    "document_id": "doc-order-1",
                    "source_path": "orders.json",
                    "namespace": "orders",
                    "chunk_id": "chunk-order-1",
                    "chunk_index": 0,
                    "is_active": True,
                },
            )
        ]
    )
    knowledge_service = create_knowledge_service(app_settings=app_settings, store=store)
    model = FakeModel(answer="unused")
    service = SceneChatService(
        scene_registry=build_default_scene_registry(
            app_settings=app_settings,
            knowledge_service=knowledge_service,
            document_retrieval_service=_build_document_retrieval_service(app_settings),
        ),
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        create_response = client.post(
            "/sessions",
            json={
                "scene": "generic_assistant",
                "mounted_knowledge_sources": ["documents"],
            },
        )
        assert create_response.status_code == 200
        response = client.post(
            "/chat",
            json={
                "message": "你好",
                "session_id": create_response.json()["session_id"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert "暂时没有检索到足够相关的文档知识" in payload["answer"]
    assert model.get_runnable_calls == []


def test_chat_api_real_runtime_greeting_does_not_rewrite_into_faq_hit() -> None:
    runtime_dir = make_test_runtime_dir("chat-api-real-runtime-greeting-faq-no-rewrite")
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        app={
            "active_scene": "generic_assistant",
        },
        session={
            "sqlite_path": sqlite_path,
            "window_size": 3,
        },
        vector_store={
            "provider": "chroma",
            "chroma": {"persist_directory": runtime_dir / ".chroma"},
        },
    )
    store = VectorStoreFactory.create(app_settings)
    store.ensure_document_indexes()
    store.upsert_document_chunks(
        [
            VectorStoreDocument(
                id="chunk-faq-1",
                content="Support FAQ：如需查询已上传知识库，请提供具体文档主题、术语或章节名称。",
                metadata={
                    "document_id": "doc-faq-1",
                    "source_path": "support-faq.md",
                    "namespace": "documents",
                    "chunk_id": "chunk-faq-1",
                    "chunk_index": 0,
                    "is_active": True,
                    "is_managed_document": True,
                },
            )
        ]
    )
    knowledge_service = create_knowledge_service(app_settings=app_settings, store=store)
    model = FakeModel(answer="unused")
    service = SceneChatService(
        scene_registry=build_default_scene_registry(
            app_settings=app_settings,
            knowledge_service=knowledge_service,
            document_retrieval_service=_build_document_retrieval_service(app_settings),
        ),
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        create_response = client.post(
            "/sessions",
            json={
                "scene": "generic_assistant",
                "mounted_knowledge_sources": ["documents"],
            },
        )
        assert create_response.status_code == 200
        response = client.post(
            "/chat",
            json={
                "message": "你好",
                "session_id": create_response.json()["session_id"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert payload["retrieval_trace"]["knowledge_used"] is False
    assert payload["retrieval_trace"]["filtered_candidates_count"] == 0
    assert payload["retrieval_trace"]["top_k_chunks"] == []
    assert payload["retrieval_trace"]["citations"] == []
    assert "暂时没有检索到足够相关的文档知识" in payload["answer"]
    assert model.get_runnable_calls == []


def test_chat_api_ignores_builtin_orders_json_in_documents_only_session() -> None:
    runtime_dir = make_test_runtime_dir("chat-api-ignore-builtin-orders")
    sqlite_path = runtime_dir / "chat-sessions.db"
    (runtime_dir / "files").mkdir(parents=True, exist_ok=True)
    app_settings = AppSettings(
        data_dir=runtime_dir,
        app={
            "active_scene": "generic_assistant",
        },
        session={
            "sqlite_path": sqlite_path,
            "window_size": 3,
        },
        vector_store={
            "provider": "chroma",
            "chroma": {"persist_directory": runtime_dir / ".chroma"},
        },
    )
    store = VectorStoreFactory.create(app_settings)
    store.ensure_document_indexes()
    store.upsert_document_chunks(
        [
            VectorStoreDocument(
                id="chunk-order-1",
                content='{"carrier":"EMS","status":"已签收","tracking_no":"EMS001"}',
                metadata={
                    "document_id": "doc-order-1",
                    "source_path": "orders.json",
                    "namespace": "orders",
                    "chunk_id": "chunk-order-1",
                    "chunk_index": 0,
                    "is_active": True,
                },
            )
        ]
    )
    knowledge_service = create_knowledge_service(app_settings=app_settings, store=store)
    model = FakeModel(answer="unused")
    service = SceneChatService(
        scene_registry=build_default_scene_registry(
            app_settings=app_settings,
            knowledge_service=knowledge_service,
            document_retrieval_service=_build_document_retrieval_service(app_settings),
        ),
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        create_response = client.post(
            "/sessions",
            json={
                "scene": "generic_assistant",
                "mounted_knowledge_sources": ["documents"],
            },
        )
        response = client.post(
            "/chat",
            json={
                "message": "你好",
                "session_id": create_response.json()["session_id"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert model.get_runnable_calls == []

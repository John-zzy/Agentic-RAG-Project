import json
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from backend.application.runtime.api.app import create_app
from backend.application.runtime import ChatService, SceneChatService, build_default_scene_registry
from backend.platform.config.settings import AppSettings
from backend.platform.knowledge.repositories import VectorStoreFactory
from backend.platform.memory.base.session_store import SQLiteSessionStore
from backend.platform.memory.chat.prompt_context import PromptContextBuilder
from backend.platform.rag.document_retrieval import DocumentChunkRetrievalResult
from backend.platform.rag.document_retrieval_service import DocumentRetrievalService
from backend.platform.retrieval import VectorSearchResult, VectorStoreDocument
from backend.scenes.base import SceneDefinition, SceneFallbackPolicy
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

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
    ) -> list[DocumentChunkRetrievalResult]:
        del query, namespace
        return [
            DocumentChunkRetrievalResult(
                document=result.document,
                score=result.score,
                vector_score=result.score,
                vector_rank=index,
                matched_by=["vector"],
            )
            for index, result in enumerate(self._documents[:top_k], start=1)
        ]


class FakeModel:
    def __init__(self, answer: str = "mock-answer", stream_chunks: list[str] | None = None) -> None:
        self.answer = answer
        self.chat_model_calls: list[str] = []
        self.stream_template_calls: list[str] = []
        self.stream_chunks = stream_chunks or [answer]

    def build_chat_model_for_complexity(self, complexity: str):
        self.chat_model_calls.append(complexity)
        return RunnableLambda(lambda _: self.answer)

    def invoke_template(self, prompt_template: Any, variables: dict[str, Any], complexity: str = "simple") -> str:
        del prompt_template, variables
        self.chat_model_calls.append(complexity)
        return self.answer

    def stream_template(
        self,
        prompt_template: Any,
        variables: dict[str, Any],
        complexity: str = "simple",
    ):
        del prompt_template, variables
        self.stream_template_calls.append(complexity)
        for chunk in self.stream_chunks:
            yield chunk


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
    saved_session = service.session_store.get_session(payload["session_id"])
    assert saved_session is not None
    assert saved_session.mounted_knowledge_sources == ("documents",)
    assert model.chat_model_calls


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
    assert [event["event"] for event in events] == ["start", "chunk", "chunk", "done"]

    start_data = events[0]["data"]
    done_data = events[-1]["data"]
    assert start_data
    assert done_data

    start_payload = json.loads(start_data)
    done_payload = json.loads(done_data)
    assert start_payload["session_id"]
    assert start_payload["request_id"]
    assert start_payload["knowledge_used"] is True
    assert done_payload["answer"] == "推荐 P001，续航表现较好。[1]"
    assert done_payload["knowledge_used"] is True
    assert done_payload["scene"] == "generic_assistant"
    assert len(done_payload["citations"]) == 1
    saved_turns, total_turns = service.session_store.get_session_detail(
        done_payload["session_id"],
        limit=10,
    )
    assert total_turns == 1
    assert saved_turns[0].assistant_answer == "推荐 P001，续航表现较好。[1]"
    assert model.stream_template_calls == ["simple"]


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
    assert model.chat_model_calls == []


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
    assert [event["event"] for event in events] == ["start", "chunk", "done"]

    chunk_payload = json.loads(events[1]["data"])
    done_payload = json.loads(events[2]["data"])
    assert "暂时没有检索到足够相关的文档知识" in chunk_payload["delta"]
    assert done_payload["knowledge_used"] is False
    assert done_payload["citations"] == []
    assert model.stream_template_calls == []
    assert model.chat_model_calls == []


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
        assert empty_session_response.json()["total_turns"] == 0
        assert empty_session_response.json()["turns"] == []

        chat_response = client.post("/chat", json={"message": "你好", "session_id": session_id})
        assert chat_response.status_code == 200

        populated_session_response = client.get(f"/sessions/{session_id}")
        assert populated_session_response.status_code == 200
        payload = populated_session_response.json()
        assert payload["session_id"] == session_id
        assert payload["mounted_knowledge_sources"] == ["documents"]
        assert payload["total_turns"] == 1
        assert len(payload["turns"]) == 1
        assert payload["turns"][0]["user_message"] == "你好"

        delete_response = client.delete(f"/sessions/{session_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["deleted_turns"] == 1

        after_delete_response = client.get(f"/sessions/{session_id}")
        assert after_delete_response.status_code == 200
        assert after_delete_response.json()["total_turns"] == 0


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
    turn = response.json()["turns"][0]
    assert turn["retrieval_snippets"] == [
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
    assert model.chat_model_calls == []


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
    assert model.chat_model_calls == []

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from backend.api.app import create_app
from backend.api.chat_service import ChatService
from backend.config.settings import AppSettings
from backend.knowledge.store import VectorSearchResult, VectorStoreDocument
from backend.memory.prompt_context import PromptContextBuilder
from backend.memory.session_store import SQLiteSessionStore
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


class FakeKnowledgeService:
    def __init__(
        self,
        products: list[VectorSearchResult] | None = None,
        reviews: list[VectorSearchResult] | None = None,
    ) -> None:
        self._products = products or []
        self._reviews = reviews or []

    def search_products(self, query: str, top_k: int | None = None) -> list[VectorSearchResult]:
        return self._products

    def search_reviews(self, query: str, top_k: int | None = None) -> list[VectorSearchResult]:
        return self._reviews


class FakeModel:
    def __init__(self, answer: str = "mock-answer") -> None:
        self.answer = answer
        self.chat_model_calls: list[str] = []

    def build_chat_model_for_complexity(self, complexity: str):
        self.chat_model_calls.append(complexity)
        return RunnableLambda(lambda _: self.answer)


def _build_chat_service(
    test_name: str,
    knowledge_service: FakeKnowledgeService,
    model: FakeModel,
) -> ChatService:
    runtime_dir = make_test_runtime_dir(test_name)
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        session={
            "sqlite_path": sqlite_path,
            "window_size": 3,
        }
    )
    return ChatService(
        app_settings=app_settings,
        knowledge_service=knowledge_service,  # type: ignore[arg-type]
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=model,
    )


def test_chat_api_success_path() -> None:
    knowledge = FakeKnowledgeService(
        products=[
            _result(
                doc_id="P001",
                content="P001 手机，续航强，电池 5000mAh。",
                score=0.92,
                metadata={"product_id": "P001"},
            )
        ]
    )
    model = FakeModel(answer="推荐 P001，续航表现较好。")
    service = _build_chat_service("chat-api-success", knowledge, model)
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "推荐续航好的手机"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["session_id"]
    assert payload["request_id"]
    assert payload["answer"] == "推荐 P001，续航表现较好。"
    assert payload["knowledge_used"] is True
    assert len(payload["citations"]) == 1
    assert model.chat_model_calls


def test_chat_api_validation_error_when_message_missing() -> None:
    service = _build_chat_service("chat-api-validation-error", FakeKnowledgeService(), FakeModel())
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_api_no_hit_fallback_sets_knowledge_used_false() -> None:
    model = FakeModel(answer="unused")
    service = _build_chat_service("chat-api-no-hit", FakeKnowledgeService(), model)
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "火星基地快递多久到"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert "暂时没有检索到足够相关的商品知识" in payload["answer"]
    assert model.chat_model_calls == []

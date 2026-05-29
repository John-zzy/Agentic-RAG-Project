from __future__ import annotations

import json
import importlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.documents import Document

settings_module = importlib.import_module("backend.platform.config.settings")
from backend.platform.config.settings import (
    build_model_routing_settings,
    load_model_routing_config,
)
from backend.platform.models.base.router import (
    ModelRouter,
    RoutedEmbeddingModel,
    RoutedRerankModel,
)
from backend.platform.models.llm.embedding import EmbeddingStrategyFactory
from backend.platform.models.llm.rerank import RerankWrapperFactory
from backend.platform.rag.contracts import RetrievalResult
from backend.platform.rag.post_retrieval import DashScopeRetrievalReranker


def test_model_routing_config_loads_embedding_and_rerank_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routing_file = Path(tempfile.mkdtemp()) / "model_routing.json"
    routing_file.write_text(
        json.dumps(
            {
                "models": {
                    "simple": {
                        "provider": "dashscope",
                        "model_name": "qwen-turbo",
                        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": "AI_RAG_MODELS__SIMPLE__API_KEY",
                    },
                    "moderate": {
                        "provider": "dashscope",
                        "model_name": "qwen-plus",
                        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": "AI_RAG_MODELS__MODERATE__API_KEY",
                    },
                    "complex": {
                        "provider": "dashscope",
                        "model_name": "qwen-max",
                        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": "AI_RAG_MODELS__COMPLEX__API_KEY",
                    },
                    "embedding": {
                        "provider": "dashscope",
                        "model_name": "text-embedding-v4",
                        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": "AI_RAG_MODELS__EMBEDDING__API_KEY",
                        "dimensions": 256,
                    },
                    "rerank": {
                        "provider": "dashscope",
                        "model_name": "gte-rerank-v2",
                        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": "AI_RAG_MODELS__RERANK__API_KEY",
                        "top_n": 3,
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_module, "MODEL_ROUTING_FILE", routing_file)
    monkeypatch.setenv("AI_RAG_MODELS__EMBEDDING__API_KEY", "embedding-key")
    monkeypatch.setenv("AI_RAG_MODELS__RERANK__API_KEY", "rerank-key")

    config = build_model_routing_settings()

    assert config.embedding.provider == "dashscope"
    assert config.embedding.model_name == "text-embedding-v4"
    assert config.embedding.dimensions == 256
    assert config.embedding.api_key == "embedding-key"
    assert config.embedding.api_key_env == "AI_RAG_MODELS__EMBEDDING__API_KEY"
    assert config.rerank.provider == "dashscope"
    assert config.rerank.model_name == "gte-rerank-v2"
    assert config.rerank.top_n == 3
    assert config.rerank.api_key == "rerank-key"
    assert config.rerank.api_key_env == "AI_RAG_MODELS__RERANK__API_KEY"


def test_model_routing_config_reads_keys_from_configured_env_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routing_file = Path(tempfile.mkdtemp()) / "model_routing.json"
    routing_file.write_text(
        json.dumps(
            {
                "models": {
                    "simple": {
                        "provider": "dashscope",
                        "model_name": "qwen-turbo",
                        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": "CUSTOM_SIMPLE_KEY",
                    },
                    "moderate": {
                        "provider": "dashscope",
                        "model_name": "qwen-plus",
                        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": "CUSTOM_MODERATE_KEY",
                    },
                    "complex": {
                        "provider": "dashscope",
                        "model_name": "qwen-max",
                        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": "CUSTOM_COMPLEX_KEY",
                    },
                    "embedding": {
                        "provider": "dashscope",
                        "model_name": "text-embedding-v4",
                        "api_key_env": "CUSTOM_EMBEDDING_KEY",
                        "dimensions": 256,
                    },
                    "rerank": {
                        "provider": "dashscope",
                        "model_name": "gte-rerank-v2",
                        "api_key_env": "CUSTOM_RERANK_KEY",
                        "top_n": 3,
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_module, "MODEL_ROUTING_FILE", routing_file)
    monkeypatch.setenv("CUSTOM_SIMPLE_KEY", "simple-key")
    monkeypatch.setenv("CUSTOM_MODERATE_KEY", "moderate-key")
    monkeypatch.setenv("CUSTOM_COMPLEX_KEY", "complex-key")
    monkeypatch.setenv("CUSTOM_EMBEDDING_KEY", "embedding-key")
    monkeypatch.setenv("CUSTOM_RERANK_KEY", "rerank-key")

    config = build_model_routing_settings()

    assert config.simple.api_key_env == "CUSTOM_SIMPLE_KEY"
    assert config.simple.api_key == "simple-key"
    assert config.moderate.api_key == "moderate-key"
    assert config.complex.api_key == "complex-key"
    assert config.embedding.api_key_env == "CUSTOM_EMBEDDING_KEY"
    assert config.embedding.api_key == "embedding-key"
    assert config.rerank.api_key_env == "CUSTOM_RERANK_KEY"
    assert config.rerank.api_key == "rerank-key"


def test_model_router_exposes_embedding_and_rerank_models() -> None:
    router = ModelRouter()

    embedding_model = router.select_embedding()
    rerank_model = router.select_rerank()

    assert isinstance(embedding_model, RoutedEmbeddingModel)
    assert isinstance(rerank_model, RoutedRerankModel)


def test_embedding_factory_passes_provider_model_dimensions_and_api_key() -> None:
    observed: dict[str, Any] = {}

    class _FakeClient:
        def call(self, **kwargs: Any) -> Any:
            observed.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                code="",
                message="",
                output={"embeddings": [{"embedding": [0.1, 0.2, 0.3]}]},
            )

    factory = EmbeddingStrategyFactory(client_factory=lambda: _FakeClient())
    strategy = factory.create(
        RoutedEmbeddingModel(
            provider="dashscope",
            model_name="text-embedding-v4",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="AI_RAG_MODELS__EMBEDDING__API_KEY",
            api_key="embedding-key",
            dimensions=3,
            timeout_seconds=30,
            max_retries=3,
        )
    )

    embedding = strategy.embed("你好")

    assert embedding == [0.1, 0.2, 0.3]
    assert observed["model"] == "text-embedding-v4"
    assert observed["dimension"] == 3
    assert "dimensions" not in observed
    assert observed["api_key"] == "embedding-key"
    assert observed["text_type"] == "query"


def test_embedding_factory_raises_when_api_key_missing() -> None:
    factory = EmbeddingStrategyFactory(client_factory=lambda: object())

    with pytest.raises(ValueError, match="Missing API key for embedding model"):
        factory.create(
            RoutedEmbeddingModel(
                provider="dashscope",
                model_name="text-embedding-v4",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key_env="AI_RAG_MODELS__EMBEDDING__API_KEY",
                api_key=None,
                dimensions=256,
                timeout_seconds=30,
                max_retries=3,
            )
        )


def test_rerank_factory_passes_provider_model_top_n_and_api_key() -> None:
    observed: dict[str, Any] = {}

    class _FakeWrapper:
        def __init__(self, **kwargs: Any) -> None:
            observed.update(kwargs)

    factory = RerankWrapperFactory(
        client_factory=lambda: "fake-client",
        wrapper_factory=_FakeWrapper,
    )
    wrapper = factory.create(
        RoutedRerankModel(
            provider="dashscope",
            model_name="gte-rerank-v2",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="AI_RAG_MODELS__RERANK__API_KEY",
            api_key="rerank-key",
            top_n=7,
            timeout_seconds=30,
        )
    )

    assert isinstance(wrapper, _FakeWrapper)
    assert observed["client"] == "fake-client"
    assert observed["model"] == "gte-rerank-v2"
    assert observed["top_n"] == 7
    assert observed["api_key"] == "rerank-key"


def test_rerank_factory_raises_when_api_key_missing() -> None:
    factory = RerankWrapperFactory(client_factory=lambda: "fake-client", wrapper_factory=lambda **kwargs: kwargs)

    with pytest.raises(ValueError, match="Missing API key for rerank model"):
        factory.create(
            RoutedRerankModel(
                provider="dashscope",
                model_name="gte-rerank-v2",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key_env="AI_RAG_MODELS__RERANK__API_KEY",
                api_key=None,
                top_n=3,
                timeout_seconds=30,
            )
        )


def test_dashscope_reranker_uses_wrapper_default_top_n_when_policy_top_n_missing() -> None:
    observed_calls: list[dict[str, Any]] = []

    class _FakeWrapper:
        def rerank(self, documents: list[Document], query: str, **kwargs: Any) -> list[dict[str, object]]:
            del documents, query
            observed_calls.append(kwargs)
            return [{"index": 0, "relevance_score": 0.9}]

    result = RetrievalResult.ok(
        tool_name="test_tool",
        query="query",
        documents=[Document(page_content="candidate", metadata={"citation_id": "c1"})],
    )
    reranker = DashScopeRetrievalReranker(wrapper_factory=lambda: _FakeWrapper())

    reranker.rerank(query="query", result=result, top_n=None)
    reranker.rerank(query="query", result=result, top_n=1)

    assert observed_calls == [{}, {"top_n": 1}]


def test_model_routing_config_loads_from_json_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_RAG_MODELS__EMBEDDING__API_KEY", raising=False)
    monkeypatch.delenv("AI_RAG_MODELS__RERANK__API_KEY", raising=False)

    config = load_model_routing_config()

    assert "embedding" in config["models"]
    assert "rerank" in config["models"]


def test_model_routing_config_raises_when_json_file_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_file = Path(tempfile.mkdtemp()) / "missing-model-routing.json"
    monkeypatch.setattr(settings_module, "MODEL_ROUTING_FILE", missing_file)

    with pytest.raises(FileNotFoundError, match="Model routing config file is required"):
        load_model_routing_config()

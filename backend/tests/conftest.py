from __future__ import annotations

import pytest

import backend.platform.knowledge.base.store as store_module
import backend.platform.rag.retrieval.documents.embedding as document_embedding_module
from backend.tests.test_support import FakeEmbeddingStrategy


@pytest.fixture(autouse=True)
def use_fake_embedding_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """单元测试默认使用 fake embedding，避免访问真实外部模型服务。"""
    monkeypatch.setattr(store_module, "get_embedding_strategy", lambda: FakeEmbeddingStrategy())
    monkeypatch.setattr(document_embedding_module, "get_embedding_strategy", lambda: FakeEmbeddingStrategy())

from __future__ import annotations

from backend.platform.models.llm import get_embedding_strategy
from backend.platform.search_foundation import EmbeddingStrategy


class DocumentEmbeddingStrategy:
    """文档检索 embedding 策略适配器。"""

    def __init__(self, embedding_strategy: EmbeddingStrategy | None = None) -> None:
        self._embedding_strategy = embedding_strategy or get_embedding_strategy()

    @property
    def dimensions(self) -> int:
        return self._embedding_strategy.dimensions

    def embed(self, text: str) -> list[float]:
        return self._embedding_strategy.embed(text)

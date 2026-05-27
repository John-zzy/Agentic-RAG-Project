from __future__ import annotations

from backend.platform.search_foundation import LocalHashingEmbedder


class DocumentEmbeddingStrategy:
    """文档检索默认 embedding 策略。"""

    def __init__(self, embedder: LocalHashingEmbedder | None = None) -> None:
        self._embedder = embedder or LocalHashingEmbedder()

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed(text)

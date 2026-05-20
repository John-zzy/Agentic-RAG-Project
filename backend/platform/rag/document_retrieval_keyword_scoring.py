from __future__ import annotations

from langchain_community.retrievers import BM25Retriever

from backend.platform.knowledge.base.store import LocalHashingEmbedder


class DocumentKeywordScoreCalculator:
    """负责从 BM25 结果中计算并归一化关键词得分。"""

    def __init__(self, *, embedder: LocalHashingEmbedder | None = None) -> None:
        self._embedder = embedder or LocalHashingEmbedder()

    def score_documents(self, retriever: BM25Retriever, query: str) -> dict[str, float]:
        query_tokens = retriever.preprocess_func(query)
        if not query_tokens:
            return {}
        raw_scores = retriever.vectorizer.get_scores(query_tokens)
        normalized_scores = self._normalize_scores(list(raw_scores))
        if not any(score > 0 for score in normalized_scores):
            normalized_scores = self._fallback_overlap_scores(retriever, query)

        scored_documents: dict[str, float] = {}
        for document, score in zip(retriever.docs, normalized_scores, strict=False):
            chunk_id = str(document.metadata.get("_chunk_id") or document.metadata.get("chunk_id") or "")
            if chunk_id and score > 0:
                scored_documents[chunk_id] = score
        return scored_documents

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        positive_scores = [float(score) for score in scores if float(score) > 0]
        if not positive_scores:
            return [0.0 for _ in scores]
        max_score = max(positive_scores)
        if max_score <= 0:
            return [0.0 for _ in scores]
        return [max(float(score), 0.0) / max_score for score in scores]

    def _fallback_overlap_scores(self, retriever: BM25Retriever, query: str) -> list[float]:
        query_terms = set(self._tokenize(query))
        if not query_terms:
            return [0.0 for _ in retriever.docs]

        overlap_scores: list[float] = []
        for document in retriever.docs:
            document_terms = set(self._tokenize(document.page_content))
            if not document_terms:
                overlap_scores.append(0.0)
                continue
            overlap_scores.append(len(query_terms & document_terms) / len(query_terms))
        return overlap_scores

    def _tokenize(self, text: str) -> list[str]:
        return self._embedder._tokenize(text.strip().lower())

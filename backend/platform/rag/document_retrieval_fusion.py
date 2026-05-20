from __future__ import annotations

from backend.platform.rag.document_retrieval_types import (
    DocumentChunkRetrievalResult,
    merge_matched_by,
)


class HybridFusionRanker:
    """融合语义召回与关键词召回。"""

    def __init__(
        self,
        *,
        vector_weight: float = 0.65,
        keyword_weight: float = 0.35,
    ) -> None:
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def rank(
        self,
        *,
        vector_results: list[DocumentChunkRetrievalResult],
        keyword_results: list[DocumentChunkRetrievalResult],
        top_k: int,
    ) -> list[DocumentChunkRetrievalResult]:
        merged: dict[str, DocumentChunkRetrievalResult] = {}
        fused_scores: dict[str, float] = {}
        normalized_vector_scores = self._normalize_scores(
            {result.document.id: result.vector_score for result in vector_results}
        )
        normalized_keyword_scores = self._normalize_scores(
            {result.document.id: result.keyword_score for result in keyword_results}
        )

        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.document.id
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + self.vector_weight * normalized_vector_scores.get(chunk_id, 0.0)
            )
            merged[chunk_id] = result.model_copy(
                update={
                    "vector_rank": rank,
                    "vector_score": normalized_vector_scores.get(chunk_id, result.vector_score),
                }
            )

        for rank, result in enumerate(keyword_results, start=1):
            chunk_id = result.document.id
            keyword_score = normalized_keyword_scores.get(chunk_id, 0.0)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + self.keyword_weight * keyword_score
            existing = merged.get(chunk_id)
            if existing is None:
                merged[chunk_id] = result.model_copy(
                    update={
                        "keyword_score": keyword_score,
                        "keyword_rank": rank,
                    }
                )
                continue
            merged[chunk_id] = existing.model_copy(
                update={
                    "keyword_score": keyword_score,
                    "keyword_rank": rank,
                    "matched_by": merge_matched_by(existing.matched_by, ["keyword"]),
                }
            )

        ranked_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)
        ranked_results: list[DocumentChunkRetrievalResult] = []
        for chunk_id in ranked_ids[:top_k]:
            result = merged[chunk_id]
            ranked_results.append(
                result.model_copy(
                    update={
                        "score": fused_scores[chunk_id],
                        "matched_by": result.matched_by or ["vector"],
                    }
                )
            )
        return ranked_results

    def _normalize_scores(self, scores: dict[str, float | None]) -> dict[str, float]:
        positive_scores = {key: float(value) for key, value in scores.items() if value is not None and float(value) > 0}
        if not positive_scores:
            return {key: 0.0 for key in scores}
        max_score = max(positive_scores.values())
        if max_score <= 0:
            return {key: 0.0 for key in scores}
        return {
            key: (float(value) / max_score) if value is not None and float(value) > 0 else 0.0
            for key, value in scores.items()
        }

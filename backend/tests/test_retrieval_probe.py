from __future__ import annotations

import json
from typing import Any

from backend.evals.retrieval_probe import SAFE_RANKED_ITEM_FIELDS, run_retrieval_probe
from backend.platform.rag.retrieval.documents import DocumentChunkRetrievalResult
from backend.platform.search_foundation import VectorSearchResult, VectorStoreDocument


class FakeDocumentRetrievalService:
    def __init__(self, results: list[VectorSearchResult]) -> None:
        self._results = results
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        minimum_relevance: float | None = None,
        recall_strategy: str = "hybrid",
    ) -> list[DocumentChunkRetrievalResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "namespace": namespace,
                "minimum_relevance": minimum_relevance,
                "recall_strategy": recall_strategy,
            }
        )
        return [
            DocumentChunkRetrievalResult(
                document=result.document,
                score=result.score,
                vector_score=result.score,
                vector_rank=index,
                matched_by=["vector"],
            )
            for index, result in enumerate(self._results[:top_k], start=1)
        ]


def _result() -> VectorSearchResult:
    return VectorSearchResult(
        document=VectorStoreDocument(
            id="chunk-1",
            content="SECRET BODY TEXT must never be serialized",
            metadata={
                "document_id": "doc-1",
                "source_path": "eval-benchmark-quickstart.md",
                "namespace": "documents",
                "chunk_id": "chunk-1",
                "chunk_index": 0,
                "snippet": "SECRET SNIPPET",
                "prompt": "SECRET PROMPT",
                "reason": "SECRET REASON",
                "rewrite_reason": "SECRET REWRITE REASON",
            },
        ),
        score=0.91,
    )


def test_retrieval_probe_returns_only_safe_ranked_fields() -> None:
    service = FakeDocumentRetrievalService([_result()])

    payload = run_retrieval_probe(
        samples=[{"sample_id": "sample-1", "query": "Python version docs"}],
        document_retrieval_service=service,  # type: ignore[arg-type]
    )

    sample = payload["samples"][0]
    ranked_item = sample["ranked_list"][0]
    assert list(ranked_item) == list(SAFE_RANKED_ITEM_FIELDS)
    assert ranked_item == {
        "rank": 1,
        "source_doc": "eval-benchmark-quickstart.md",
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "chunk_index": 0,
        "score": 0.91,
        "matched_by": ["vector"],
    }
    assert service.calls[-1]["top_k"] == 5
    assert service.calls[-1]["namespace"] == "documents"
    assert service.calls[-1]["minimum_relevance"] == 0.8
    assert service.calls[-1]["recall_strategy"] == "hybrid"


def test_retrieval_probe_no_hit_returns_empty_ranked_list() -> None:
    payload = run_retrieval_probe(
        samples=[{"sample_id": "no-hit", "query": "unrelated stock count"}],
        document_retrieval_service=FakeDocumentRetrievalService([]),  # type: ignore[arg-type]
    )

    assert payload["samples"][0]["status"] == "ok"
    assert payload["samples"][0]["ranked_list"] == []


def test_retrieval_probe_serialization_does_not_leak_text_fields() -> None:
    payload = run_retrieval_probe(
        samples=[{"sample_id": "sample-1", "query": "Python version docs"}],
        document_retrieval_service=FakeDocumentRetrievalService([_result()]),  # type: ignore[arg-type]
    )

    serialized = json.dumps(payload, ensure_ascii=False)
    for forbidden in (
        '"snippet"',
        '"content"',
        '"prompt"',
        '"reason"',
        '"rewrite_reason"',
        "SECRET BODY TEXT",
        "SECRET SNIPPET",
        "SECRET PROMPT",
        "SECRET REASON",
        "SECRET REWRITE REASON",
    ):
        assert forbidden not in serialized


def test_retrieval_probe_filters_to_allowed_source_docs() -> None:
    other_result = VectorSearchResult(
        document=VectorStoreDocument(
            id="chunk-other",
            content="Other document text",
            metadata={
                "document_id": "doc-other",
                "source_path": "other.md",
                "namespace": "documents",
                "chunk_id": "chunk-other",
                "chunk_index": 0,
            },
        ),
        score=0.99,
    )

    payload = run_retrieval_probe(
        samples=[{"sample_id": "sample-1", "query": "Python version docs"}],
        allowed_source_docs=["eval-benchmark-quickstart.md"],
        document_retrieval_service=FakeDocumentRetrievalService([other_result, _result()]),  # type: ignore[arg-type]
    )

    ranked_list = payload["samples"][0]["ranked_list"]
    assert [item["source_doc"] for item in ranked_list] == ["eval-benchmark-quickstart.md"]
    assert ranked_list[0]["rank"] == 1
    assert payload["allowed_source_docs"] == ["eval-benchmark-quickstart.md"]


def test_retrieval_probe_constrains_to_requested_namespace() -> None:
    service = FakeDocumentRetrievalService([_result()])

    payload = run_retrieval_probe(
        samples=[{"sample_id": "sample-1", "query": "Python version docs"}],
        namespace="faq",
        document_retrieval_service=service,  # type: ignore[arg-type]
    )

    assert payload["namespace"] == "faq"
    assert service.calls[-1]["namespace"] == "faq"

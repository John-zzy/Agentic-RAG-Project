from __future__ import annotations

from math import log2

import pytest

from backend.evals.retrieval_metrics import (
    compute_retrieval_benchmark_metrics,
    compute_retrieval_metrics,
    dedupe_ranked_chunks,
    qrels_list_to_mapping,
)


def test_chunk_metrics_handle_multiple_relevant_chunks() -> None:
    qrels = {
        "sample_id": "multi_chunk",
        "documents": [
            {"source_doc": "doc-a.md", "relevance": 2},
            {"source_doc": "doc-b.md", "relevance": 1},
        ],
        "chunks": [
            {"source_doc": "doc-a.md", "chunk_index": 0, "relevance": 2},
            {"source_doc": "doc-b.md", "chunk_index": 1, "relevance": 1},
        ],
    }
    ranked = [
        {"source_doc": "doc-x.md", "chunk_index": 0},
        {"source_doc": "doc-b.md", "chunk_index": 1},
        {"source_doc": "doc-a.md", "chunk_index": 0},
    ]

    metrics = compute_retrieval_metrics(qrels=qrels, ranked_list=ranked)

    assert metrics["precision_at_k"]["1"] == 0.0
    assert metrics["precision_at_k"]["3"] == pytest.approx(2 / 3)
    assert metrics["recall_at_k"]["1"] == 0.0
    assert metrics["recall_at_k"]["3"] == 1.0
    assert metrics["mrr"] == pytest.approx(1 / 2)

    dcg_at_3 = (2**1 - 1) / log2(3) + (2**2 - 1) / log2(4)
    ideal_dcg_at_3 = (2**2 - 1) / log2(2) + (2**1 - 1) / log2(3)
    assert metrics["ndcg_at_k"]["3"] == pytest.approx(dcg_at_3 / ideal_dcg_at_3)


def test_empty_ranked_list_returns_zero_for_hit_sample() -> None:
    qrels = {
        "sample_id": "empty_ranked",
        "documents": [{"source_doc": "doc-a.md", "relevance": 2}],
        "chunks": [{"source_doc": "doc-a.md", "chunk_index": 0, "relevance": 2}],
    }

    metrics = compute_retrieval_metrics(qrels=qrels, ranked_list=[])

    assert metrics["is_no_hit"] is False
    assert metrics["precision_at_k"] == {"1": 0.0, "3": 0.0, "5": 0.0}
    assert metrics["recall_at_k"] == {"1": 0.0, "3": 0.0, "5": 0.0}
    assert metrics["mrr"] == 0.0
    assert metrics["ndcg_at_k"] == {"1": 0.0, "3": 0.0, "5": 0.0}
    assert metrics["document_recall_at_k"] == {"1": 0.0, "3": 0.0, "5": 0.0}
    assert metrics["expected_document_hit"] is False


def test_duplicate_chunks_are_deduped_before_metrics() -> None:
    qrels = {
        "sample_id": "duplicate_chunk",
        "documents": [{"source_doc": "doc-a.md", "relevance": 2}],
        "chunks": [{"source_doc": "doc-a.md", "chunk_index": 0, "relevance": 2}],
    }
    ranked = [
        {"source_doc": "doc-a.md", "chunk_index": 0, "rank": 1},
        {"source_doc": "doc-a.md", "chunk_index": 0, "rank": 2},
        {"source_doc": "doc-b.md", "chunk_index": 0, "rank": 3},
    ]

    metrics = compute_retrieval_metrics(qrels=qrels, ranked_list=ranked)

    assert dedupe_ranked_chunks(ranked) == [ranked[0], ranked[2]]
    assert metrics["deduped_ranked_count"] == 2
    assert metrics["duplicate_chunk_count"] == 1
    assert metrics["precision_at_k"]["3"] == pytest.approx(1 / 3)
    assert metrics["recall_at_k"]["3"] == 1.0


def test_document_metrics_use_source_doc_qrels() -> None:
    qrels = {
        "sample_id": "document_level",
        "documents": [
            {"source_doc": "doc-a.md", "relevance": 2},
            {"source_doc": "doc-b.md", "relevance": 1},
        ],
        "chunks": [{"source_doc": "doc-b.md", "chunk_index": 0, "relevance": 1}],
    }
    ranked = [
        {"source_doc": "doc-a.md", "chunk_index": 9},
        {"source_doc": "doc-c.md", "chunk_index": 0},
        {"source_doc": "doc-b.md", "chunk_index": 0},
    ]

    metrics = compute_retrieval_metrics(qrels=qrels, ranked_list=ranked)

    assert metrics["document_recall_at_k"]["1"] == pytest.approx(1 / 2)
    assert metrics["document_recall_at_k"]["3"] == 1.0
    assert metrics["expected_document_hit"] is True
    assert metrics["recall_at_k"]["3"] == 1.0


def test_no_hit_samples_are_excluded_from_core_averages() -> None:
    qrels_by_sample_id = {
        "hit": {
            "sample_id": "hit",
            "documents": [{"source_doc": "doc-a.md", "relevance": 2}],
            "chunks": [{"source_doc": "doc-a.md", "chunk_index": 0, "relevance": 2}],
        },
        "no_hit_clean": {"sample_id": "no_hit_clean", "documents": [], "chunks": []},
        "no_hit_false_positive": {"sample_id": "no_hit_false_positive", "documents": [], "chunks": []},
    }
    ranked_lists_by_sample_id = {
        "hit": [{"source_doc": "doc-a.md", "chunk_index": 0}],
        "no_hit_clean": [],
        "no_hit_false_positive": [{"source_doc": "doc-z.md", "chunk_index": 0}],
    }

    result = compute_retrieval_benchmark_metrics(
        qrels_by_sample_id=qrels_by_sample_id,
        ranked_lists_by_sample_id=ranked_lists_by_sample_id,
    )

    aggregate = result["aggregate"]
    assert aggregate["hit_sample_count"] == 1
    assert aggregate["no_hit_sample_count"] == 2
    assert aggregate["precision_at_k"]["1"] == 1.0
    assert aggregate["recall_at_k"]["1"] == 1.0
    assert aggregate["mrr"] == 1.0
    assert aggregate["no_hit_false_positive_rate"] == pytest.approx(1 / 2)
    assert result["samples"]["no_hit_clean"]["precision_at_k"]["1"] is None
    assert result["samples"]["no_hit_false_positive"]["no_hit_false_positive"] is True


def test_qrels_list_to_mapping_rejects_duplicate_sample_ids() -> None:
    payload = {
        "qrels": [
            {"sample_id": "same", "documents": [], "chunks": []},
            {"sample_id": "same", "documents": [], "chunks": []},
        ]
    }

    with pytest.raises(ValueError, match="duplicate qrels sample_id"):
        qrels_list_to_mapping(payload)

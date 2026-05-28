from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import log2
from typing import Any


DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5)


def compute_retrieval_metrics(
    *,
    qrels: Mapping[str, Any],
    ranked_list: Sequence[Mapping[str, Any]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Compute per-sample retrieval metrics from qrels and a safe ranked list."""
    normalized_k_values = _normalize_k_values(k_values)
    unique_ranked = dedupe_ranked_chunks(ranked_list)
    relevant_chunks = _relevant_chunk_map(qrels)
    relevant_docs = _relevant_document_map(qrels)
    is_no_hit = not relevant_chunks and not relevant_docs

    base: dict[str, Any] = {
        "is_no_hit": is_no_hit,
        "ranked_count": len(ranked_list),
        "deduped_ranked_count": len(unique_ranked),
        "duplicate_chunk_count": len(ranked_list) - len(unique_ranked),
    }
    if is_no_hit:
        base.update(
            {
                "precision_at_k": _none_by_k(normalized_k_values),
                "recall_at_k": _none_by_k(normalized_k_values),
                "mrr": None,
                "ndcg_at_k": _none_by_k(normalized_k_values),
                "document_recall_at_k": _none_by_k(normalized_k_values),
                "expected_document_hit": None,
                "no_hit_false_positive": bool(unique_ranked),
            }
        )
        return base

    base.update(
        {
            "precision_at_k": {
                str(k): _precision_at_k(unique_ranked, relevant_chunks, k)
                for k in normalized_k_values
            },
            "recall_at_k": {
                str(k): _recall_at_k(unique_ranked, relevant_chunks, k)
                for k in normalized_k_values
            },
            "mrr": _mrr(unique_ranked, relevant_chunks),
            "ndcg_at_k": {
                str(k): _ndcg_at_k(unique_ranked, relevant_chunks, k)
                for k in normalized_k_values
            },
            "document_recall_at_k": {
                str(k): _document_recall_at_k(unique_ranked, relevant_docs, k)
                for k in normalized_k_values
            },
            "expected_document_hit": _expected_document_hit(unique_ranked, relevant_docs),
            "no_hit_false_positive": None,
        }
    )
    return base


def compute_retrieval_benchmark_metrics(
    *,
    qrels_by_sample_id: Mapping[str, Mapping[str, Any]],
    ranked_lists_by_sample_id: Mapping[str, Sequence[Mapping[str, Any]]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Compute sample-level and aggregate retrieval benchmark metrics."""
    normalized_k_values = _normalize_k_values(k_values)
    samples = {
        sample_id: compute_retrieval_metrics(
            qrels=qrels,
            ranked_list=ranked_lists_by_sample_id.get(sample_id, []),
            k_values=normalized_k_values,
        )
        for sample_id, qrels in qrels_by_sample_id.items()
    }
    hit_samples = [item for item in samples.values() if item["is_no_hit"] is False]
    no_hit_samples = [item for item in samples.values() if item["is_no_hit"] is True]
    no_hit_false_positive_count = sum(1 for item in no_hit_samples if item["no_hit_false_positive"] is True)

    aggregate = {
        "sample_count": len(samples),
        "hit_sample_count": len(hit_samples),
        "no_hit_sample_count": len(no_hit_samples),
        "precision_at_k": _average_metric_by_k(hit_samples, "precision_at_k", normalized_k_values),
        "recall_at_k": _average_metric_by_k(hit_samples, "recall_at_k", normalized_k_values),
        "mrr": _average_scalar(hit_samples, "mrr"),
        "ndcg_at_k": _average_metric_by_k(hit_samples, "ndcg_at_k", normalized_k_values),
        "document_recall_at_k": _average_metric_by_k(hit_samples, "document_recall_at_k", normalized_k_values),
        "expected_document_hit": _average_bool(hit_samples, "expected_document_hit"),
        "no_hit_false_positive_rate": (
            no_hit_false_positive_count / len(no_hit_samples)
            if no_hit_samples
            else 0.0
        ),
    }
    return {
        "aggregate": aggregate,
        "samples": samples,
    }


def qrels_list_to_mapping(qrels_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert a qrels JSON payload into a sample_id-keyed mapping."""
    qrels_items = qrels_payload.get("qrels", [])
    if not isinstance(qrels_items, list):
        raise ValueError("qrels payload must contain a list field named 'qrels'.")
    mapping: dict[str, dict[str, Any]] = {}
    for item in qrels_items:
        if not isinstance(item, dict) or not item.get("sample_id"):
            raise ValueError("each qrels item must be an object with sample_id.")
        sample_id = str(item["sample_id"])
        if sample_id in mapping:
            raise ValueError(f"duplicate qrels sample_id: {sample_id}")
        mapping[sample_id] = dict(item)
    return mapping


def dedupe_ranked_chunks(ranked_list: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first occurrence of each chunk in ranked order."""
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in ranked_list:
        key = _chunk_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
    return deduped


def _normalize_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
    if not normalized:
        raise ValueError("k_values must contain at least one positive integer.")
    return normalized


def _none_by_k(k_values: Sequence[int]) -> dict[str, None]:
    return {str(k): None for k in k_values}


def _chunk_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    source_doc = item.get("source_doc")
    chunk_index = item.get("chunk_index")
    if source_doc is not None and chunk_index is not None:
        return ("source_doc_chunk_index", str(source_doc), int(chunk_index))
    chunk_id = item.get("chunk_id")
    if chunk_id is not None:
        return ("chunk_id", str(chunk_id))
    return ("ranked_item_identity", id(item))


def _document_key(item: Mapping[str, Any]) -> str | None:
    source_doc = item.get("source_doc")
    if source_doc is not None:
        return str(source_doc)
    return None


def _relevant_chunk_map(qrels: Mapping[str, Any]) -> dict[tuple[Any, ...], int]:
    relevant: dict[tuple[Any, ...], int] = {}
    for item in qrels.get("chunks", []) or []:
        if not isinstance(item, Mapping):
            continue
        relevance = int(item.get("relevance", 0))
        if relevance <= 0:
            continue
        relevant[_chunk_key(item)] = relevance
    return relevant


def _relevant_document_map(qrels: Mapping[str, Any]) -> dict[str, int]:
    relevant: dict[str, int] = {}
    for item in qrels.get("documents", []) or []:
        if not isinstance(item, Mapping):
            continue
        source_doc = item.get("source_doc")
        relevance = int(item.get("relevance", 0))
        if source_doc is None or relevance <= 0:
            continue
        relevant[str(source_doc)] = relevance
    return relevant


def _precision_at_k(
    ranked_list: Sequence[Mapping[str, Any]],
    relevant_chunks: Mapping[tuple[Any, ...], int],
    k: int,
) -> float:
    if not relevant_chunks:
        return 0.0
    top_k = ranked_list[:k]
    hits = sum(1 for item in top_k if _chunk_key(item) in relevant_chunks)
    return hits / k


def _recall_at_k(
    ranked_list: Sequence[Mapping[str, Any]],
    relevant_chunks: Mapping[tuple[Any, ...], int],
    k: int,
) -> float:
    if not relevant_chunks:
        return 0.0
    top_k = ranked_list[:k]
    hits = {_chunk_key(item) for item in top_k if _chunk_key(item) in relevant_chunks}
    return len(hits) / len(relevant_chunks)


def _mrr(
    ranked_list: Sequence[Mapping[str, Any]],
    relevant_chunks: Mapping[tuple[Any, ...], int],
) -> float:
    for rank, item in enumerate(ranked_list, start=1):
        if _chunk_key(item) in relevant_chunks:
            return 1 / rank
    return 0.0


def _ndcg_at_k(
    ranked_list: Sequence[Mapping[str, Any]],
    relevant_chunks: Mapping[tuple[Any, ...], int],
    k: int,
) -> float:
    if not relevant_chunks:
        return 0.0
    gains = [
        relevant_chunks.get(_chunk_key(item), 0)
        for item in ranked_list[:k]
    ]
    dcg = _dcg(gains)
    ideal_gains = sorted(relevant_chunks.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal_gains)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def _dcg(gains: Sequence[int]) -> float:
    return sum((2**gain - 1) / log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def _document_recall_at_k(
    ranked_list: Sequence[Mapping[str, Any]],
    relevant_docs: Mapping[str, int],
    k: int,
) -> float:
    if not relevant_docs:
        return 0.0
    retrieved_docs = {
        document_key
        for item in ranked_list[:k]
        if (document_key := _document_key(item)) is not None
    }
    return len(retrieved_docs & set(relevant_docs)) / len(relevant_docs)


def _expected_document_hit(
    ranked_list: Sequence[Mapping[str, Any]],
    relevant_docs: Mapping[str, int],
) -> bool:
    if not relevant_docs:
        return False
    expected_docs = set(relevant_docs)
    return any(_document_key(item) in expected_docs for item in ranked_list)


def _average_metric_by_k(
    samples: Sequence[Mapping[str, Any]],
    metric_name: str,
    k_values: Sequence[int],
) -> dict[str, float]:
    return {
        str(k): _average(
            [
                sample.get(metric_name, {}).get(str(k))
                for sample in samples
                if isinstance(sample.get(metric_name), Mapping)
            ]
        )
        for k in k_values
    }


def _average_scalar(samples: Sequence[Mapping[str, Any]], metric_name: str) -> float:
    return _average([sample.get(metric_name) for sample in samples])


def _average_bool(samples: Sequence[Mapping[str, Any]], metric_name: str) -> float:
    return _average(
        [
            1.0 if sample.get(metric_name) is True else 0.0
            for sample in samples
            if sample.get(metric_name) is not None
        ]
    )


def _average(values: Sequence[Any]) -> float:
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    if not numeric_values:
        return 0.0
    return sum(numeric_values) / len(numeric_values)

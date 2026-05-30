from __future__ import annotations

import json
from pathlib import Path
import re

from backend.evals.run_http_eval import (
    _build_assertions,
    _build_observed_from_chat_response,
    _extract_policy_evidence,
    _parse_sse_events,
    build_comparison_payload,
)
from backend.platform.knowledge.documents.chunker import SlidingWindowTextSplitter
from backend.platform.knowledge.processing.config import DEFAULT_PROCESSING_CHUNK_CONFIG


ROOT_DIR = Path(__file__).resolve().parents[2]
EVALS_DIR = ROOT_DIR / "backend" / "evals"
SAMPLES_DIR = EVALS_DIR / "samples"
FIXTURES_DIR = EVALS_DIR / "fixtures"
QRELS_DIR = EVALS_DIR / "qrels"
FORBIDDEN_TERMS = ("ecommerce", "product", "inventory", "sku", "商品", "订单", "库存")
FORBIDDEN_ID_PATTERNS = (
    re.compile(r"\bP\d{3,}\b", re.IGNORECASE),
    re.compile(r"\bSKU\b", re.IGNORECASE),
    re.compile(r"\bO\d{6,}\b", re.IGNORECASE),
)


def test_minimal_eval_sample_manifest_is_well_formed() -> None:
    manifest = json.loads((SAMPLES_DIR / "minimal.json").read_text(encoding="utf-8"))

    assert manifest["sample_set"] == "minimal"
    assert manifest["namespace"] == "documents"
    assert isinstance(manifest["fixtures"], list) and manifest["fixtures"]
    assert isinstance(manifest["samples"], list) and manifest["samples"]

    fixture_ids = set()
    fixture_filenames = set()
    for fixture in manifest["fixtures"]:
        assert set(fixture) == {"id", "filename"}
        assert fixture["id"] not in fixture_ids
        fixture_ids.add(fixture["id"])
        fixture_filenames.add(fixture["filename"])
        assert (FIXTURES_DIR / fixture["filename"]).exists()
        normalized = f"{fixture['id']} {fixture['filename']}".lower()
        assert not any(term in normalized for term in FORBIDDEN_TERMS)

    sample_ids = set()
    for sample in manifest["samples"]:
        assert {"sample_id", "query", "source_doc", "expected"} <= set(sample)
        assert sample["sample_id"] not in sample_ids
        sample_ids.add(sample["sample_id"])
        normalized = sample["sample_id"].lower()
        assert not any(term in normalized for term in FORBIDDEN_TERMS)
        if sample["source_doc"] is not None:
            assert sample["source_doc"] in fixture_filenames
        assert isinstance(sample["expected"], dict)
        assert "knowledge_used" in sample["expected"]
        assert "min_citations" in sample["expected"] or "max_citations" in sample["expected"]
        for pattern in FORBIDDEN_ID_PATTERNS:
            assert not pattern.search(sample["query"])


def test_minimal_eval_includes_strict_no_hit_fallback_boundary() -> None:
    manifest = json.loads((SAMPLES_DIR / "minimal.json").read_text(encoding="utf-8"))
    samples = {sample["sample_id"]: sample for sample in manifest["samples"]}

    assert "no_hit_fallback" in samples
    no_hit = samples["no_hit_fallback"]

    assert no_hit["source_doc"] is None
    assert no_hit["expected"]["knowledge_used"] is False
    assert no_hit["expected"]["max_citations"] == 0
    assert no_hit["expected"]["citations_empty"] is True
    assert no_hit["expected"]["requires_visible_marker"] is False
    assert no_hit["expected"]["answer_contains_any"]


def test_minimal_eval_includes_stream_replay_boundaries() -> None:
    manifest = json.loads((SAMPLES_DIR / "minimal.json").read_text(encoding="utf-8"))
    stream_samples = {
        sample["sample_id"]: sample
        for sample in manifest["samples"]
        if sample.get("eval_stream") is True
    }

    assert "quickstart_setup_requirement" in stream_samples
    assert "no_hit_fallback" in stream_samples
    assert stream_samples["quickstart_setup_requirement"]["expected"]["knowledge_used"] is True
    assert stream_samples["no_hit_fallback"]["expected"]["knowledge_used"] is False


def test_retrieval_benchmark_manifest_is_well_formed() -> None:
    manifest = json.loads((SAMPLES_DIR / "retrieval_benchmark.json").read_text(encoding="utf-8"))

    assert manifest["sample_set"] == "retrieval_benchmark"
    assert manifest["namespace"] == "documents"
    assert manifest["append_eval_anchors"] is False
    assert manifest["qrels_path"] == "retrieval_benchmark.json"
    assert len(manifest["fixtures"]) == 5
    assert len(manifest["samples"]) == 16

    fixture_ids = set()
    fixture_filenames = set()
    for fixture in manifest["fixtures"]:
        assert set(fixture) == {"id", "filename"}
        assert fixture["id"] not in fixture_ids
        fixture_ids.add(fixture["id"])
        fixture_filenames.add(fixture["filename"])
        assert fixture["filename"].startswith("eval-benchmark-")
        assert (FIXTURES_DIR / fixture["filename"]).exists()

    sample_ids = set()
    for sample in manifest["samples"]:
        assert {"sample_id", "query", "source_doc", "expected"} <= set(sample)
        assert sample["sample_id"] not in sample_ids
        sample_ids.add(sample["sample_id"])
        if sample["source_doc"] is not None:
            assert sample["source_doc"] in fixture_filenames
            assert sample["expected"]["knowledge_used"] is True
            assert sample["expected"]["min_citations"] >= 1
        else:
            assert sample["expected"]["knowledge_used"] is False
            assert sample["expected"]["max_citations"] == 0
            assert sample["expected"]["citations_empty"] is True


def test_retrieval_benchmark_qrels_align_with_manifest_and_fixtures() -> None:
    manifest = json.loads((SAMPLES_DIR / "retrieval_benchmark.json").read_text(encoding="utf-8"))
    qrels_payload = json.loads((QRELS_DIR / manifest["qrels_path"]).read_text(encoding="utf-8"))

    assert qrels_payload["sample_set"] == manifest["sample_set"]
    assert qrels_payload["namespace"] == manifest["namespace"]

    fixture_filenames = {fixture["filename"] for fixture in manifest["fixtures"]}
    sample_by_id = {sample["sample_id"]: sample for sample in manifest["samples"]}
    qrels_by_sample_id = {item["sample_id"]: item for item in qrels_payload["qrels"]}

    assert set(qrels_by_sample_id) == set(sample_by_id)

    chunker = SlidingWindowTextSplitter(
        chunk_size=DEFAULT_PROCESSING_CHUNK_CONFIG.chunk_size,
        chunk_overlap=DEFAULT_PROCESSING_CHUNK_CONFIG.chunk_overlap,
    )
    chunk_counts = {
        filename: len(chunker.split_text((FIXTURES_DIR / filename).read_text(encoding="utf-8").strip()))
        for filename in fixture_filenames
    }

    for sample_id, sample in sample_by_id.items():
        qrels = qrels_by_sample_id[sample_id]
        documents = qrels["documents"]
        chunks = qrels["chunks"]
        if sample["source_doc"] is None:
            assert documents == []
            assert chunks == []
            continue

        assert documents
        assert chunks
        referenced_docs = {item["source_doc"] for item in documents}
        assert sample["source_doc"] in referenced_docs
        assert referenced_docs <= fixture_filenames
        for item in documents:
            assert item["relevance"] in {1, 2}
        for item in chunks:
            assert item["source_doc"] in fixture_filenames
            assert item["relevance"] in {1, 2}
            assert 0 <= item["chunk_index"] < chunk_counts[item["source_doc"]]


def test_evaluation_harness_documents_scene_retrieval_policy_observability() -> None:
    guide = (EVALS_DIR / "evaluation-harness.md").read_text(encoding="utf-8")

    assert "retrieval policy" in guide
    assert "tool" in guide
    for field in (
        "top_k",
        "min_relevance_score",
        "recall_strategy",
        "no_hit_strategy",
        "rerank_enabled",
        "rerank_top_n",
    ):
        assert field in guide


def test_sse_parser_extracts_event_types_and_json_data() -> None:
    raw = (
        'event: start\ndata: {"request_id": "req-1"}\n\n'
        'event: chunk\ndata: {"delta": "hello"}\n\n'
        'event: done\ndata: {"answer": "hello", "knowledge_used": false, "citations": []}\n\n'
    )

    events = _parse_sse_events(raw)

    assert [event["event"] for event in events] == ["start", "chunk", "done"]
    assert events[1]["data"]["delta"] == "hello"
    assert events[2]["data"]["citations"] == []


def test_eval_extracts_safe_policy_evidence_from_sse_tool_event() -> None:
    events = [
        {
            "event": "tool",
            "data": {
                "mode": "agentic",
                "retrieval_policy": {
                    "top_k": 5,
                    "min_relevance_score": 0.8,
                    "recall_strategy": "hybrid",
                    "no_hit_strategy": "ask_user",
                    "rerank_enabled": False,
                    "rerank_top_n": None,
                },
                "candidate_tools": ["knowledge_document_search"],
                "documents": 1,
                "exit_reason": "sufficient",
                "success": True,
                "rounds": [
                    {
                        "round_index": 1,
                        "tool_name": "knowledge_document_search",
                        "query": "private query should not be copied",
                        "reason": "private reason should not be copied",
                        "decision": "finish",
                        "is_sufficient": True,
                        "result_count": 1,
                        "document_count": 1,
                        "success": True,
                    }
                ],
                "citations": [{"snippet": "raw private source content"}],
            },
        }
    ]

    evidence = _extract_policy_evidence(events)

    assert evidence is not None
    assert evidence["retrieval_policy"]["recall_strategy"] == "hybrid"
    assert evidence["candidate_tools"] == ["knowledge_document_search"]
    assert evidence["rounds"] == [
        {
            "round_index": 1,
            "tool_name": "knowledge_document_search",
            "decision": "finish",
            "is_sufficient": True,
            "result_count": 1,
            "document_count": 1,
            "success": True,
            "rerank": None,
        }
    ]
    assert "private query" not in json.dumps(evidence, ensure_ascii=False)
    assert "raw private source content" not in json.dumps(evidence, ensure_ascii=False)


def test_no_hit_assertions_fail_when_pseudo_citations_are_returned() -> None:
    expected = {
        "knowledge_used": False,
        "max_citations": 0,
        "citations_empty": True,
        "requires_visible_marker": False,
    }
    observed = {
        "knowledge_used": True,
        "citation_count": 1,
        "citations": [{"source_name": "eval-harness-quickstart.md"}],
        "citation_sources": ["eval-harness-quickstart.md"],
    }
    metrics = {
        "answer_keyword_hit": True,
        "expected_source_seen": False,
        "expected_source_kind_seen": True,
        "visible_marker_seen": True,
        "fallback_like": False,
    }

    assertions = _build_assertions(expected=expected, observed=observed, metrics=metrics)
    failures = {assertion["name"]: assertion for assertion in assertions if not assertion["passed"]}

    assert failures["knowledge_used"]["actual"] is True
    assert failures["max_citations"]["actual"] == 1
    assert failures["citations_empty"]["actual"] == [{"source_name": "eval-harness-quickstart.md"}]


def test_eval_observed_payload_preserves_retrieval_trace_without_full_chunk_content() -> None:
    observed = _build_observed_from_chat_response(
        {
            "answer": "hello [1]",
            "knowledge_used": True,
            "citations": [
                {
                    "citation_id": "chunk-1",
                    "chunk_id": "chunk-1",
                    "source_name": "doc.md",
                    "snippet": "existing citation snippet",
                }
            ],
            "retrieval_trace": {
                "original_query": "hello",
                "final_query": "hello",
                "rewritten_query": None,
                "tool_call_count": 1,
                "candidate_tools": ["knowledge_document_search"],
                "exit_reason": "sufficient",
                "final_decision": "answer_with_evidence",
                "success": True,
                "raw_candidates_count": 1,
                "filtered_candidates_count": 1,
                "knowledge_used": True,
                "top_k_chunks": [
                    {
                        "rank": 1,
                        "citation_id": "chunk-1",
                        "chunk_id": "chunk-1",
                        "source_name": "doc.md",
                        "score": 0.9,
                    }
                ],
                "citations": [],
                "rounds": [],
            },
        }
    )

    assert observed["retrieval_trace"]["top_k_chunks"][0]["citation_id"] == "chunk-1"
    assert observed["retrieval_trace"]["final_decision"] == "answer_with_evidence"
    assert observed["retrieval_trace"]["success"] is True
    assert "full source content" not in json.dumps(observed["retrieval_trace"], ensure_ascii=False)


def test_eval_assertions_require_no_hit_trace_and_citation_match() -> None:
    expected = {"knowledge_used": False, "citations_empty": True}
    observed = {
        "answer": "暂时没有检索到足够相关的文档知识",
        "knowledge_used": False,
        "citation_count": 0,
        "citations": [],
        "citation_sources": [],
        "retrieval_trace": {
            "knowledge_used": False,
            "filtered_candidates_count": 0,
            "top_k_chunks": [],
        },
    }
    metrics = {
        "answer_keyword_hit": True,
        "expected_source_seen": False,
        "expected_source_kind_seen": True,
        "visible_marker_seen": False,
        "fallback_like": True,
    }

    assertions = _build_assertions(expected=expected, observed=observed, metrics=metrics)
    by_name = {assertion["name"]: assertion for assertion in assertions}

    assert by_name["retrieval_trace_present"]["passed"] is True
    assert by_name["retrieval_trace_no_hit"]["passed"] is True

    hit_expected = {"knowledge_used": True}
    hit_observed = {
        "answer": "hello [1]",
        "knowledge_used": True,
        "citation_count": 1,
        "citations": [{"citation_id": "chunk-1", "chunk_id": "chunk-1"}],
        "citation_sources": ["doc.md"],
        "retrieval_trace": {
            "knowledge_used": True,
            "filtered_candidates_count": 1,
            "top_k_chunks": [{"citation_id": "chunk-1", "chunk_id": "chunk-1"}],
        },
    }
    hit_metrics = {
        "answer_keyword_hit": True,
        "expected_source_seen": False,
        "expected_source_kind_seen": True,
        "visible_marker_seen": True,
        "fallback_like": False,
    }

    hit_assertions = _build_assertions(
        expected=hit_expected,
        observed=hit_observed,
        metrics=hit_metrics,
    )
    hit_by_name = {assertion["name"]: assertion for assertion in hit_assertions}

    assert hit_by_name["retrieval_trace_citations_match_top_chunks"]["passed"] is True


def test_eval_comparison_reports_policy_and_no_hit_regression() -> None:
    baseline = {
        "run_id": "baseline",
        "sample_set": "minimal",
        "results": [
            {
                "sample_id": "no_hit_fallback",
                "status": "ok",
                "passed": True,
                "observed": {
                    "knowledge_used": False,
                    "citation_count": 0,
                    "citations": [],
                },
                "stream": {
                    "passed": True,
                    "event_types": ["start", "history", "tool", "chunk", "done"],
                    "observed": {"knowledge_used": False, "citation_count": 0},
                    "policy_evidence": {
                        "retrieval_policy": {
                            "recall_strategy": "hybrid",
                            "rerank_enabled": False,
                        }
                    },
                },
            }
        ],
    }
    candidate = {
        "run_id": "candidate",
        "sample_set": "minimal",
        "results": [
            {
                "sample_id": "no_hit_fallback",
                "status": "ok",
                "passed": False,
                "observed": {
                    "knowledge_used": True,
                    "citation_count": 1,
                    "citations": [{"source_name": "pseudo.md"}],
                },
                "stream": {
                    "passed": False,
                    "event_types": ["start", "history", "tool", "done"],
                    "observed": {"knowledge_used": True, "citation_count": 1},
                    "policy_evidence": {
                        "retrieval_policy": {
                            "recall_strategy": "keyword",
                            "rerank_enabled": True,
                        }
                    },
                },
            }
        ],
    }

    comparison = build_comparison_payload(
        baseline_payload=baseline,
        candidate_payload=candidate,
    )
    sample = comparison["samples"][0]

    assert comparison["summary"]["changed_samples"] == 1
    assert sample["sample_id"] == "no_hit_fallback"
    assert "knowledge_used" in sample["differences"]
    assert "citation_count" in sample["differences"]
    assert "no_hit_citations_empty" in sample["differences"]
    assert "policy_evidence" in sample["differences"]
    assert sample["compared"]["no_hit_citations_empty"]["baseline"] is True
    assert sample["compared"]["no_hit_citations_empty"]["candidate"] is False


def test_eval_comparison_reports_missing_samples_by_sample_id() -> None:
    comparison = build_comparison_payload(
        baseline_payload={
            "run_id": "baseline",
            "results": [{"sample_id": "baseline_only", "passed": True}],
        },
        candidate_payload={
            "run_id": "candidate",
            "results": [{"sample_id": "candidate_only", "passed": True}],
        },
    )
    statuses = {item["sample_id"]: item["status"] for item in comparison["samples"]}

    assert statuses == {
        "baseline_only": "missing_candidate",
        "candidate_only": "missing_baseline",
    }


def test_eval_fixtures_remain_generic_document_assets() -> None:
    fixture_names = sorted(path.name for path in FIXTURES_DIR.glob("*.md"))

    assert fixture_names == [
        "eval-benchmark-access-control.md",
        "eval-benchmark-quickstart.md",
        "eval-benchmark-release-runbook.md",
        "eval-benchmark-security-policy.md",
        "eval-benchmark-support-faq.md",
        "eval-harness-it-policy.md",
        "eval-harness-quickstart.md",
        "eval-harness-support-faq.md",
    ]
    for name in fixture_names:
        assert not any(term in name.lower() for term in FORBIDDEN_TERMS)

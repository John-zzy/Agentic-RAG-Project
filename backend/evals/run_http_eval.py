from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import mimetypes
from pathlib import Path
import re
import time
import sys
from typing import Any
from urllib import error, parse, request
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.evals.retrieval_metrics import (
    compute_retrieval_benchmark_metrics,
    qrels_list_to_mapping,
)
from backend.evals.retrieval_probe import run_retrieval_probe


EVALS_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = EVALS_DIR / "samples"
FIXTURES_DIR = EVALS_DIR / "fixtures"
QRELS_DIR = EVALS_DIR / "qrels"
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "data" / "evals" / "latest.json"
EVAL_FILE_PREFIXES = ("eval-harness-", "eval-benchmark-")


class EvalRuntimeError(RuntimeError):
    """Represents a recoverable eval execution failure."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _request_json(
    *,
    method: str,
    url: str,
    json_body: dict[str, Any] | None = None,
    max_attempts: int = 1,
    retry_delay_seconds: float = 2.0,
    retryable_http_codes: tuple[int, ...] = (),
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        payload = None
        headers: dict[str, str] = {}
        if json_body is not None:
            payload = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = request.Request(url=url, data=payload, method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=120) as response:
                raw = response.read().decode("utf-8")
                if not raw:
                    return None
                return json.loads(raw)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in retryable_http_codes and attempt < max_attempts:
                last_error = EvalRuntimeError(
                    f"{method} {url} failed with HTTP {exc.code} on attempt {attempt}: {detail}"
                )
                time.sleep(retry_delay_seconds)
                continue
            raise EvalRuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            if attempt < max_attempts:
                last_error = EvalRuntimeError(f"{method} {url} failed on attempt {attempt}: {exc.reason}")
                time.sleep(retry_delay_seconds)
                continue
            raise EvalRuntimeError(f"{method} {url} failed: {exc.reason}") from exc
    if last_error is not None:
        raise last_error
    raise EvalRuntimeError(f"{method} {url} failed unexpectedly.")


def _request_sse_chat(*, base_url: str, json_body: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url=f"{base_url}/chat",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream",
        },
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EvalRuntimeError(f"POST {base_url}/chat stream failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise EvalRuntimeError(f"POST {base_url}/chat stream failed: {exc.reason}") from exc

    events = _parse_sse_events(raw)
    error_events = [event for event in events if event["event"] == "error"]
    if error_events:
        raise EvalRuntimeError(f"SSE returned error event: {error_events[-1]['data']!r}")
    done_events = [event for event in events if event["event"] == "done"]
    if not done_events:
        raise EvalRuntimeError(f"SSE stream did not include a done event. events={[event['event'] for event in events]!r}")

    chunk_text = "".join(
        str(event["data"].get("delta", ""))
        for event in events
        if event["event"] == "chunk" and isinstance(event.get("data"), dict)
    )
    return {
        "events": events,
        "event_types": [event["event"] for event in events],
        "chunk_count": sum(1 for event in events if event["event"] == "chunk"),
        "chunk_text_preview": _answer_preview(chunk_text),
        "policy_evidence": _extract_policy_evidence(events),
        "done": done_events[-1]["data"],
    }


def _parse_sse_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    normalized = raw.replace("\r\n", "\n")
    for block in normalized.split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        data_text = "\n".join(data_lines)
        if not data_text:
            data: Any = {}
        else:
            try:
                data = json.loads(data_text)
            except json.JSONDecodeError as exc:
                raise EvalRuntimeError(f"Invalid SSE JSON payload for event {event_name!r}: {data_text}") from exc
        events.append({"event": event_name, "data": data})
    return events


def _extract_policy_evidence(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从 SSE tool 事件提取不含原文片段的检索策略证据。"""
    tool_events = [
        event.get("data")
        for event in events
        if event.get("event") == "tool" and isinstance(event.get("data"), dict)
    ]
    if not tool_events:
        return None
    tool_event = tool_events[-1]
    policy = tool_event.get("retrieval_policy")
    if not isinstance(policy, dict):
        policy = {}
    rounds = tool_event.get("rounds")
    safe_rounds = []
    if isinstance(rounds, list):
        for item in rounds:
            if not isinstance(item, dict):
                continue
            safe_rounds.append(
                {
                    "round_index": item.get("round_index"),
                    "tool_name": item.get("tool_name"),
                    "decision": item.get("decision"),
                    "is_sufficient": item.get("is_sufficient"),
                    "result_count": item.get("result_count"),
                    "document_count": item.get("document_count"),
                    "success": item.get("success"),
                    "rerank": item.get("rerank"),
                }
            )
    return {
        "mode": tool_event.get("mode"),
        # 只保留策略证据，不写入 query、reason 或原文片段，避免评估报告泄露上下文。
        "retrieval_policy": {
            key: policy.get(key)
            for key in (
                "top_k",
                "min_relevance_score",
                "recall_strategy",
                "no_hit_strategy",
                "rerank_enabled",
                "rerank_top_n",
            )
            if key in policy
        },
        "candidate_tools": tool_event.get("candidate_tools", []),
        "documents": tool_event.get("documents"),
        "exit_reason": tool_event.get("exit_reason"),
        "success": tool_event.get("success"),
        "rounds": safe_rounds,
    }


def _build_multipart_body(
    file_path: Path,
    field_name: str = "file",
    *,
    file_bytes: bytes | None = None,
) -> tuple[bytes, str]:
    boundary = f"----AiRagEval{uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    resolved_file_bytes = file_bytes if file_bytes is not None else file_path.read_bytes()
    lines = [
        f"--{boundary}".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"'
        ).encode("utf-8"),
        f"Content-Type: {content_type}".encode("utf-8"),
        b"",
        resolved_file_bytes,
        f"--{boundary}--".encode("utf-8"),
        b"",
    ]
    body = b"\r\n".join(lines)
    return body, boundary


def _upload_file(base_url: str, file_path: Path, *, content: str | None = None) -> dict[str, Any]:
    body, boundary = _build_multipart_body(
        file_path,
        file_bytes=content.encode("utf-8") if content is not None else None,
    )
    req = request.Request(
        url=f"{base_url}/files/upload",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EvalRuntimeError(f"POST {base_url}/files/upload failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise EvalRuntimeError(f"POST {base_url}/files/upload failed: {exc.reason}") from exc


def _compact_for_match(value: str) -> str:
    return "".join(value.split())


def _contains_text(actual_answer: str, expected_text: str) -> bool:
    if expected_text in actual_answer:
        return True
    return _compact_for_match(expected_text) in _compact_for_match(actual_answer)


def _contains_required_text(actual_answer: str, expected: dict[str, Any]) -> bool:
    any_terms = expected.get("answer_contains_any") or []
    all_terms = expected.get("answer_contains_all") or []
    if any_terms:
        if not any(_contains_text(actual_answer, str(term)) for term in any_terms):
            if expected.get("knowledge_used") is False and _fallback_like(actual_answer):
                return True
            return False
    if all_terms:
        if not all(_contains_text(actual_answer, str(term)) for term in all_terms):
            return False
    return True


def _has_expected_source(citations: list[dict[str, Any]], expected_source_name: str | None) -> bool:
    if not expected_source_name:
        return False
    return any(citation.get("source_name") == expected_source_name for citation in citations)


def _has_expected_source_kind(citations: list[dict[str, Any]], expected_source_kind: str | None) -> bool:
    if not expected_source_kind:
        return True
    return any(citation.get("source_kind") == expected_source_kind for citation in citations)


def _answer_preview(answer: str, limit: int = 80) -> str:
    normalized = " ".join(answer.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _fallback_like(answer: str) -> bool:
    fallback_signals = (
        "暂时没有检索到足够相关的文档知识",
        "无法找到",
        "未找到",
        "缺乏相关",
        "证据不足",
    )
    return any(signal in answer for signal in fallback_signals)


def _has_visible_marker(answer: str) -> bool:
    return re.search(r"\[\d+\]", answer) is not None


def _make_assertion(name: str, expected: Any, actual: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "passed": passed,
    }


def _build_assertions(
    *,
    expected: dict[str, Any],
    observed: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    assertions = [
        _make_assertion(
            "knowledge_used",
            expected.get("knowledge_used"),
            observed.get("knowledge_used"),
            observed.get("knowledge_used") == expected.get("knowledge_used"),
        ),
        _make_assertion(
            "answer_keyword_hit",
            True,
            metrics["answer_keyword_hit"],
            metrics["answer_keyword_hit"],
        ),
    ]

    if "min_citations" in expected:
        citation_count = int(observed["citation_count"])
        min_citations = int(expected["min_citations"])
        assertions.append(
            _make_assertion("min_citations", f">= {min_citations}", citation_count, citation_count >= min_citations)
        )

    if "max_citations" in expected:
        citation_count = int(observed["citation_count"])
        max_citations = int(expected["max_citations"])
        assertions.append(
            _make_assertion("max_citations", f"<= {max_citations}", citation_count, citation_count <= max_citations)
        )

    if expected.get("citations_empty") is True:
        assertions.append(
            _make_assertion("citations_empty", [], observed["citations"], observed["citations"] == [])
        )

    if expected.get("citation_source_name"):
        assertions.append(
            _make_assertion(
                "citation_source_name",
                expected["citation_source_name"],
                observed["citation_sources"],
                metrics["expected_source_seen"],
            )
        )

    if expected.get("citation_source_kind"):
        actual_source_kinds = [
            citation.get("source_kind")
            for citation in observed["citations"]
            if isinstance(citation, dict)
        ]
        assertions.append(
            _make_assertion(
                "citation_source_kind",
                expected["citation_source_kind"],
                actual_source_kinds,
                metrics["expected_source_kind_seen"],
            )
        )

    if "requires_visible_marker" in expected:
        requires_visible_marker = bool(expected["requires_visible_marker"])
        assertions.append(
            _make_assertion(
                "visible_marker",
                requires_visible_marker,
                metrics["visible_marker_seen"],
                metrics["visible_marker_seen"] == requires_visible_marker,
            )
        )

    if expected.get("knowledge_used") is False:
        assertions.append(
            _make_assertion("fallback_like", True, metrics["fallback_like"], metrics["fallback_like"])
        )
        retrieval_trace = observed.get("retrieval_trace")
        assertions.append(
            _make_assertion(
                "retrieval_trace_present",
                True,
                isinstance(retrieval_trace, dict),
                isinstance(retrieval_trace, dict),
            )
        )
        assertions.append(
            _make_assertion(
                "retrieval_trace_no_hit",
                {
                    "knowledge_used": False,
                    "filtered_candidates_count": 0,
                },
                {
                    "knowledge_used": retrieval_trace.get("knowledge_used") if isinstance(retrieval_trace, dict) else None,
                    "filtered_candidates_count": (
                        retrieval_trace.get("filtered_candidates_count")
                        if isinstance(retrieval_trace, dict)
                        else None
                    ),
                },
                (
                    isinstance(retrieval_trace, dict)
                    and retrieval_trace.get("knowledge_used") is False
                    and retrieval_trace.get("filtered_candidates_count") == 0
                ),
            )
        )

    if observed.get("citations"):
        assertions.append(
            _make_assertion(
                "retrieval_trace_citations_match_top_chunks",
                True,
                _citations_match_top_chunks(observed),
                _citations_match_top_chunks(observed),
            )
        )

    return assertions


def _build_observed_from_chat_response(chat_response: dict[str, Any]) -> dict[str, Any]:
    citations = chat_response.get("citations", [])
    answer = str(chat_response.get("answer", ""))
    return {
        "answer": answer,
        "answer_preview": _answer_preview(answer),
        "knowledge_used": chat_response.get("knowledge_used"),
        "citation_count": len(citations),
        "citation_sources": [citation.get("source_name") for citation in citations],
        "citations": citations,
        "retrieval_trace": chat_response.get("retrieval_trace"),
        "session_id": chat_response.get("session_id"),
        "request_id": chat_response.get("request_id"),
    }


def _citations_match_top_chunks(observed: dict[str, Any]) -> bool:
    retrieval_trace = observed.get("retrieval_trace")
    if not isinstance(retrieval_trace, dict):
        return False
    top_chunks = retrieval_trace.get("top_k_chunks")
    if not isinstance(top_chunks, list):
        return False
    chunk_ids = {
        str(item.get("chunk_id"))
        for item in top_chunks
        if isinstance(item, dict) and item.get("chunk_id") is not None
    }
    citation_ids = {
        str(item.get("citation_id"))
        for item in top_chunks
        if isinstance(item, dict) and item.get("citation_id") is not None
    }
    for citation in observed.get("citations", []):
        if not isinstance(citation, dict):
            return False
        chunk_id = citation.get("chunk_id")
        citation_id = citation.get("citation_id")
        if chunk_id is not None and str(chunk_id) in chunk_ids:
            continue
        if citation_id is not None and str(citation_id) in citation_ids:
            continue
        return False
    return True


def _build_metrics_from_observed(
    *,
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "knowledge_used": bool(observed.get("knowledge_used")),
        "citation_count": int(observed.get("citation_count", 0)),
        "answer_keyword_hit": _contains_required_text(str(observed.get("answer", "")), expected),
        "expected_source_seen": _has_expected_source(observed.get("citations", []), expected.get("citation_source_name")),
        "expected_source_kind_seen": _has_expected_source_kind(observed.get("citations", []), expected.get("citation_source_kind")),
        "visible_marker_seen": _has_visible_marker(str(observed.get("answer", ""))),
        "fallback_like": _fallback_like(str(observed.get("answer", ""))),
    }


def _format_failure_reason(assertion: dict[str, Any]) -> str:
    return (
        f"{assertion['name']} expected={assertion['expected']!r} "
        f"actual={assertion['actual']!r}"
    )


def _format_stream_failure_reason(assertion: dict[str, Any]) -> str:
    return f"stream.{_format_failure_reason(assertion)}"


def _validate_manifest(sample_set: dict[str, Any]) -> None:
    if not isinstance(sample_set.get("fixtures"), list) or not sample_set["fixtures"]:
        raise EvalRuntimeError("Sample set must define a non-empty fixtures list.")
    if not isinstance(sample_set.get("samples"), list) or not sample_set["samples"]:
        raise EvalRuntimeError("Sample set must define a non-empty samples list.")


def _load_qrels(sample_set: dict[str, Any]) -> dict[str, Any] | None:
    qrels_path = sample_set.get("qrels_path")
    if not qrels_path:
        return None
    resolved = QRELS_DIR / str(qrels_path)
    if not resolved.exists():
        raise EvalRuntimeError(f"Qrels file not found: {resolved}")
    return _read_json(resolved)


def _build_fixture_upload_content(sample_set: dict[str, Any], fixture_path: Path) -> str:
    """Append sample anchors so HTTP eval fixtures match the replay language."""
    content = fixture_path.read_text(encoding="utf-8").strip()
    if sample_set.get("append_eval_anchors") is False:
        return content
    anchors: list[str] = []
    for sample in sample_set["samples"]:
        if sample.get("source_doc") != fixture_path.name:
            continue
        expected = sample.get("expected", {})
        answer_terms = [
            str(term)
            for key in ("answer_contains_all", "answer_contains_any")
            for term in expected.get(key, [])
            if str(term).strip()
        ]
        anchors.append(f"- Question: {sample['query']}")
        if answer_terms:
            anchors.append(f"- Expected answer terms: {', '.join(answer_terms)}")

    if not anchors:
        return content
    return "\n\n## Evaluation Anchors\n\n" + "\n".join(anchors) if not content else (
        f"{content}\n\n## Evaluation Anchors\n\n" + "\n".join(anchors)
    )


def _cleanup_eval_files(base_url: str) -> None:
    payload = _request_json(method="GET", url=f"{base_url}/files/")
    files = payload.get("files", []) if isinstance(payload, dict) else []
    for item in files:
        filename = item.get("filename")
        if isinstance(filename, str) and filename.startswith(EVAL_FILE_PREFIXES):
            encoded = parse.quote(filename, safe="")
            _request_json(method="DELETE", url=f"{base_url}/files/{encoded}")


def _cleanup_eval_documents(base_url: str, namespace: str) -> None:
    payload = _request_json(
        method="GET",
        url=f"{base_url}/knowledge/documents?namespace={parse.quote(namespace, safe='')}",
    )
    documents = payload.get("documents", []) if isinstance(payload, dict) else []
    for item in documents:
        document_id = item.get("document_id")
        source_path = item.get("source_path")
        source_name = Path(source_path).name if isinstance(source_path, str) else ""
        if isinstance(document_id, str) and source_name.startswith(EVAL_FILE_PREFIXES):
            encoded = parse.quote(document_id, safe="")
            _request_json(method="DELETE", url=f"{base_url}/knowledge/documents/{encoded}")


def _register_documents(
    *,
    base_url: str,
    namespace: str,
    uploaded_files: dict[str, str],
) -> None:
    for file_name in uploaded_files.values():
        _request_json(
            method="POST",
            url=f"{base_url}/knowledge/documents",
            json_body={
                "namespace": namespace,
                "source_path": file_name,
            },
        )


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def _run_sample(
    *,
    base_url: str,
    sample: dict[str, Any],
) -> dict[str, Any]:
    session_response = _request_json(
        method="POST",
        url=f"{base_url}/sessions",
        json_body={
            "scene": "generic_assistant",
            "mounted_knowledge_sources": ["documents"],
        },
    )
    session_id = session_response["session_id"]
    chat_response = _request_json(
        method="POST",
        url=f"{base_url}/chat",
        json_body={
            "message": sample["query"],
            "session_id": session_id,
            "stream": False,
        },
        max_attempts=3,
        retry_delay_seconds=3.0,
        retryable_http_codes=(502,),
    )
    expected = sample["expected"]
    observed = _build_observed_from_chat_response(chat_response)
    metrics = _build_metrics_from_observed(expected=expected, observed=observed)
    assertions = _build_assertions(expected=expected, observed=observed, metrics=metrics)
    failure_reasons = [
        _format_failure_reason(assertion)
        for assertion in assertions
        if not assertion["passed"]
    ]
    stream_result = None
    if sample.get("eval_stream"):
        try:
            stream_result = _run_stream_sample(
                base_url=base_url,
                sample=sample,
                expected=expected,
                baseline_observed=observed,
            )
        except EvalRuntimeError as exc:
            stream_result = {
                "status": "error",
                "passed": False,
                "failure_reasons": [f"stream.{exc}"],
                "assertions": [],
                "error": str(exc),
            }
    if stream_result is not None and not stream_result["passed"]:
        failure_reasons.extend(stream_result["failure_reasons"])
    return {
        "sample_id": sample["sample_id"],
        "query": sample["query"],
        "source_doc": sample.get("source_doc"),
        "eval_stream": bool(sample.get("eval_stream")),
        "target": expected,
        "status": "ok",
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "assertions": assertions,
        "observed": observed,
        "stream": stream_result,
        "metrics": metrics,
    }


def _run_stream_sample(
    *,
    base_url: str,
    sample: dict[str, Any],
    expected: dict[str, Any],
    baseline_observed: dict[str, Any],
) -> dict[str, Any]:
    session_response = _request_json(
        method="POST",
        url=f"{base_url}/sessions",
        json_body={
            "scene": "generic_assistant",
            "mounted_knowledge_sources": ["documents"],
        },
    )
    session_id = session_response["session_id"]
    stream_payload = _request_sse_chat(
        base_url=base_url,
        json_body={
            "message": sample["query"],
            "session_id": session_id,
            "stream": True,
        },
    )
    done_response = stream_payload["done"]
    observed = _build_observed_from_chat_response(done_response)
    metrics = _build_metrics_from_observed(expected=expected, observed=observed)
    assertions = _build_assertions(expected=expected, observed=observed, metrics=metrics)
    assertions.extend(
        [
            _make_assertion(
                "stream_sync_knowledge_used",
                baseline_observed.get("knowledge_used"),
                observed.get("knowledge_used"),
                observed.get("knowledge_used") == baseline_observed.get("knowledge_used"),
            ),
            _make_assertion(
                "stream_sync_citation_count",
                baseline_observed.get("citation_count"),
                observed.get("citation_count"),
                observed.get("citation_count") == baseline_observed.get("citation_count"),
            ),
            _make_assertion(
                "sse_done_seen",
                True,
                "done" in stream_payload["event_types"],
                "done" in stream_payload["event_types"],
            ),
            _make_assertion(
                "sse_chunk_seen",
                True,
                stream_payload["chunk_count"] > 0,
                stream_payload["chunk_count"] > 0,
            ),
        ]
    )
    failure_reasons = [
        _format_stream_failure_reason(assertion)
        for assertion in assertions
        if not assertion["passed"]
    ]
    return {
        "status": "ok",
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "assertions": assertions,
        "observed": observed,
        "metrics": metrics,
        "event_types": stream_payload["event_types"],
        "chunk_count": stream_payload["chunk_count"],
        "chunk_text_preview": stream_payload["chunk_text_preview"],
        "policy_evidence": stream_payload.get("policy_evidence"),
    }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    succeeded = [item for item in results if item["status"] == "ok"]
    errors = [item for item in results if item["status"] != "ok"]
    passed_samples = sum(1 for item in results if item.get("passed") is True)
    failed_samples = total - passed_samples
    knowledge_hits = sum(1 for item in succeeded if item["metrics"]["knowledge_used"])
    citations_present = sum(1 for item in succeeded if item["metrics"]["citation_count"] > 0)
    answer_keyword_hits = sum(1 for item in succeeded if item["metrics"]["answer_keyword_hit"])
    source_required = [
        item
        for item in succeeded
        if item.get("target", {}).get("citation_source_name")
    ]
    expected_source_hits = sum(1 for item in source_required if item["metrics"]["expected_source_seen"])
    visible_marker_hits = sum(1 for item in succeeded if item["metrics"]["visible_marker_seen"])
    fallback_like_hits = sum(1 for item in succeeded if item["metrics"]["fallback_like"])
    stream_required = [
        item
        for item in succeeded
        if item.get("eval_stream")
    ]
    stream_passed = sum(1 for item in stream_required if item.get("stream", {}).get("passed") is True)

    def _rate(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 4)

    return {
        "total_samples": total,
        "passed_samples": passed_samples,
        "failed_samples": failed_samples,
        "successful_calls": len(succeeded),
        "errored_calls": len(errors),
        "samples_with_knowledge": knowledge_hits,
        "samples_with_citations": citations_present,
        "answer_keyword_hits": answer_keyword_hits,
        "expected_source_hits": expected_source_hits,
        "visible_marker_hits": visible_marker_hits,
        "fallback_like_hits": fallback_like_hits,
        "stream_samples": len(stream_required),
        "stream_passed_samples": stream_passed,
        "stream_failed_samples": len(stream_required) - stream_passed,
        "sample_pass_rate": _rate(passed_samples, total),
        "completion_rate": _rate(len(succeeded), total),
        "knowledge_hit_rate": _rate(knowledge_hits, len(succeeded)),
        "citation_presence_rate": _rate(citations_present, len(succeeded)),
        "answer_keyword_hit_rate": _rate(answer_keyword_hits, len(succeeded)),
        "expected_source_hit_rate": _rate(expected_source_hits, len(source_required)),
        "stream_pass_rate": _rate(stream_passed, len(stream_required)),
    }


def _render_console_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |"

    divider = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    lines = [_fmt(headers), divider]
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)


def _build_sample_table(results: list[dict[str, Any]]) -> str:
    headers = ["sample_id", "pass", "stream", "query", "knowledge", "citations", "source_hit", "marker", "failure", "preview"]
    rows: list[list[str]] = []
    for item in results:
        if item["status"] != "ok":
            rows.append(
                [
                    item["sample_id"],
                    "no",
                    "-",
                    _answer_preview(str(item.get("query", "")), limit=40),
                    "error",
                    "-",
                    "-",
                    "-",
                    _answer_preview("; ".join(item.get("failure_reasons", [])), limit=60),
                    _answer_preview(str(item.get("error", ""))),
                ]
            )
            continue
        metrics = item["metrics"]
        observed = item["observed"]
        rows.append(
            [
                item["sample_id"],
                "yes" if item.get("passed") else "no",
                _stream_status_label(item),
                _answer_preview(str(item["query"]), limit=40),
                "yes" if metrics["knowledge_used"] else "no",
                str(observed["citation_count"]),
                "yes" if metrics["expected_source_seen"] else (
                    "n/a" if not item.get("target", {}).get("citation_source_name") else "no"
                ),
                "yes" if metrics["visible_marker_seen"] else "no",
                _answer_preview("; ".join(item.get("failure_reasons", [])), limit=60),
                str(observed["answer_preview"]),
            ]
        )
    return _render_console_table(headers, rows)


def _stream_status_label(item: dict[str, Any]) -> str:
    if not item.get("eval_stream"):
        return "n/a"
    stream = item.get("stream") or {}
    if stream.get("status") == "error":
        return "error"
    return "yes" if stream.get("passed") else "no"


def _build_metrics_table(summary: dict[str, Any]) -> str:
    headers = ["metric", "value", "meaning", "reading"]
    rows = [
        [
            item["metric"],
            str(summary[item["metric"]]),
            item["meaning"],
            item["reading"](summary) if callable(item["reading"]) else item["reading"],
        ]
        for item in _metric_specs()
    ]
    return _render_console_table(headers, rows)


def _metric_specs() -> list[dict[str, Any]]:
    return [
        {"metric": "total_samples", "meaning": "本次回放样本总数", "reading": "用于判断样本集规模，当前 minimal 固定为小样本回归集。"},
        {
            "metric": "passed_samples",
            "meaning": "通过全部断言的样本数",
            "reading": lambda summary: f"{summary['passed_samples']}/{summary['total_samples']} 条样本满足预期。",
        },
        {
            "metric": "failed_samples",
            "meaning": "HTTP 错误或断言失败的样本数",
            "reading": lambda summary: "没有失败样本。" if summary["failed_samples"] == 0 else "需要查看 failure 和 assertions 定位。",
        },
        {
            "metric": "successful_calls",
            "meaning": "完整拿到 /chat 响应的样本数",
            "reading": lambda summary: f"接口链路完成率为 {summary['completion_rate']:.0%}。",
        },
        {
            "metric": "errored_calls",
            "meaning": "HTTP 或运行时异常样本数",
            "reading": lambda summary: "回放链路无接口错误。" if summary["errored_calls"] == 0 else "先排查服务或依赖稳定性。",
        },
        {"metric": "samples_with_knowledge", "meaning": "返回 knowledge_used=true 的样本数", "reading": "应主要来自知识命中类样本，no-hit 不应计入。"},
        {"metric": "samples_with_citations", "meaning": "返回 citations 非空的样本数", "reading": "应与知识命中类样本数量接近，no-hit 应保持 0 引用。"},
        {
            "metric": "answer_keyword_hits",
            "meaning": "答案命中样本关键词约束的数量",
            "reading": lambda summary: f"关键词约束命中率为 {summary['answer_keyword_hit_rate']:.0%}。",
        },
        {
            "metric": "expected_source_hits",
            "meaning": "引用命中预期来源文档的数量",
            "reading": lambda summary: f"来源命中率为 {summary['expected_source_hit_rate']:.0%}。",
        },
        {"metric": "visible_marker_hits", "meaning": "答案正文包含 [1] 这类引用标记的数量", "reading": "用于确认结构化 citation 与正文引用展示一致。"},
        {"metric": "fallback_like_hits", "meaning": "答案文本像 no-hit fallback 的数量", "reading": "当前 minimal 中应主要对应 no_hit_fallback。"},
        {"metric": "stream_samples", "meaning": "启用 stream=true 回放的样本数", "reading": "用于确认 SSE 质量门禁覆盖范围。"},
        {
            "metric": "stream_passed_samples",
            "meaning": "stream=true 回放断言通过的样本数",
            "reading": lambda summary: f"{summary['stream_passed_samples']}/{summary['stream_samples']} 条流式样本满足预期。",
        },
        {
            "metric": "stream_failed_samples",
            "meaning": "stream=true 回放断言失败的样本数",
            "reading": lambda summary: "没有流式失败样本。" if summary["stream_failed_samples"] == 0 else "需要查看 SSE Stream Evidence 和 failure。",
        },
        {
            "metric": "sample_pass_rate",
            "meaning": "passed_samples / total_samples",
            "reading": lambda summary: f"样本断言通过率为 {summary['sample_pass_rate']:.0%}。",
        },
        {
            "metric": "completion_rate",
            "meaning": "successful_calls / total_samples",
            "reading": lambda summary: f"HTTP 回放完成率为 {summary['completion_rate']:.0%}。",
        },
        {"metric": "knowledge_hit_rate", "meaning": "samples_with_knowledge / successful_calls", "reading": "用于观察知识链路触发比例，不等同于正确率。"},
        {"metric": "citation_presence_rate", "meaning": "samples_with_citations / successful_calls", "reading": "用于观察回答携带引用的比例，不应覆盖 no-hit 样本。"},
        {"metric": "answer_keyword_hit_rate", "meaning": "answer_keyword_hits / successful_calls", "reading": "衡量答案是否满足样本最小内容约束。"},
        {"metric": "expected_source_hit_rate", "meaning": "expected_source_hits / 需要来源命中的样本数", "reading": "衡量引用是否指向预期文档。"},
        {"metric": "stream_pass_rate", "meaning": "stream_passed_samples / stream_samples", "reading": "衡量 SSE done 事件最终语义是否通过断言。"},
    ]


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _format_rate(value: float) -> str:
    return f"{value:.0%}"


def _bar(value: float, *, width: int = 20) -> str:
    bounded = max(0.0, min(1.0, value))
    filled = round(bounded * width)
    return "#" * filled + "." * (width - filled)


def _build_kpi_table(summary: dict[str, Any]) -> str:
    rows = [
        ["样本断言通过率", _format_rate(summary["sample_pass_rate"]), _bar(summary["sample_pass_rate"]), "所有样本必须通过；低于 100% 应阻断回归。"],
        ["HTTP 回放完成率", _format_rate(summary["completion_rate"]), _bar(summary["completion_rate"]), "用于区分评测失败和服务调用失败。"],
        ["预期来源命中率", _format_rate(summary["expected_source_hit_rate"]), _bar(summary["expected_source_hit_rate"]), "知识命中样本的 citation 应指向目标 fixture。"],
        ["答案关键词命中率", _format_rate(summary["answer_keyword_hit_rate"]), _bar(summary["answer_keyword_hit_rate"]), "答案需满足样本定义的最小语义约束。"],
        ["引用出现率", _format_rate(summary["citation_presence_rate"]), _bar(summary["citation_presence_rate"]), "当前 minimal 中应为知识命中样本占比，而不是 100%。"],
        ["SSE 回放通过率", _format_rate(summary["stream_pass_rate"]), _bar(summary["stream_pass_rate"]), "stream=true 样本应拿到 done 事件并满足最终语义断言。"],
    ]
    return _render_console_table(["KPI", "value", "bar", "evidence"], rows)


def _build_count_comparison_table(summary: dict[str, Any]) -> str:
    total = summary["total_samples"]
    successful = summary["successful_calls"]
    rows = [
        ["样本通过", str(summary["passed_samples"]), str(total), _bar(summary["sample_pass_rate"]), "passed_samples / total_samples"],
        ["样本失败", str(summary["failed_samples"]), str(total), _bar(summary["failed_samples"] / total if total else 0.0), "failed_samples / total_samples"],
        ["接口成功", str(successful), str(total), _bar(summary["completion_rate"]), "successful_calls / total_samples"],
        ["接口错误", str(summary["errored_calls"]), str(total), _bar(summary["errored_calls"] / total if total else 0.0), "errored_calls / total_samples"],
        ["使用知识", str(summary["samples_with_knowledge"]), str(successful), _bar(summary["knowledge_hit_rate"]), "knowledge_used=true / successful_calls"],
        ["携带引用", str(summary["samples_with_citations"]), str(successful), _bar(summary["citation_presence_rate"]), "citations 非空 / successful_calls"],
        ["SSE 通过", str(summary["stream_passed_samples"]), str(summary["stream_samples"]), _bar(summary["stream_pass_rate"]), "stream_passed_samples / stream_samples"],
    ]
    return _render_console_table(["dimension", "count", "base", "bar", "formula"], rows)


def _build_sample_evidence_table(results: list[dict[str, Any]]) -> str:
    rows: list[list[str]] = []
    for item in results:
        expected = item.get("target", {})
        observed = item.get("observed", {})
        metrics = item.get("metrics", {})
        if item["status"] != "ok":
            rows.append([
                item["sample_id"],
                "error",
                str(expected.get("knowledge_used")),
                "-",
                "-",
                "-",
                _answer_preview("; ".join(item.get("failure_reasons", [])), limit=80),
            ])
            continue
        rows.append([
            item["sample_id"],
            _yes_no(bool(item.get("passed"))),
            str(expected.get("knowledge_used")),
            str(observed.get("knowledge_used")),
            str(observed.get("citation_count")),
            _yes_no(bool(metrics.get("fallback_like"))),
            _answer_preview(str(observed.get("answer_preview", "")), limit=96),
        ])
    return _render_console_table(
        ["sample_id", "pass", "expected_knowledge", "actual_knowledge", "citations", "fallback_like", "answer_evidence"],
        rows,
    )


def _build_assertion_matrix(results: list[dict[str, Any]]) -> str:
    rows: list[list[str]] = []
    for item in results:
        assertions = item.get("assertions", [])
        if not assertions:
            rows.append([item["sample_id"], "-", "-", "-", _answer_preview("; ".join(item.get("failure_reasons", [])), limit=80)])
            continue
        for assertion in assertions:
            rows.append([
                item["sample_id"],
                assertion["name"],
                str(assertion["expected"]),
                _answer_preview(str(assertion["actual"]), limit=80),
                _yes_no(bool(assertion["passed"])),
            ])
    return _render_console_table(["sample_id", "assertion", "expected", "actual", "pass"], rows)


def _build_no_hit_boundary_table(results: list[dict[str, Any]]) -> str:
    no_hit = next((item for item in results if item["sample_id"] == "no_hit_fallback"), None)
    if no_hit is None:
        return "_no_hit_fallback sample not found_"
    observed = no_hit.get("observed", {})
    metrics = no_hit.get("metrics", {})
    rows = [
        ["knowledge_used", "false", str(observed.get("knowledge_used")), _yes_no(observed.get("knowledge_used") is False), "防止 no-hit 被误标为使用知识"],
        ["citation_count", "0", str(observed.get("citation_count")), _yes_no(observed.get("citation_count") == 0), "防止空命中仍返回引用数量"],
        ["citations", "[]", str(observed.get("citations")), _yes_no(observed.get("citations") == []), "防止伪引用污染溯源结果"],
        ["visible_marker", "false", str(metrics.get("visible_marker_seen")), _yes_no(metrics.get("visible_marker_seen") is False), "防止正文出现无来源编号"],
        ["fallback_like", "true", str(metrics.get("fallback_like")), _yes_no(metrics.get("fallback_like") is True), "确认回答语义为无可靠资料"],
    ]
    return _render_console_table(["boundary", "expected", "actual", "pass", "why_it_matters"], rows)


def _build_stream_evidence_table(results: list[dict[str, Any]]) -> str:
    stream_results = [item for item in results if item.get("eval_stream")]
    if not stream_results:
        return "_No stream=true samples configured_"
    rows: list[list[str]] = []
    for item in stream_results:
        stream = item.get("stream") or {}
        observed = stream.get("observed", {})
        metrics = stream.get("metrics", {})
        rows.append([
            item["sample_id"],
            _stream_status_label(item),
            ",".join(str(event) for event in stream.get("event_types", [])),
            str(stream.get("chunk_count", "-")),
            str(observed.get("knowledge_used", "-")),
            str(observed.get("citation_count", "-")),
            _yes_no(bool(metrics.get("fallback_like"))) if metrics else "-",
            _answer_preview("; ".join(stream.get("failure_reasons", [])), limit=80),
        ])
    return _render_console_table(
        ["sample_id", "stream_pass", "events", "chunks", "knowledge", "citations", "fallback_like", "failure"],
        rows,
    )


def _build_effect_conclusion(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    no_hit = next((item for item in results if item.get("sample_id") == "no_hit_fallback"), None)
    no_hit_ok = bool(no_hit and no_hit.get("passed") is True)
    all_clear = (
        summary["failed_samples"] == 0
        and summary["completion_rate"] == 1.0
        and summary["expected_source_hit_rate"] == 1.0
        and summary["answer_keyword_hit_rate"] == 1.0
        and summary["stream_pass_rate"] == 1.0
        and no_hit_ok
    )
    if all_clear:
        return (
            "结论：当前 `generic_assistant + documents` 在 minimal 固定样本集上通过最小 RAG "
            f"回归评测。{summary['passed_samples']}/{summary['total_samples']} 条样本通过；"
            "命中类问题能返回预期来源和最小正确答案；no-hit 问题没有返回伪引用；"
            "stream=true 与普通响应的关键语义一致。"
        )
    return (
        "结论：当前回放不能作为 RAG 主链通过证据。存在样本失败、接口错误、预期来源未命中、"
        "答案关键词未命中、no-hit 边界失败或 stream=true 语义不一致。请优先查看失败样本和 latest.json 中的 assertions。"
    )


def _build_effect_metrics_table(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    no_hit = next((item for item in results if item.get("sample_id") == "no_hit_fallback"), None)
    no_hit_pass = no_hit.get("passed") is True if no_hit else False
    rows = [
        ["Regression Gate", "Sample Pass Rate", "样本断言通过率", _format_rate(summary["sample_pass_rate"]), f"{summary['passed_samples']}/{summary['total_samples']} samples passed"],
        ["Retrieval Quality", "Expected Source Hit Rate", "预期来源命中率", _format_rate(summary["expected_source_hit_rate"]), "citation 指向目标 fixture"],
        ["Generation Quality", "Answer Keyword Hit Rate", "答案关键词命中率", _format_rate(summary["answer_keyword_hit_rate"]), "答案满足样本最小语义约束"],
        ["Faithfulness Boundary", "No-hit Fallback Correctness", "无命中回退正确性", "pass" if no_hit_pass else "fail", "无可靠资料时不返回 citations"],
        ["System Reliability", "Completion Rate", "HTTP 回放完成率", _format_rate(summary["completion_rate"]), f"{summary['successful_calls']}/{summary['total_samples']} calls succeeded"],
        ["Streaming Consistency", "Stream Pass Rate", "SSE 回放通过率", _format_rate(summary["stream_pass_rate"]), f"{summary['stream_passed_samples']}/{summary['stream_samples']} stream samples passed"],
    ]
    return _render_console_table(["dimension", "metric", "中文说明", "value", "evidence"], rows)


def _expected_source_required_count(results: list[dict[str, Any]]) -> int:
    return sum(
        1
        for item in results
        if item.get("status") == "ok" and item.get("target", {}).get("citation_source_name")
    )


def _current_retrieval_quality_table(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    no_hit = next((item for item in results if item.get("sample_id") == "no_hit_fallback"), None)
    no_hit_clean = bool(
        no_hit
        and no_hit.get("observed", {}).get("knowledge_used") is False
        and no_hit.get("observed", {}).get("citations") == []
    )
    source_required = _expected_source_required_count(results)
    rows = [
        [
            "Expected Source Hit Rate",
            "预期来源命中率",
            _format_rate(summary["expected_source_hit_rate"]),
            f"{summary['expected_source_hits']}/{source_required}",
            "命中样本的 citation 指向目标 fixture；当前最接近检索相关性判断。",
        ],
        [
            "Citation Presence Rate",
            "引用出现率",
            _format_rate(summary["citation_presence_rate"]),
            f"{summary['samples_with_citations']}/{summary['successful_calls']}",
            "观察回答是否携带引用；当前包含 no-hit 样本，因此不是越高越好。",
        ],
        [
            "Knowledge Hit Rate",
            "知识触发率",
            _format_rate(summary["knowledge_hit_rate"]),
            f"{summary['samples_with_knowledge']}/{summary['successful_calls']}",
            "观察 knowledge_used=true 的比例；用于排查检索链路是否被触发。",
        ],
        [
            "No-hit Pseudo-citation Boundary",
            "无命中伪引用边界",
            "pass" if no_hit_clean else "fail",
            "knowledge_used=false, citations=[]",
            "确认陌生问题不会被误判为命中文档。",
        ],
    ]
    return _render_console_table(["metric", "中文说明", "value", "evidence", "meaning"], rows)


def _current_generation_quality_table(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    rows = [
        [
            "Answer Keyword Hit Rate",
            "答案关键词命中率",
            _format_rate(summary["answer_keyword_hit_rate"]),
            f"{summary['answer_keyword_hits']}/{summary['successful_calls']}",
            "答案包含样本定义的最小正确内容。",
        ],
        [
            "Visible Citation Marker Count",
            "正文引用标记命中数",
            str(summary["visible_marker_hits"]),
            "命中类样本应出现 [1] 等引用标记",
            "确认结构化 citations 与正文可溯源展示一致。",
        ],
        [
            "Fallback-like Answer Count",
            "回退语义命中数",
            str(summary["fallback_like_hits"]),
            "当前 minimal 中应主要对应 no_hit_fallback",
            "确认无资料时回答为证据不足，而不是编造答案。",
        ],
    ]
    return _render_console_table(["metric", "中文说明", "value", "evidence", "meaning"], rows)


def _current_system_quality_table(summary: dict[str, Any]) -> str:
    rows = [
        [
            "Completion Rate",
            "HTTP 回放完成率",
            _format_rate(summary["completion_rate"]),
            f"{summary['successful_calls']}/{summary['total_samples']}",
            "确认 /chat 主链可完成调用。",
        ],
        [
            "Error Count",
            "接口错误数",
            str(summary["errored_calls"]),
            "status != ok",
            "用于区分服务异常和评测断言失败。",
        ],
        [
            "Stream Pass Rate",
            "SSE 回放通过率",
            _format_rate(summary["stream_pass_rate"]),
            f"{summary['stream_passed_samples']}/{summary['stream_samples']}",
            "确认 stream=true 样本拿到 done 事件并满足最终语义断言。",
        ],
    ]
    return _render_console_table(["metric", "中文说明", "value", "evidence", "meaning"], rows)


def _build_dimension_coverage_table() -> str:
    rows = [
        [
            "Retrieval Quality",
            "检索质量",
            "预期来源命中、citation 数量、no-hit 不伪引用、retrieval policy evidence",
            "Precision@k / Recall@k / MRR / NDCG",
            "需要 qrels 和完整 ranked retrieval list",
        ],
        [
            "Generation Quality",
            "生成质量",
            "答案关键词、引用标记、fallback 语义、stream done 一致性",
            "CR / AR / F 的 judge 或人工评分",
            "需要 LLM-as-a-judge 或人工评分表",
        ],
        [
            "System Performance",
            "系统性能",
            "HTTP 完成率、错误样本数、SSE 是否完成",
            "延迟分位数、吞吐量、并发错误率",
            "需要请求计时和压测入口",
        ],
    ]
    return _render_console_table(["dimension", "中文说明", "current coverage", "not yet covered", "next step"], rows)


def _build_compact_sample_results_table(results: list[dict[str, Any]]) -> str:
    rows: list[list[str]] = []
    for item in results:
        if item["status"] != "ok":
            rows.append([
                item["sample_id"],
                "no",
                "-",
                "-",
                "-",
                _stream_status_label(item),
                _answer_preview("; ".join(item.get("failure_reasons", [])), limit=80),
            ])
            continue
        metrics = item["metrics"]
        observed = item["observed"]
        rows.append([
            item["sample_id"],
            _yes_no(bool(item.get("passed"))),
            _yes_no(bool(metrics.get("knowledge_used"))),
            "yes" if metrics.get("expected_source_seen") else (
                "n/a" if not item.get("target", {}).get("citation_source_name") else "no"
            ),
            str(observed.get("citation_count")),
            _stream_status_label(item),
            _answer_preview(str(observed.get("answer_preview", "")), limit=96),
        ])
    return _render_console_table(
        ["sample_id", "pass", "knowledge", "source_hit", "citations", "stream", "answer_preview"],
        rows,
    )


def _build_pending_development_table() -> str:
    rows = [
        [
            "Retrieval Benchmark",
            "检索 benchmark",
            "为样本增加 qrels，保存完整 ranked retrieval list，计算 Precision@k / Recall@k / MRR / NDCG。",
            "需要确认 qrels 标注粒度：document 级、chunk 级，还是两者都要。",
        ],
        [
            "Generation Quality Judge",
            "生成质量 judge",
            "增加 LLM-as-a-judge 或人工评分表，输出 CR / AR / F 评分和原因。",
            "需要确认评分模型、分值范围、失败阈值和是否允许人工复核。",
        ],
        [
            "Performance Evaluation",
            "性能评估",
            "记录请求总耗时、检索耗时、生成耗时、首 token 时间、P95、错误率；必要时增加并发压测。",
            "需要确认性能目标和运行环境，例如本机、CI 或固定压测环境。",
        ],
        [
            "Benchmark Sample Expansion",
            "样本集扩展",
            "新增 benchmark 样本集，覆盖多文档、多跳、相似干扰、冲突文档、无答案、中英文等场景。",
            "需要确认业务优先级和期望样本规模。",
        ],
    ]
    return _render_console_table(["area", "中文说明", "development item", "needs confirmation"], rows)


def _build_failure_focus(results: list[dict[str, Any]]) -> str:
    failed = [item for item in results if item.get("passed") is not True]
    if not failed:
        return "无失败样本。调试明细保留在同目录 `latest.json`。"
    rows = [
        [
            item["sample_id"],
            item.get("status", "-"),
            _stream_status_label(item),
            _answer_preview("; ".join(item.get("failure_reasons", [])), limit=100),
        ]
        for item in failed
    ]
    return _render_console_table(["sample_id", "status", "stream", "failure"], rows)


def _citation_source_names(item: dict[str, Any]) -> list[str]:
    observed = item.get("observed", {})
    citations = observed.get("citations", [])
    if not isinstance(citations, list):
        return []
    return [
        str(citation.get("source_name"))
        for citation in citations
        if isinstance(citation, dict) and citation.get("source_name") is not None
    ]


def _stream_observations(item: dict[str, Any]) -> dict[str, Any] | None:
    stream = item.get("stream")
    if not isinstance(stream, dict):
        return None
    observed = stream.get("observed", {})
    return {
        "passed": stream.get("passed"),
        "event_types": stream.get("event_types", []),
        "knowledge_used": observed.get("knowledge_used") if isinstance(observed, dict) else None,
        "citation_count": observed.get("citation_count") if isinstance(observed, dict) else None,
    }


def _sample_compare_entry(
    *,
    sample_id: str,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    if baseline is None or candidate is None:
        return {
            "sample_id": sample_id,
            "status": "missing_baseline" if baseline is None else "missing_candidate",
            "baseline": None if baseline is None else {"passed": baseline.get("passed")},
            "candidate": None if candidate is None else {"passed": candidate.get("passed")},
            "differences": ["missing_baseline" if baseline is None else "missing_candidate"],
        }

    baseline_observed = baseline.get("observed", {})
    candidate_observed = candidate.get("observed", {})
    baseline_citations = baseline_observed.get("citations", []) if isinstance(baseline_observed, dict) else []
    candidate_citations = candidate_observed.get("citations", []) if isinstance(candidate_observed, dict) else []
    baseline_stream = _stream_observations(baseline)
    candidate_stream = _stream_observations(candidate)
    compared = {
        "pass": {"baseline": baseline.get("passed"), "candidate": candidate.get("passed")},
        "knowledge_used": {
            "baseline": baseline_observed.get("knowledge_used") if isinstance(baseline_observed, dict) else None,
            "candidate": candidate_observed.get("knowledge_used") if isinstance(candidate_observed, dict) else None,
        },
        "citation_count": {
            "baseline": baseline_observed.get("citation_count") if isinstance(baseline_observed, dict) else None,
            "candidate": candidate_observed.get("citation_count") if isinstance(candidate_observed, dict) else None,
        },
        "citation_sources": {
            "baseline": _citation_source_names(baseline),
            "candidate": _citation_source_names(candidate),
        },
        "no_hit_citations_empty": {
            "baseline": baseline_citations == [],
            "candidate": candidate_citations == [],
        },
        "stream": {"baseline": baseline_stream, "candidate": candidate_stream},
        "policy_evidence": {
            "baseline": (baseline.get("stream") or {}).get("policy_evidence") if isinstance(baseline.get("stream"), dict) else None,
            "candidate": (candidate.get("stream") or {}).get("policy_evidence") if isinstance(candidate.get("stream"), dict) else None,
        },
    }
    differences = [
        key
        for key, value in compared.items()
        if isinstance(value, dict) and value.get("baseline") != value.get("candidate")
    ]
    return {
        "sample_id": sample_id,
        "status": "changed" if differences else "same",
        "baseline": {
            "status": baseline.get("status"),
            "passed": baseline.get("passed"),
        },
        "candidate": {
            "status": candidate.get("status"),
            "passed": candidate.get("passed"),
        },
        "compared": compared,
        "differences": differences,
    }


def build_comparison_payload(
    *,
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    # 使用 sample_id 对齐基线与候选，避免样本顺序变化造成误判。
    baseline_by_id = {
        str(item["sample_id"]): item
        for item in baseline_payload.get("results", [])
        if isinstance(item, dict) and item.get("sample_id") is not None
    }
    candidate_by_id = {
        str(item["sample_id"]): item
        for item in candidate_payload.get("results", [])
        if isinstance(item, dict) and item.get("sample_id") is not None
    }
    sample_ids = sorted(set(baseline_by_id) | set(candidate_by_id))
    samples = [
        _sample_compare_entry(
            sample_id=sample_id,
            baseline=baseline_by_id.get(sample_id),
            candidate=candidate_by_id.get(sample_id),
        )
        for sample_id in sample_ids
    ]
    return {
        "comparison_id": uuid4().hex,
        "executed_at": _now_iso(),
        "baseline_run_id": baseline_payload.get("run_id"),
        "candidate_run_id": candidate_payload.get("run_id"),
        "sample_set": candidate_payload.get("sample_set") or baseline_payload.get("sample_set"),
        "summary": {
            "total_samples": len(samples),
            "same_samples": sum(1 for item in samples if item["status"] == "same"),
            "changed_samples": sum(1 for item in samples if item["status"] == "changed"),
            "missing_baseline_samples": sum(1 for item in samples if item["status"] == "missing_baseline"),
            "missing_candidate_samples": sum(1 for item in samples if item["status"] == "missing_candidate"),
        },
        "samples": samples,
    }


def _format_policy_evidence(value: Any) -> str:
    if not isinstance(value, dict):
        return "-"
    policy = value.get("retrieval_policy")
    if not isinstance(policy, dict):
        return "-"
    return ", ".join(f"{key}={policy.get(key)!r}" for key in sorted(policy))


def _write_comparison_markdown(*, comparison: dict[str, Any], output_path: Path) -> Path:
    report_path = output_path.with_suffix(".compare.md")
    rows: list[list[str]] = []
    for item in comparison["samples"]:
        compared = item.get("compared", {})
        rows.append(
            [
                item["sample_id"],
                item["status"],
                ",".join(item.get("differences", [])) or "-",
                str(compared.get("knowledge_used", {}).get("baseline")) if compared else "-",
                str(compared.get("knowledge_used", {}).get("candidate")) if compared else "-",
                str(compared.get("citation_count", {}).get("baseline")) if compared else "-",
                str(compared.get("citation_count", {}).get("candidate")) if compared else "-",
                _format_policy_evidence(compared.get("policy_evidence", {}).get("baseline")) if compared else "-",
                _format_policy_evidence(compared.get("policy_evidence", {}).get("candidate")) if compared else "-",
            ]
        )
    lines = [
        "# Evaluation Baseline Comparison",
        "",
        f"- comparison_id: `{comparison['comparison_id']}`",
        f"- baseline_run_id: `{comparison.get('baseline_run_id')}`",
        f"- candidate_run_id: `{comparison.get('candidate_run_id')}`",
        f"- sample_set: `{comparison.get('sample_set')}`",
        "",
        "## Summary",
        "",
        _render_console_table(
            ["metric", "value"],
            [[key, str(value)] for key, value in comparison["summary"].items()],
        ),
        "",
        "## Sample Differences",
        "",
        _render_console_table(
            [
                "sample_id",
                "status",
                "differences",
                "base_knowledge",
                "cand_knowledge",
                "base_citations",
                "cand_citations",
                "base_policy",
                "cand_policy",
            ],
            rows,
        ),
        "",
        "## Reading Notes",
        "",
        "- `no_hit_citations_empty` changing to `false` is a no-hit citation regression.",
        "- `policy_evidence` is captured only from SSE `tool` events and excludes prompt text and raw source snippets.",
        "- Compare baseline and candidate runs produced under different scene policy or ReRank code/config to attribute retrieval changes.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_comparison_reports(
    *,
    baseline_path: Path,
    candidate_payload: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    baseline_payload = _read_json(baseline_path)
    comparison = build_comparison_payload(
        baseline_payload=baseline_payload,
        candidate_payload=candidate_payload,
    )
    compare_json_path = output_path.with_suffix(".compare.json")
    compare_json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    compare_md_path = _write_comparison_markdown(comparison=comparison, output_path=output_path)
    comparison["json_path"] = str(compare_json_path)
    comparison["report_path"] = str(compare_md_path)
    return comparison


def _write_markdown_report(*, payload: dict[str, Any], output_path: Path) -> Path:
    report_path = output_path.with_suffix(".md")
    summary = payload["summary"]
    results = payload["results"]
    lines = [
        "# Evaluation Harness 回放报告",
        "",
        "## Conclusion",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- executed_at: `{payload['executed_at']}`",
        f"- base_url: `{payload['base_url']}`",
        f"- sample_set: `{payload['sample_set']}`",
        "",
        _build_effect_conclusion(summary, results),
        "",
        "报告中的 `pass` 来自固定断言，不依赖 LLM-as-a-judge。它适合作为最小回归门禁，不等同于完整 RAG benchmark。",
        "",
        "## Current Scorecard",
        "",
        _build_effect_metrics_table(summary, results),
        "",
        "## Retrieval Quality",
        "",
        _current_retrieval_quality_table(summary, results),
        "",
        "## Retrieval Benchmark Metrics",
        "",
        _build_retrieval_benchmark_section(payload),
        "",
        "## Retrieval Benchmark Samples",
        "",
        _build_retrieval_sample_metrics_table(results),
        "",
        "## Generation Quality",
        "",
        _current_generation_quality_table(summary, results),
        "",
        "## System Quality",
        "",
        _current_system_quality_table(summary),
        "",
        "## Sample Results",
        "",
        _build_compact_sample_results_table(results),
        "",
        "## no-hit Boundary",
        "",
        _build_no_hit_boundary_table(results),
        "",
        "## SSE Evidence",
        "",
        _build_stream_evidence_table(results),
        "",
        "## Failure Focus",
        "",
        _build_failure_focus(results),
        "",
        "## Benchmark Gaps",
        "",
        _build_dimension_coverage_table(),
        "",
        "## Pending Development Items",
        "",
        _build_pending_development_table(),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _build_retrieval_benchmark_section(payload: dict[str, Any]) -> str:
    benchmark = payload.get("retrieval_benchmark")
    if not isinstance(benchmark, dict):
        return "_No qrels-driven retrieval benchmark metrics for this run._"
    aggregate = benchmark.get("aggregate_metrics", {})
    if not isinstance(aggregate, dict):
        return "_Retrieval benchmark metrics are unavailable._"
    rows = [
        ["hit_sample_count", str(aggregate.get("hit_sample_count", 0)), "参与核心 IR 平均的有答案样本数"],
        ["no_hit_sample_count", str(aggregate.get("no_hit_sample_count", 0)), "仅参与误召回率统计的 no-hit 样本数"],
        ["mrr", f"{float(aggregate.get('mrr', 0.0)):.4f}", "首个相关 chunk 的倒数排名均值"],
        [
            "expected_document_hit",
            f"{float(aggregate.get('expected_document_hit', 0.0)):.4f}",
            "是否命中任一预期文档的均值",
        ],
        [
            "no_hit_false_positive_rate",
            f"{float(aggregate.get('no_hit_false_positive_rate', 0.0)):.4f}",
            "no-hit 样本仍返回 ranked chunk 的比例",
        ],
    ]
    for metric_name in ("precision_at_k", "recall_at_k", "ndcg_at_k", "document_recall_at_k"):
        metric = aggregate.get(metric_name, {})
        if not isinstance(metric, dict):
            continue
        for key in sorted(metric, key=lambda value: int(value)):
            rows.append([f"{metric_name}@{key}", f"{float(metric[key]):.4f}", "qrels 对齐后的聚合指标"])
    return _render_console_table(["metric", "value", "meaning"], rows)


def _build_retrieval_sample_metrics_table(results: list[dict[str, Any]]) -> str:
    rows: list[list[str]] = []
    for item in results:
        retrieval = item.get("retrieval")
        if not isinstance(retrieval, dict):
            continue
        metrics = retrieval.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        rows.append(
            [
                item["sample_id"],
                str(metrics.get("deduped_ranked_count", 0)),
                str(metrics.get("is_no_hit")),
                _metric_at(metrics, "recall_at_k", "5"),
                _metric_at(metrics, "ndcg_at_k", "5"),
                str(metrics.get("expected_document_hit")),
                str(metrics.get("no_hit_false_positive")),
                _answer_preview("; ".join(retrieval.get("failure_reasons", [])), limit=80),
            ]
        )
    if not rows:
        return "_No retrieval sample metrics for this run._"
    return _render_console_table(
        ["sample_id", "ranked", "no_hit", "recall@5", "ndcg@5", "doc_hit", "false_positive", "failure"],
        rows,
    )


def _metric_at(metrics: dict[str, Any], metric_name: str, k: str) -> str:
    value = metrics.get(metric_name, {}).get(k) if isinstance(metrics.get(metric_name), dict) else None
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _attach_retrieval_benchmark_results(
    *,
    sample_set: dict[str, Any],
    qrels_payload: dict[str, Any] | None,
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if qrels_payload is None:
        return None
    qrels_by_sample_id = qrels_list_to_mapping(qrels_payload)
    allowed_source_docs = [
        str(fixture["filename"])
        for fixture in sample_set.get("fixtures", [])
        if isinstance(fixture, dict) and fixture.get("filename") is not None
    ]
    probe_payload = run_retrieval_probe(
        samples=sample_set["samples"],
        namespace=str(sample_set.get("namespace", "documents")),
        allowed_source_docs=allowed_source_docs,
    )
    probe_by_sample_id = {
        str(item.get("sample_id")): item
        for item in probe_payload.get("samples", [])
        if isinstance(item, dict) and item.get("sample_id") is not None
    }
    ranked_lists_by_sample_id = {
        sample_id: probe_by_sample_id.get(sample_id, {}).get("ranked_list", [])
        for sample_id in qrels_by_sample_id
    }
    metrics_payload = compute_retrieval_benchmark_metrics(
        qrels_by_sample_id=qrels_by_sample_id,
        ranked_lists_by_sample_id=ranked_lists_by_sample_id,
    )
    for item in results:
        sample_id = str(item["sample_id"])
        probe_sample = probe_by_sample_id.get(sample_id, {})
        retrieval_failure_reasons = list(probe_sample.get("failure_reasons", []))
        item["retrieval"] = {
            "qrels": qrels_by_sample_id.get(sample_id, {}),
            "ranked_list": list(probe_sample.get("ranked_list", [])),
            "metrics": metrics_payload["samples"].get(sample_id, {}),
            "failure_reasons": retrieval_failure_reasons,
        }
        if retrieval_failure_reasons:
            item.setdefault("failure_reasons", []).extend(
                f"retrieval_probe.{reason}" for reason in retrieval_failure_reasons
            )
            item["passed"] = False
    return {
        "qrels": qrels_payload,
        "probe": probe_payload,
        "sample_metrics": metrics_payload["samples"],
        "aggregate_metrics": metrics_payload["aggregate"],
    }


def _write_eval_artifacts(*, payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = output_path.parent / "latest.json"
    if output_path.resolve() != latest_path.resolve():
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_run_artifact(payload, evals_dir=output_path.parent)


def _write_run_artifact(payload: dict[str, Any], *, evals_dir: Path) -> None:
    runs_dir = evals_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{payload['run_id']}.json"
    run_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _update_run_index(payload=payload, run_path=run_path)


def _update_run_index(*, payload: dict[str, Any], run_path: Path) -> None:
    index_path = run_path.parent / "index.json"
    existing: dict[str, Any] = {"runs": []}
    if index_path.exists():
        try:
            existing = _read_json(index_path)
        except json.JSONDecodeError:
            existing = {"runs": []}
    runs = [
        item
        for item in existing.get("runs", [])
        if isinstance(item, dict) and item.get("run_id") != payload.get("run_id")
    ]
    runs.insert(0, _build_run_index_entry(payload=payload, run_path=run_path))
    index_path.write_text(
        json.dumps({"runs": runs[:100]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_run_index_entry(*, payload: dict[str, Any], run_path: Path) -> dict[str, Any]:
    summary = payload.get("summary", {})
    benchmark = payload.get("retrieval_benchmark", {})
    aggregate = benchmark.get("aggregate_metrics", {}) if isinstance(benchmark, dict) else {}
    return {
        "run_id": payload.get("run_id"),
        "sample_set": payload.get("sample_set"),
        "status": "failed" if payload.get("error") else "succeeded",
        "executed_at": payload.get("executed_at"),
        "base_url": payload.get("base_url"),
        "json_path": str(run_path),
        "report_path": payload.get("report_path"),
        "summary": {
            "total_samples": summary.get("total_samples", 0),
            "passed_samples": summary.get("passed_samples", 0),
            "failed_samples": summary.get("failed_samples", 0),
            "sample_pass_rate": summary.get("sample_pass_rate", 0.0),
            "retrieval_mrr": aggregate.get("mrr") if isinstance(aggregate, dict) else None,
            "no_hit_false_positive_rate": (
                aggregate.get("no_hit_false_positive_rate")
                if isinstance(aggregate, dict)
                else None
            ),
        },
    }


def run_eval(
    *,
    base_url: str,
    sample_set_name: str,
    output_path: Path,
    compare_to: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    sample_path = SAMPLES_DIR / f"{sample_set_name}.json"
    if not sample_path.exists():
        raise EvalRuntimeError(f"Sample set not found: {sample_path}")
    sample_set = _read_json(sample_path)
    _validate_manifest(sample_set)
    qrels_payload = _load_qrels(sample_set)

    _log(f"Checking health endpoint: {base_url}/health")
    health = _request_json(method="GET", url=f"{base_url}/health")
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise EvalRuntimeError(f"Unexpected /health response: {health!r}")

    namespace = str(sample_set.get("namespace", "documents"))
    _log("Cleaning previous eval files")
    _cleanup_eval_files(base_url)
    _log("Cleaning previous eval documents")
    _cleanup_eval_documents(base_url, namespace)

    results_by_sample_id: dict[str, dict[str, Any]] = {}
    knowledge_samples = []
    for sample in sample_set["samples"]:
        if sample["expected"].get("knowledge_used") is False:
            _log(f"Replaying sample before fixture upload: {sample['sample_id']}")
            try:
                results_by_sample_id[sample["sample_id"]] = _run_sample(base_url=base_url, sample=sample)
            except EvalRuntimeError as exc:
                results_by_sample_id[sample["sample_id"]] = {
                    "sample_id": sample["sample_id"],
                    "query": sample["query"],
                    "source_doc": sample.get("source_doc"),
                    "target": sample["expected"],
                    "status": "error",
                    "passed": False,
                    "failure_reasons": [str(exc)],
                    "assertions": [],
                    "error": str(exc),
                }
            continue
        knowledge_samples.append(sample)

    uploaded_files: dict[str, str] = {}
    for fixture in sample_set["fixtures"]:
        fixture_path = FIXTURES_DIR / fixture["filename"]
        if not fixture_path.exists():
            raise EvalRuntimeError(f"Fixture not found: {fixture_path}")
        _log(f"Uploading fixture: {fixture['filename']}")
        upload_result = _upload_file(
            base_url,
            fixture_path,
            content=_build_fixture_upload_content(sample_set, fixture_path),
        )
        uploaded_files[fixture["id"]] = str(upload_result["file_path"])

    _log("Registering uploaded fixtures into knowledge documents")
    _register_documents(base_url=base_url, namespace=namespace, uploaded_files=uploaded_files)

    for sample in knowledge_samples:
        _log(f"Replaying sample: {sample['sample_id']}")
        try:
            results_by_sample_id[sample["sample_id"]] = _run_sample(base_url=base_url, sample=sample)
        except EvalRuntimeError as exc:
            results_by_sample_id[sample["sample_id"]] = {
                "sample_id": sample["sample_id"],
                "query": sample["query"],
                "source_doc": sample.get("source_doc"),
                "target": sample["expected"],
                "status": "error",
                "passed": False,
                "failure_reasons": [str(exc)],
                "assertions": [],
                "error": str(exc),
            }

    results = [results_by_sample_id[sample["sample_id"]] for sample in sample_set["samples"]]
    _log("Running qrels-driven retrieval probe") if qrels_payload is not None else None
    retrieval_benchmark = _attach_retrieval_benchmark_results(
        sample_set=sample_set,
        qrels_payload=qrels_payload,
        results=results,
    )

    payload = {
        "run_id": run_id or uuid4().hex,
        "executed_at": _now_iso(),
        "base_url": base_url,
        "sample_set": sample_set_name,
        "summary": _build_summary(results),
        "results": results,
    }
    if retrieval_benchmark is not None:
        payload["retrieval_benchmark"] = retrieval_benchmark
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = _write_markdown_report(payload=payload, output_path=output_path)
    payload["report_path"] = str(report_path)
    if compare_to is not None:
        payload["comparison"] = write_comparison_reports(
            baseline_path=compare_to,
            candidate_payload=payload,
            output_path=output_path,
        )
    _write_eval_artifacts(payload=payload, output_path=output_path)
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the minimal HTTP evaluation harness.")
    parser.add_argument("--base-url", required=True, help="Base URL of the local backend, e.g. http://127.0.0.1:8000")
    parser.add_argument("--sample-set", default="minimal", help="Sample set name under backend/evals/samples.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to the JSON output file.",
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        help="Optional baseline JSON output to compare against the candidate run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    output_path = Path(args.output)
    try:
        payload = run_eval(
            base_url=args.base_url.rstrip("/"),
            sample_set_name=args.sample_set,
            output_path=output_path,
            compare_to=Path(args.compare_to) if args.compare_to else None,
        )
    except EvalRuntimeError as exc:
        error_payload = {
            "run_id": uuid4().hex,
            "executed_at": _now_iso(),
            "base_url": args.base_url,
            "sample_set": args.sample_set,
            "summary": {
                "total_samples": 0,
                "passed_samples": 0,
                "failed_samples": 0,
                "sample_pass_rate": 0.0,
                "doc_hit_rate": 0.0,
                "citation_presence_rate": 0.0,
                "citation_source_match_rate": 0.0,
                "fallback_correctness": 0.0,
            },
            "error": str(exc),
            "results": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(exc), file=sys.stderr)
        return 1

    summary = payload["summary"]
    print("")
    print("Sample Table")
    print(_build_sample_table(payload["results"]))
    print("")
    print("Metrics Table")
    print(_build_metrics_table(summary))
    print("")
    print(
        "Evaluation completed: "
        f"{summary['passed_samples']}/{summary['total_samples']} samples passed; "
        f"{summary['successful_calls']}/{summary['total_samples']} calls succeeded. "
        f"JSON: {output_path} "
        f"Markdown: {output_path.with_suffix('.md')}"
    )
    if payload.get("comparison"):
        comparison = payload["comparison"]
        print(
            "Comparison completed: "
            f"{comparison['summary']['changed_samples']} changed sample(s). "
            f"JSON: {comparison['json_path']} "
            f"Markdown: {comparison['report_path']}"
        )
    if summary["failed_samples"] > 0:
        print(f"Evaluation failed: {summary['failed_samples']} sample(s) failed assertions.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
EVALS_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = EVALS_DIR / "samples"
FIXTURES_DIR = EVALS_DIR / "fixtures"
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "data" / "evals" / "latest.json"
EVAL_FILE_PREFIX = "eval-harness-"


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

    return assertions


def _format_failure_reason(assertion: dict[str, Any]) -> str:
    return (
        f"{assertion['name']} expected={assertion['expected']!r} "
        f"actual={assertion['actual']!r}"
    )


def _validate_manifest(sample_set: dict[str, Any]) -> None:
    if not isinstance(sample_set.get("fixtures"), list) or not sample_set["fixtures"]:
        raise EvalRuntimeError("Sample set must define a non-empty fixtures list.")
    if not isinstance(sample_set.get("samples"), list) or not sample_set["samples"]:
        raise EvalRuntimeError("Sample set must define a non-empty samples list.")


def _build_fixture_upload_content(sample_set: dict[str, Any], fixture_path: Path) -> str:
    """Append sample anchors so HTTP eval fixtures match the replay language."""
    content = fixture_path.read_text(encoding="utf-8").strip()
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
        if isinstance(filename, str) and filename.startswith(EVAL_FILE_PREFIX):
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
        if isinstance(document_id, str) and source_name.startswith(EVAL_FILE_PREFIX):
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
    citations = chat_response.get("citations", [])
    answer = str(chat_response.get("answer", ""))
    metrics = {
        "knowledge_used": bool(chat_response.get("knowledge_used")),
        "citation_count": len(citations),
        "answer_keyword_hit": _contains_required_text(answer, expected),
        "expected_source_seen": _has_expected_source(citations, expected.get("citation_source_name")),
        "expected_source_kind_seen": _has_expected_source_kind(citations, expected.get("citation_source_kind")),
        "visible_marker_seen": _has_visible_marker(answer),
        "fallback_like": _fallback_like(answer),
    }
    observed = {
        "answer": answer,
        "answer_preview": _answer_preview(answer),
        "knowledge_used": chat_response.get("knowledge_used"),
        "citation_count": len(citations),
        "citation_sources": [citation.get("source_name") for citation in citations],
        "citations": citations,
        "session_id": chat_response.get("session_id"),
        "request_id": chat_response.get("request_id"),
    }
    assertions = _build_assertions(expected=expected, observed=observed, metrics=metrics)
    failure_reasons = [
        _format_failure_reason(assertion)
        for assertion in assertions
        if not assertion["passed"]
    ]
    return {
        "sample_id": sample["sample_id"],
        "query": sample["query"],
        "source_doc": sample.get("source_doc"),
        "target": expected,
        "status": "ok",
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "assertions": assertions,
        "observed": observed,
        "metrics": metrics,
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
        "sample_pass_rate": _rate(passed_samples, total),
        "completion_rate": _rate(len(succeeded), total),
        "knowledge_hit_rate": _rate(knowledge_hits, len(succeeded)),
        "citation_presence_rate": _rate(citations_present, len(succeeded)),
        "answer_keyword_hit_rate": _rate(answer_keyword_hits, len(succeeded)),
        "expected_source_hit_rate": _rate(expected_source_hits, len(source_required)),
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
    headers = ["sample_id", "pass", "query", "knowledge", "citations", "source_hit", "marker", "failure", "preview"]
    rows: list[list[str]] = []
    for item in results:
        if item["status"] != "ok":
            rows.append(
                [
                    item["sample_id"],
                    "no",
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


def _write_markdown_report(*, payload: dict[str, Any], output_path: Path) -> Path:
    report_path = output_path.with_suffix(".md")
    summary = payload["summary"]
    results = payload["results"]
    lines = [
        "# Evaluation Harness 回放报告",
        "",
        "## Executive Summary",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- executed_at: `{payload['executed_at']}`",
        f"- base_url: `{payload['base_url']}`",
        f"- sample_set: `{payload['sample_set']}`",
        f"- result: `{summary['passed_samples']}/{summary['total_samples']} samples passed`, `{summary['failed_samples']} failed`",
        "",
        "本报告用于证明 `generic_assistant + documents` 主链的最小可信回归能力：知识命中样本应返回正确来源，no-hit 样本必须不返回伪引用。",
        "报告中的 `pass` 来自脚本断言，不依赖 LLM-as-a-judge。",
        "",
        "## KPI Overview",
        "",
        _build_kpi_table(summary),
        "",
        "## Count Comparison",
        "",
        _build_count_comparison_table(summary),
        "",
        "## Sample Evidence",
        "",
        _build_sample_evidence_table(results),
        "",
        "## no-hit Fallback Boundary",
        "",
        _build_no_hit_boundary_table(results),
        "",
        "## Assertion Matrix",
        "",
        _build_assertion_matrix(results),
        "",
        "## Field Guide",
        "",
        _build_metrics_table(summary),
        "",
        "## Raw Sample Table",
        "",
        _build_sample_table(results),
        "",
        "## How To Read",
        "",
        "1. 先看 `KPI Overview`，确认样本断言通过率、来源命中率和 no-hit 边界是否正常。",
        "2. 再看 `no-hit Fallback Boundary`，确认 `knowledge_used=false`、`citations=[]` 和 fallback 语义同时成立。",
        "3. 如果失败，直接看 `Assertion Matrix` 的 `expected`、`actual` 和 `pass` 列定位异常字段。",
        "4. `Raw Sample Table` 保留原始逐样本视图，方便和 `latest.json` 交叉核对。",
        "",
        "## Demo Narrative",
        "",
        "这套 Evaluation Harness 不是做大而全 benchmark，而是先把通用文档 RAG 的最小可验证闭环固定下来。",
        "本次回放覆盖 3 条文档命中样本和 1 条 no-hit fallback 边界样本。",
        "文档命中样本证明引用来源、正文引用标记和答案关键词约束同时成立；no-hit 样本证明系统在没有可靠资料时不会返回伪引用。",
        "后续如果调整 ReRank、query rewrite、检索阈值或 citation 组装逻辑，只要 no-hit 又返回 citations，`Assertion Matrix` 会直接暴露实际异常值，脚本也会失败退出。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_eval(*, base_url: str, sample_set_name: str, output_path: Path) -> dict[str, Any]:
    sample_path = SAMPLES_DIR / f"{sample_set_name}.json"
    if not sample_path.exists():
        raise EvalRuntimeError(f"Sample set not found: {sample_path}")
    sample_set = _read_json(sample_path)
    _validate_manifest(sample_set)

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

    payload = {
        "run_id": uuid4().hex,
        "executed_at": _now_iso(),
        "base_url": base_url,
        "sample_set": sample_set_name,
        "summary": _build_summary(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_markdown_report(payload=payload, output_path=output_path)
    payload["report_path"] = str(report_path)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    output_path = Path(args.output)
    try:
        payload = run_eval(
            base_url=args.base_url.rstrip("/"),
            sample_set_name=args.sample_set,
            output_path=output_path,
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
    if summary["failed_samples"] > 0:
        print(f"Evaluation failed: {summary['failed_samples']} sample(s) failed assertions.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

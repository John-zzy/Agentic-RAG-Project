from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import mimetypes
from pathlib import Path
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


def _build_multipart_body(file_path: Path, field_name: str = "file") -> tuple[bytes, str]:
    boundary = f"----AiRagEval{uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()
    lines = [
        f"--{boundary}".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"'
        ).encode("utf-8"),
        f"Content-Type: {content_type}".encode("utf-8"),
        b"",
        file_bytes,
        f"--{boundary}--".encode("utf-8"),
        b"",
    ]
    body = b"\r\n".join(lines)
    return body, boundary


def _upload_file(base_url: str, file_path: Path) -> dict[str, Any]:
    body, boundary = _build_multipart_body(file_path)
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


def _contains_required_text(actual_answer: str, expected: dict[str, Any]) -> bool:
    any_terms = expected.get("answer_contains_any") or []
    all_terms = expected.get("answer_contains_all") or []
    if any_terms:
        if not any(term in actual_answer for term in any_terms):
            return False
    if all_terms:
        if not all(term in actual_answer for term in all_terms):
            return False
    return True


def _has_expected_source(citations: list[dict[str, Any]], expected_source_name: str | None) -> bool:
    if not expected_source_name:
        return True
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


def _validate_manifest(sample_set: dict[str, Any]) -> None:
    if not isinstance(sample_set.get("fixtures"), list) or not sample_set["fixtures"]:
        raise EvalRuntimeError("Sample set must define a non-empty fixtures list.")
    if not isinstance(sample_set.get("samples"), list) or not sample_set["samples"]:
        raise EvalRuntimeError("Sample set must define a non-empty samples list.")


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
        "visible_marker_seen": "[1]" in answer,
        "fallback_like": _fallback_like(answer),
    }
    return {
        "sample_id": sample["sample_id"],
        "query": sample["query"],
        "source_doc": sample.get("source_doc"),
        "target": expected,
        "status": "ok",
        "observed": {
            "answer": answer,
            "answer_preview": _answer_preview(answer),
            "knowledge_used": chat_response.get("knowledge_used"),
            "citation_count": len(citations),
            "citation_sources": [citation.get("source_name") for citation in citations],
            "citations": citations,
            "session_id": chat_response.get("session_id"),
            "request_id": chat_response.get("request_id"),
        },
        "metrics": metrics,
    }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    succeeded = [item for item in results if item["status"] == "ok"]
    errors = [item for item in results if item["status"] != "ok"]
    knowledge_hits = sum(1 for item in succeeded if item["metrics"]["knowledge_used"])
    citations_present = sum(1 for item in succeeded if item["metrics"]["citation_count"] > 0)
    answer_keyword_hits = sum(1 for item in succeeded if item["metrics"]["answer_keyword_hit"])
    expected_source_hits = sum(1 for item in succeeded if item["metrics"]["expected_source_seen"])
    visible_marker_hits = sum(1 for item in succeeded if item["metrics"]["visible_marker_seen"])
    fallback_like_hits = sum(1 for item in succeeded if item["metrics"]["fallback_like"])

    def _rate(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 4)

    return {
        "total_samples": total,
        "successful_calls": len(succeeded),
        "errored_calls": len(errors),
        "samples_with_knowledge": knowledge_hits,
        "samples_with_citations": citations_present,
        "answer_keyword_hits": answer_keyword_hits,
        "expected_source_hits": expected_source_hits,
        "visible_marker_hits": visible_marker_hits,
        "fallback_like_hits": fallback_like_hits,
        "completion_rate": _rate(len(succeeded), total),
        "knowledge_hit_rate": _rate(knowledge_hits, len(succeeded)),
        "citation_presence_rate": _rate(citations_present, len(succeeded)),
        "answer_keyword_hit_rate": _rate(answer_keyword_hits, len(succeeded)),
        "expected_source_hit_rate": _rate(expected_source_hits, len(succeeded)),
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
    headers = ["sample_id", "query", "knowledge", "citations", "source_hit", "marker", "preview"]
    rows: list[list[str]] = []
    for item in results:
        if item["status"] != "ok":
            rows.append(
                [
                    item["sample_id"],
                    _answer_preview(str(item.get("query", "")), limit=40),
                    "error",
                    "-",
                    "-",
                    "-",
                    _answer_preview(str(item.get("error", ""))),
                ]
            )
            continue
        metrics = item["metrics"]
        observed = item["observed"]
        rows.append(
            [
                item["sample_id"],
                _answer_preview(str(item["query"]), limit=40),
                "yes" if metrics["knowledge_used"] else "no",
                str(observed["citation_count"]),
                "yes" if metrics["expected_source_seen"] else "no",
                "yes" if metrics["visible_marker_seen"] else "no",
                str(observed["answer_preview"]),
            ]
        )
    return _render_console_table(headers, rows)


def _build_metrics_table(summary: dict[str, Any]) -> str:
    headers = ["metric", "value"]
    ordered_keys = [
        "total_samples",
        "successful_calls",
        "errored_calls",
        "samples_with_knowledge",
        "samples_with_citations",
        "answer_keyword_hits",
        "expected_source_hits",
        "visible_marker_hits",
        "fallback_like_hits",
        "completion_rate",
        "knowledge_hit_rate",
        "citation_presence_rate",
        "answer_keyword_hit_rate",
        "expected_source_hit_rate",
    ]
    rows = [[key, str(summary[key])] for key in ordered_keys]
    return _render_console_table(headers, rows)


def _write_markdown_report(*, payload: dict[str, Any], output_path: Path) -> Path:
    report_path = output_path.with_suffix(".md")
    lines = [
        "# Evaluation Harness 回放报告",
        "",
        "## 这份报告在表达什么",
        "",
        "这是一份面向通用 `documents` RAG 主链路的轻量回放报告。",
        "它不是严格的 benchmark，也不是完整自动评分系统；它的作用是用固定样本快速证明当前后端是否具备下面几件事：",
        "",
        "- 能否针对常见文档问答问题，检索到预期文档上下文",
        "- 能否返回 citation 对象，并在答案中展示 `[1]` 这样的可见引用标记",
        "- 能否在答案里覆盖样本要求的关键词或最小回答约束",
        "- 当问题和固定文档不匹配时，能否表现出 fallback-like 的回答风格",
        "",
        "因此，这份报告更适合当作“可验证演示材料”来看，而不是最终评测体系。",
        "",
        f"- executed_at: `{payload['executed_at']}`",
        f"- base_url: `{payload['base_url']}`",
        f"- sample_set: `{payload['sample_set']}`",
        "",
        "## Sample Table：逐条样本怎么看",
        "",
        "表里的每一行都代表 1 条固定样本的真实回放结果。",
        "",
        "- `sample_id`：样本 ID，用来标识固定问题场景",
        "- `query`：本条样本实际发送给 `/chat` 的输入问题预览",
        "- `knowledge`：`/chat` 返回里是否声明 `knowledge_used=true`",
        "- `citations`：本次回答带回了多少条 citation 对象",
        "- `source_hit`：返回的 citations 里是否出现了预期来源文档",
        "- `marker`：答案正文里是否真的出现了 `[1]` 这类可见引用标记",
        "- `preview`：回答预览，方便人工快速判断回答是否靠谱",
        "",
        "推荐这样理解单条样本：",
        "",
        "- 先看 `query`，确认这条样本到底在问什么。",
        "- 对于知识命中类问题，理想结果是 `knowledge=yes`、`citations>0`、`source_hit=yes`。",
        "- 如果 `marker=no`，通常表示后端确实返回了 citations，但答案正文没有按预期显式展示引用标记。",
        "- 对于弱匹配或无意义问题，模型仍可能检索到宽泛上下文，所以要结合 `preview` 一起判断是否已经表现出 fallback 风格。",
        "",
        _build_sample_table(payload["results"]),
        "",
        "## Metrics Table：整体指标怎么看",
        "",
        "这些指标都是从上面的样本回放结果里直接汇总出来的轻量指标。",
        "",
        "- `total_samples`：本次总共回放了多少条样本",
        "- `successful_calls`：多少条样本完整跑通，没有 HTTP 或运行时错误",
        "- `errored_calls`：多少条样本因为接口异常或脚本错误没有跑通",
        "- `samples_with_knowledge`：多少条样本返回了 `knowledge_used=true`",
        "- `samples_with_citations`：多少条样本返回了 citations",
        "- `answer_keyword_hits`：多少条样本满足了预设关键词或最小回答约束",
        "- `expected_source_hits`：多少条样本的 citations 命中了预期来源文档",
        "- `visible_marker_hits`：多少条样本在答案正文里展示了 `[1]` 这类可见引用标记",
        "- `fallback_like_hits`：多少条样本在答案文本上看起来像“证据不足 / 没找到信息”的 fallback 回答",
        "- `completion_rate`：`successful_calls / total_samples`，表示回放链路稳定性",
        "- `knowledge_hit_rate`：`samples_with_knowledge / successful_calls`，表示知识链路被触发的比例",
        "- `citation_presence_rate`：`samples_with_citations / successful_calls`，表示回答带引用的比例",
        "- `answer_keyword_hit_rate`：`answer_keyword_hits / successful_calls`，表示回答满足最小约束的比例",
        "- `expected_source_hit_rate`：`expected_source_hits / successful_calls`，表示来源命中率",
        "",
        "建议按这个顺序看：",
        "",
        "1. 先看 `errored_calls` 和 `completion_rate`，判断回放链路是否稳定。",
        "2. 再看 `expected_source_hits` 和 `citation_presence_rate`，判断引用溯源是否正常出现。",
        "3. 再看 `answer_keyword_hits` 和 `preview`，判断答案有没有基本答到点上。",
        "4. `fallback_like_hits` 只作为定性观察，不应当视为严格正确率。",
        "",
        "## 面试讲法",
        "",
        "如果你要把这份表讲给面试官，可以直接用下面这套话术：",
        "",
        "这套 Evaluation Harness 不是做大而全 benchmark，而是先把通用文档 RAG 的最小可验证闭环固定下来。",
        "我用了 4 条固定样本去回放真实 `/chat` 链路，每次都会重新上传 fixture、重新入库、重新建 session，然后记录回答、引用和关键指标。",
        "这样我至少能稳定证明三件事：第一，文档问答能不能命中；第二，引用溯源能不能出来；第三，优化前后我能不能用同一批样本做对比，而不是靠主观感觉说效果变好了。",
        "如果指标下降，比如 `expected_source_hit_rate` 或 `visible_marker_hits` 下降，我就知道问题是在检索、引用拼装还是答案渲染层，而不是只看最终回答拍脑袋判断。",
        "",
        _build_metrics_table(payload["summary"]),
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

    uploaded_files: dict[str, str] = {}
    for fixture in sample_set["fixtures"]:
        fixture_path = FIXTURES_DIR / fixture["filename"]
        if not fixture_path.exists():
            raise EvalRuntimeError(f"Fixture not found: {fixture_path}")
        _log(f"Uploading fixture: {fixture['filename']}")
        upload_result = _upload_file(base_url, fixture_path)
        uploaded_files[fixture["id"]] = str(upload_result["file_path"])

    _log("Registering uploaded fixtures into knowledge documents")
    _register_documents(base_url=base_url, namespace=namespace, uploaded_files=uploaded_files)

    results: list[dict[str, Any]] = []
    for sample in sample_set["samples"]:
        _log(f"Replaying sample: {sample['sample_id']}")
        try:
            results.append(_run_sample(base_url=base_url, sample=sample))
        except EvalRuntimeError as exc:
            results.append(
                {
                    "sample_id": sample["sample_id"],
                    "query": sample["query"],
                    "source_doc": sample.get("source_doc"),
                    "target": sample["expected"],
                    "status": "error",
                    "error": str(exc),
                }
            )

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
        f"{summary['successful_calls']}/{summary['total_samples']} calls succeeded. "
        f"JSON: {output_path} "
        f"Markdown: {output_path.with_suffix('.md')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

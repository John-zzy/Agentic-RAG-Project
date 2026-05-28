from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest

from backend.application.runtime.api.app import create_app
from backend.application.runtime.api.evals import routes as eval_routes
from backend.tests.test_support import make_test_runtime_dir


@pytest.fixture
def eval_tmp_path() -> Path:
    return make_test_runtime_dir("eval-api")


@contextmanager
def _client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(chat_service=object())  # type: ignore[arg-type]
    with TestClient(app) as client:
        client.app.state.settings = SimpleNamespace(data_dir=tmp_path)
        yield client


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_get_latest_returns_structured_404_when_missing(eval_tmp_path: Path) -> None:
    with _client(eval_tmp_path) as client:
        response = client.get("/evals/latest")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "EVAL_LATEST_NOT_FOUND"


def test_list_eval_runs_returns_index_or_empty(eval_tmp_path: Path) -> None:
    _write_json(
        eval_tmp_path / "evals" / "runs" / "index.json",
        {"runs": [{"run_id": "run-1", "sample_set": "minimal", "answer": "full answer"}]},
    )

    with _client(eval_tmp_path) as client:
        response = client.get("/evals/runs")

    assert response.status_code == 200
    assert response.json()["runs"] == [{"run_id": "run-1", "sample_set": "minimal"}]


def test_get_single_run_rejects_path_traversal_and_reads_only_json(eval_tmp_path: Path) -> None:
    _write_json(eval_tmp_path / "evals" / "runs" / "run-1.json", {"run_id": "run-1", "sample_set": "minimal"})

    with _client(eval_tmp_path) as client:
        ok_response = client.get("/evals/runs/run-1")
        traversal_response = client.get("/evals/runs/..evil")

    assert ok_response.status_code == 200
    assert ok_response.json()["run"]["run_id"] == "run-1"
    assert traversal_response.status_code == 404
    assert traversal_response.json()["detail"]["code"] == "EVAL_RUN_NOT_FOUND"


def test_eval_api_sanitizes_text_bearing_fields(eval_tmp_path: Path) -> None:
    _write_json(
        eval_tmp_path / "evals" / "latest.json",
        {
            "run_id": "run-safe",
            "answer": "FULL ANSWER",
            "results": [
                {
                    "sample_id": "s1",
                    "observed": {
                        "answer": "FULL ANSWER",
                        "answer_preview": "preview",
                        "citations": [{"snippet": "SECRET SNIPPET", "source_name": "doc.md"}],
                    },
                    "retrieval": {
                        "ranked_list": [{"source_doc": "doc.md", "content": "SECRET CONTENT"}],
                        "reason": "SECRET REASON",
                    },
                    "failure_reasons": [
                        "citations_empty expected=[] actual=[{'snippet': 'SECRET FAILURE SNIPPET', 'source_name': 'doc.md'}]",
                        "raw actual={\"content\":\"SECRET FAILURE CONTENT\",\"source_doc\":\"doc.md\"}",
                    ],
                    "prompt": "SECRET PROMPT",
                    "rewrite_reason": "SECRET REWRITE",
                }
            ],
        },
    )

    with _client(eval_tmp_path) as client:
        response = client.get("/evals/latest")

    assert response.status_code == 200
    serialized = json.dumps(response.json(), ensure_ascii=False)
    for forbidden in (
        "FULL ANSWER",
        "SECRET SNIPPET",
        "SECRET CONTENT",
        "SECRET REASON",
        "SECRET PROMPT",
        "SECRET REWRITE",
        "SECRET FAILURE SNIPPET",
        "SECRET FAILURE CONTENT",
        '"answer"',
        '"snippet"',
        '"content"',
        '"prompt"',
        '"reason"',
        '"rewrite_reason"',
    ):
        assert forbidden not in serialized
    assert "preview" in serialized


def test_trigger_eval_run_defaults_to_retrieval_benchmark(monkeypatch: Any, eval_tmp_path: Path) -> None:
    eval_routes.RUN_MANAGER.reset_for_tests()
    calls: list[dict[str, Any]] = []

    def fake_run_eval(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"run_id": kwargs["run_id"]}

    monkeypatch.setattr(eval_routes, "run_eval", fake_run_eval)

    with _client(eval_tmp_path) as client:
        response = client.post("/evals/runs", json={})

    assert response.status_code == 202
    payload = response.json()
    assert payload["sample_set"] == "retrieval_benchmark"
    assert payload["status"] == "queued"
    assert calls[0]["sample_set_name"] == "retrieval_benchmark"
    assert calls[0]["base_url"].startswith("http://testserver")


def test_trigger_eval_rejects_unknown_sample_set(eval_tmp_path: Path) -> None:
    eval_routes.RUN_MANAGER.reset_for_tests()

    with _client(eval_tmp_path) as client:
        response = client.post("/evals/runs", json={"sample_set": "unknown"})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "EVAL_SAMPLE_SET_NOT_ALLOWED"


def test_trigger_eval_rejects_concurrent_run(eval_tmp_path: Path) -> None:
    eval_routes.RUN_MANAGER.reset_for_tests()
    eval_routes.RUN_MANAGER.start(run_id="active-run", sample_set="minimal")

    try:
        with _client(eval_tmp_path) as client:
            response = client.post("/evals/runs", json={"sample_set": "minimal"})
    finally:
        eval_routes.RUN_MANAGER.reset_for_tests()

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "EVAL_RUN_ALREADY_RUNNING"


def test_get_run_status_from_artifact(eval_tmp_path: Path) -> None:
    eval_routes.RUN_MANAGER.reset_for_tests()
    _write_json(
        eval_tmp_path / "evals" / "runs" / "run-2.json",
        {"run_id": "run-2", "sample_set": "minimal", "executed_at": "2026-01-01T00:00:00+00:00"},
    )

    with _client(eval_tmp_path) as client:
        response = client.get("/evals/runs/run-2/status")
        missing = client.get("/evals/runs/missing/status")

    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    assert missing.status_code == 200
    assert missing.json()["status"] == "not_found"

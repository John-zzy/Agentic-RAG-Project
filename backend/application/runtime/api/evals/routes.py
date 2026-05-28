from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from backend.application.runtime.api.evals.schemas import (
    EvalRunDetailResponse,
    EvalRunListResponse,
    EvalRunStatusResponse,
    EvalRunTriggerRequest,
    EvalRunTriggerResponse,
)
from backend.evals.run_http_eval import DEFAULT_OUTPUT, run_eval
from backend.platform.config.settings import settings


ALLOWED_SAMPLE_SETS = {"minimal", "retrieval_benchmark"}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
TEXT_LEAK_KEYS = {
    "answer",
    "content",
    "page_content",
    "prompt",
    "raw_fixture_content",
    "reason",
    "rewrite_reason",
    "snippet",
    "system_prompt",
}
TEXT_LEAK_PATTERNS = (
    re.compile(r"'(?:answer|content|page_content|prompt|raw_fixture_content|reason|rewrite_reason|snippet)'\s*:\s*'[^']*'"),
    re.compile(r'"(?:answer|content|page_content|prompt|raw_fixture_content|reason|rewrite_reason|snippet)"\s*:\s*"[^"]*"'),
)


router = APIRouter(prefix="/evals", tags=["evals"])


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class EvalRunStateRecord:
    run_id: str
    sample_set: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    def to_response(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sample_set": self.sample_set,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class EvalRunManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._runs: dict[str, EvalRunStateRecord] = {}
        self._active_run_id: str | None = None

    def start(self, *, run_id: str, sample_set: str) -> EvalRunStateRecord:
        with self._lock:
            if self._active_run_id is not None:
                active = self._runs.get(self._active_run_id)
                if active is not None and active.status in {"queued", "running"}:
                    raise RuntimeError("eval run already running")
            record = EvalRunStateRecord(
                run_id=run_id,
                sample_set=sample_set,
                status="queued",
                started_at=_now_iso(),
            )
            self._runs[run_id] = record
            self._active_run_id = run_id
            return record

    def mark_running(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id].status = "running"

    def finish(self, run_id: str, *, error: str | None = None) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                record.status = "failed" if error else "succeeded"
                record.finished_at = _now_iso()
                record.error = error
            if self._active_run_id == run_id:
                self._active_run_id = None

    def get(self, run_id: str) -> EvalRunStateRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def reset_for_tests(self) -> None:
        with self._lock:
            self._runs.clear()
            self._active_run_id = None


RUN_MANAGER = EvalRunManager()


@router.get("/latest", response_model=EvalRunDetailResponse)
def get_latest_eval(request: Request) -> EvalRunDetailResponse:
    latest_path = _evals_root(request) / "latest.json"
    if not latest_path.exists():
        raise _error(status.HTTP_404_NOT_FOUND, "EVAL_LATEST_NOT_FOUND", "Latest eval artifact was not found.")
    return EvalRunDetailResponse(run=_sanitize_payload(_read_json(latest_path)))


@router.get("/runs", response_model=EvalRunListResponse)
def list_eval_runs(request: Request) -> EvalRunListResponse:
    index_path = _runs_root(request) / "index.json"
    if not index_path.exists():
        return EvalRunListResponse(runs=[])
    payload = _sanitize_payload(_read_json(index_path))
    runs = payload.get("runs", []) if isinstance(payload, dict) else []
    return EvalRunListResponse(runs=runs if isinstance(runs, list) else [])


@router.post("/runs", response_model=EvalRunTriggerResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_eval_run(
    payload: EvalRunTriggerRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> EvalRunTriggerResponse:
    sample_set = payload.sample_set
    if sample_set not in ALLOWED_SAMPLE_SETS:
        raise _error(422, "EVAL_SAMPLE_SET_NOT_ALLOWED", "Unsupported eval sample set.")
    run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    try:
        record = RUN_MANAGER.start(run_id=run_id, sample_set=sample_set)
    except RuntimeError as exc:
        raise _error(status.HTTP_409_CONFLICT, "EVAL_RUN_ALREADY_RUNNING", "An eval run is already queued or running.") from exc
    base_url = str(request.base_url).rstrip("/")
    output_path = _evals_root(request) / "latest.json"
    background_tasks.add_task(
        _run_eval_background,
        run_id=run_id,
        sample_set=sample_set,
        base_url=base_url,
        output_path=output_path,
    )
    return EvalRunTriggerResponse(**record.to_response())


@router.get("/runs/{run_id}/status", response_model=EvalRunStatusResponse)
def get_eval_run_status(run_id: str, request: Request) -> EvalRunStatusResponse:
    if not _valid_run_id(run_id):
        return EvalRunStatusResponse(run_id=run_id, status="not_found")
    record = RUN_MANAGER.get(run_id)
    if record is not None:
        return EvalRunStatusResponse(**record.to_response())
    run_path = _run_path(request, run_id)
    if run_path.exists():
        payload = _read_json(run_path)
        return EvalRunStatusResponse(
            run_id=run_id,
            sample_set=payload.get("sample_set"),
            status="failed" if payload.get("error") else "succeeded",
            started_at=payload.get("executed_at"),
            finished_at=payload.get("executed_at"),
            error=payload.get("error"),
        )
    return EvalRunStatusResponse(run_id=run_id, status="not_found")


@router.get("/runs/{run_id}", response_model=EvalRunDetailResponse)
def get_eval_run(run_id: str, request: Request) -> EvalRunDetailResponse:
    if not _valid_run_id(run_id):
        raise _error(status.HTTP_404_NOT_FOUND, "EVAL_RUN_NOT_FOUND", "Eval run was not found.")
    run_path = _run_path(request, run_id)
    if not run_path.exists():
        raise _error(status.HTTP_404_NOT_FOUND, "EVAL_RUN_NOT_FOUND", "Eval run was not found.")
    return EvalRunDetailResponse(run=_sanitize_payload(_read_json(run_path)))


def _run_eval_background(*, run_id: str, sample_set: str, base_url: str, output_path: Path) -> None:
    RUN_MANAGER.mark_running(run_id)
    try:
        run_eval(
            base_url=base_url,
            sample_set_name=sample_set,
            output_path=output_path,
            run_id=run_id,
        )
    except Exception as exc:
        RUN_MANAGER.finish(run_id, error=str(exc))
        return
    RUN_MANAGER.finish(run_id)


def _evals_root(request: Request) -> Path:
    app_settings = getattr(request.app.state, "settings", settings)
    return Path(app_settings.data_dir) / "evals"


def _runs_root(request: Request) -> Path:
    return _evals_root(request) / "runs"


def _run_path(request: Request, run_id: str) -> Path:
    runs_root = _runs_root(request).resolve()
    candidate = (runs_root / f"{run_id}.json").resolve()
    if runs_root != candidate.parent:
        raise _error(status.HTTP_404_NOT_FOUND, "EVAL_RUN_NOT_FOUND", "Eval run was not found.")
    return candidate


def _valid_run_id(run_id: str) -> bool:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        return False
    return "/" not in run_id and "\\" not in run_id and ".." not in run_id


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in TEXT_LEAK_KEYS:
                continue
            sanitized[key] = _sanitize_payload(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, str):
        sanitized = value
        for pattern in TEXT_LEAK_PATTERNS:
            sanitized = pattern.sub("[redacted]", sanitized)
        return sanitized
    return value


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})

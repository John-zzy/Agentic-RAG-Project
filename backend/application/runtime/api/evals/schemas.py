from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


EvalRunState = Literal["queued", "running", "succeeded", "failed", "not_found"]
EvalSampleSet = Literal["minimal", "retrieval_benchmark"]


class EvalErrorResponse(BaseModel):
    code: str
    message: str


class EvalRunTriggerRequest(BaseModel):
    sample_set: str = "retrieval_benchmark"


class EvalRunStatusResponse(BaseModel):
    run_id: str
    sample_set: EvalSampleSet | None = None
    status: EvalRunState
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class EvalRunListResponse(BaseModel):
    runs: list[dict[str, Any]] = Field(default_factory=list)


class EvalRunDetailResponse(BaseModel):
    run: dict[str, Any]


class EvalRunTriggerResponse(EvalRunStatusResponse):
    pass

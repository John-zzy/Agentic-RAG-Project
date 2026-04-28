from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    stream: bool = False
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    citation_id: str
    namespace: str
    snippet: str
    score: float | None = None


class ChatResponse(BaseModel):
    session_id: str
    request_id: str
    answer: str
    knowledge_used: bool
    citations: list[Citation] = Field(default_factory=list)


class SessionCreateResponse(BaseModel):
    session_id: str


class SessionTurnResponse(BaseModel):
    request_id: str
    user_message: str
    assistant_answer: str
    retrieval_snippets: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: str


class SessionDetailResponse(BaseModel):
    session_id: str
    total_turns: int
    turns: list[SessionTurnResponse] = Field(default_factory=list)


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted_turns: int

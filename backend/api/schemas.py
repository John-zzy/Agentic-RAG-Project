from __future__ import annotations

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


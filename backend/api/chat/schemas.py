from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天接口请求体。"""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    stream: bool = False
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    """回答引用片段。"""

    citation_id: str
    namespace: str
    snippet: str
    score: float | None = None


class ChatResponse(BaseModel):
    """聊天接口响应体。"""

    session_id: str
    request_id: str
    answer: str
    knowledge_used: bool
    citations: list[Citation] = Field(default_factory=list)


class SessionCreateResponse(BaseModel):
    """会话创建响应体。"""

    session_id: str


class SessionTurnResponse(BaseModel):
    """会话单轮响应体。"""

    request_id: str
    user_message: str
    assistant_answer: str
    retrieval_snippets: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: str


class SessionDetailResponse(BaseModel):
    """会话详情响应体。"""

    session_id: str
    total_turns: int
    turns: list[SessionTurnResponse] = Field(default_factory=list)


class SessionDeleteResponse(BaseModel):
    """会话删除响应体。"""

    session_id: str
    deleted_turns: int

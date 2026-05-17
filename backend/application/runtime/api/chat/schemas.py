from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天接口请求体。"""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    stream: bool = False
    top_k: int | None = Field(default=None, ge=1, le=20)


class SceneSummary(BaseModel):
    """场景列表项。"""

    scene: str
    name: str
    description: str
    is_default: bool = False


class SceneListResponse(BaseModel):
    """场景列表响应。"""

    default_scene: str
    scenes: list[SceneSummary] = Field(default_factory=list)


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
    scene: str
    agent: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class SessionCreateResponse(BaseModel):
    """会话创建响应体。"""

    session_id: str
    scene: str
    mounted_knowledge_sources: list[str] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    """会话创建请求体。"""

    scene: str | None = None
    mounted_knowledge_sources: list[str] | None = None


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
    scene: str
    mounted_knowledge_sources: list[str] = Field(default_factory=list)
    total_turns: int
    turns: list[SessionTurnResponse] = Field(default_factory=list)


class SessionDeleteResponse(BaseModel):
    """会话删除响应体。"""

    session_id: str
    deleted_turns: int

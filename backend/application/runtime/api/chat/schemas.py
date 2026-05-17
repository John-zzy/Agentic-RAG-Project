from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天接口请求体。"""

    message: str = Field(min_length=1, max_length=4000, description="用户本轮输入的问题。")
    session_id: str | None = Field(default=None, description="会话 ID，不传时由服务端自动创建。")
    stream: bool = Field(default=False, description="是否请求流式输出；当前接口暂未启用。")
    top_k: int | None = Field(default=None, ge=1, le=20, description="检索条数上限。")


class SceneSummary(BaseModel):
    """场景列表项。"""

    scene: str = Field(description="场景唯一标识。")
    name: str = Field(description="场景展示名称。")
    description: str = Field(description="场景说明文案。")
    is_default: bool = Field(default=False, description="是否为默认场景。")


class SceneListResponse(BaseModel):
    """场景列表响应。"""

    default_scene: str = Field(description="当前系统默认场景。")
    scenes: list[SceneSummary] = Field(default_factory=list, description="可选场景列表。")


class Citation(BaseModel):
    """统一的回答引用信息。"""

    index: int = Field(ge=1, description="回答中展示的引用编号，从 1 开始。")
    citation_id: str = Field(description="引用的稳定 ID。")
    namespace: str = Field(description="引用所属命名空间。")
    source_kind: str = Field(description="来源类型，例如 document_chunk、product、order。")
    source_name: str = Field(description="前端展示用的来源名称。")
    source_path: str | None = Field(default=None, description="来源路径或来源主键。")
    document_id: str | None = Field(default=None, description="文档来源对应的文档 ID。")
    chunk_id: str | None = Field(default=None, description="文档分块 ID。")
    chunk_index: int | None = Field(default=None, description="文档分块序号。")
    snippet: str = Field(description="用于展示的命中文本片段。")
    score: float | None = Field(default=None, description="检索得分。")
    rank: int = Field(ge=1, description="原始检索排序位置，从 1 开始。")


class ChatResponse(BaseModel):
    """聊天接口响应体。"""

    session_id: str = Field(description="当前会话 ID。")
    request_id: str = Field(description="本次请求 ID。")
    answer: str = Field(description="最终回答文本，包含可见引用编号。")
    knowledge_used: bool = Field(description="本轮是否使用了知识检索结果。")
    scene: str = Field(description="本轮回答所属场景。")
    agent: str | None = Field(default=None, description="场景使用的代理标识，没有则为空。")
    citations: list[Citation] = Field(default_factory=list, description="结构化引用列表。")


class SessionCreateResponse(BaseModel):
    """会话创建响应体。"""

    session_id: str = Field(description="新创建的会话 ID。")
    scene: str = Field(description="会话绑定的场景。")
    mounted_knowledge_sources: list[str] = Field(
        default_factory=list,
        description="当前会话允许使用的知识源列表。",
    )


class SessionCreateRequest(BaseModel):
    """会话创建请求体。"""

    scene: str | None = Field(default=None, description="要绑定的场景；不传时使用默认场景。")
    mounted_knowledge_sources: list[str] | None = Field(
        default=None,
        description="要挂载的知识源列表，例如 documents、ecommerce。",
    )


class SessionTurnResponse(BaseModel):
    """会话单轮响应体。"""

    request_id: str = Field(description="该轮对话的请求 ID。")
    user_message: str = Field(description="用户问题。")
    assistant_answer: str = Field(description="助手回答。")
    retrieval_snippets: list[dict[str, Any]] = Field(
        default_factory=list,
        description="该轮保存的引用片段列表，与 citations 契约兼容。",
    )
    timestamp: str = Field(description="该轮写入时间。")


class SessionDetailResponse(BaseModel):
    """会话详情响应体。"""

    session_id: str = Field(description="会话 ID。")
    scene: str = Field(description="会话绑定的场景。")
    mounted_knowledge_sources: list[str] = Field(
        default_factory=list,
        description="该会话当前挂载的知识源列表。",
    )
    total_turns: int = Field(description="该会话历史总轮数。")
    turns: list[SessionTurnResponse] = Field(default_factory=list, description="最近的会话轮次列表。")


class SessionDeleteResponse(BaseModel):
    """会话删除响应体。"""

    session_id: str = Field(description="被删除的会话 ID。")
    deleted_turns: int = Field(description="被删除的轮次数量。")

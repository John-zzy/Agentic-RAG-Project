from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from backend.application.runtime.api.chat.schemas import (
    ChatRequest,
    ChatResponse,
    SceneListResponse,
    SceneSummary,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionTurnResponse,
)
from backend.platform.knowledge.sources import MountedKnowledgeSourceValidationError


router = APIRouter()


@router.get("/health")
def healthcheck() -> dict[str, str]:
    """健康检查接口。"""
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """聊天接口：执行检索增强问答并返回回答。"""
    service = _get_chat_service(request)
    try:
        return service.chat(payload)
    except Exception as exc:
        from backend.application.runtime.service import ChatServiceError

        if not isinstance(exc, ChatServiceError):
            raise
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "code": exc.code,
                "message": exc.message,
                "request_id": exc.request_id,
            },
        ) from exc


@router.get("/scenes", response_model=SceneListResponse)
def list_scenes(request: Request) -> SceneListResponse:
    """返回当前运行时支持的场景列表。"""
    service = _get_chat_service(request)
    default_scene = service.default_scene()
    definitions = service.list_scenes()
    return SceneListResponse(
        default_scene=default_scene,
        scenes=[
            SceneSummary(
                scene=definition.scene,
                name=definition.name,
                description=definition.description,
                is_default=definition.scene == default_scene,
            )
            for definition in definitions
        ],
    )


@router.post("/sessions", response_model=SessionCreateResponse)
def create_session(
    request: Request,
    payload: SessionCreateRequest | None = None,
) -> SessionCreateResponse:
    """创建新会话并返回会话 ID。"""
    service = _get_chat_service(request)
    requested_scene = payload.scene if payload is not None else None
    requested_sources = payload.mounted_knowledge_sources if payload is not None else None
    try:
        scene = service.validate_scene(requested_scene or service.default_scene())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UNKNOWN_SCENE",
                "message": str(exc),
                "request_id": "N/A",
            },
        ) from exc
    try:
        mounted_knowledge_sources = service.validate_mounted_knowledge_sources(requested_sources)
    except MountedKnowledgeSourceValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_MOUNTED_KNOWLEDGE_SOURCES",
                "message": str(exc),
                "request_id": "N/A",
            },
        ) from exc

    created = service.create_session(
        scene=scene,
        mounted_knowledge_sources=mounted_knowledge_sources,
    )
    return SessionCreateResponse(
        session_id=created.session_id,
        scene=created.scene,
        mounted_knowledge_sources=list(created.mounted_knowledge_sources),
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(
    session_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> SessionDetailResponse:
    """查询会话详情。"""
    service = _get_chat_service(request)
    session = service.session_store.get_session(session_id)
    session_scene = session.scene if session is not None else service.default_scene()
    mounted_knowledge_sources = (
        list(session.mounted_knowledge_sources)
        if session is not None
        else list(service.default_mounted_knowledge_sources())
    )
    turns, total_turns = service.session_store.get_session_detail(session_id=session_id, limit=limit)
    return SessionDetailResponse(
        session_id=session_id,
        scene=session_scene,
        mounted_knowledge_sources=mounted_knowledge_sources,
        total_turns=total_turns,
        turns=[
            SessionTurnResponse(
                request_id=turn.request_id,
                user_message=turn.user_message,
                assistant_answer=turn.assistant_answer,
                retrieval_snippets=turn.retrieval_snippets,
                timestamp=turn.timestamp,
            )
            for turn in turns
        ],
    )


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
def delete_session(session_id: str, request: Request) -> SessionDeleteResponse:
    """删除指定会话及其全部历史消息。"""
    service = _get_chat_service(request)
    deleted_turns = service.session_store.delete_session(session_id=session_id)
    return SessionDeleteResponse(session_id=session_id, deleted_turns=deleted_turns)


def _get_chat_service(request: Request) -> Any:
    """从应用状态中获取 chat service。"""
    service = getattr(request.app.state, "chat_service", None)
    if service is not None and hasattr(service, "chat") and hasattr(service, "session_store"):
        return service

    raise HTTPException(
        status_code=500,
        detail={
            "code": "SERVICE_NOT_INITIALIZED",
            "message": "Chat service is not initialized.",
            "request_id": "N/A",
        },
    )

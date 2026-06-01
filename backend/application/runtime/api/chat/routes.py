from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.application.runtime.api.chat.schemas import (
    ChatRequest,
    ChatResponse,
    SceneListResponse,
    SceneSummary,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionMessageResponse,
)
from backend.application.runtime.service import ChatServiceError
from backend.platform.knowledge.sources import MountedKnowledgeSourceValidationError


router = APIRouter()


def _build_error_detail(*, code: str, message: str, request_id: str = "N/A") -> dict[str, str]:
    """统一构造 API 错误体，避免各路由返回结构不一致。"""
    return {
        "code": code,
        "message": message,
        "request_id": request_id,
    }


@router.get("/health")
def healthcheck() -> dict[str, str]:
    """健康检查接口。"""
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse | StreamingResponse:
    """聊天接口：执行检索增强问答并返回回答。"""
    service = _get_chat_service(request)
    if payload.stream:
        return StreamingResponse(
            _stream_chat_events(service=service, payload=payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    try:
        return service.chat(payload)
    except Exception as exc:
        if not isinstance(exc, ChatServiceError):
            raise
        raise HTTPException(
            status_code=exc.status_code,
            detail=_build_error_detail(
                code=exc.code,
                message=exc.message,
                request_id=exc.request_id,
            ),
        ) from exc


def _stream_chat_events(service: Any, payload: ChatRequest) -> Iterator[str]:
    """将业务层结构化事件编码为 SSE 文本流。"""
    try:
        yield from (
            _encode_sse_event(event.event, event.data)
            for event in service.chat_stream(payload)
        )
    except Exception as exc:
        if not isinstance(exc, ChatServiceError):
            raise
        yield _encode_sse_event(
            "error",
            _build_error_detail(
                code=exc.code,
                message=exc.message,
                request_id=exc.request_id,
            ),
        )


def _encode_sse_event(event: str, data: dict[str, Any]) -> str:
    """统一编码 SSE 事件，data 固定为 JSON。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


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
            detail=_build_error_detail(code="UNKNOWN_SCENE", message=str(exc)),
        ) from exc
    try:
        mounted_knowledge_sources = service.validate_mounted_knowledge_sources(requested_sources)
    except MountedKnowledgeSourceValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=_build_error_detail(
                code="INVALID_MOUNTED_KNOWLEDGE_SOURCES",
                message=str(exc),
            ),
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
    messages, total_messages = service.session_store.get_session_messages(
        session_id=session_id,
        limit=limit,
    )
    return SessionDetailResponse(
        session_id=session_id,
        scene=session_scene,
        mounted_knowledge_sources=mounted_knowledge_sources,
        total_messages=total_messages,
        messages=[
            SessionMessageResponse(
                type=message.message_type,
                content=message.content,
                request_id=message.request_id,
                timestamp=message.timestamp,
                knowledge_used=message.knowledge_used,
                citations=message.citations,
            )
            for message in messages
        ],
    )


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
def delete_session(session_id: str, request: Request) -> SessionDeleteResponse:
    """删除指定会话及其全部历史消息。"""
    service = _get_chat_service(request)
    if not hasattr(service, "delete_session"):
        raise HTTPException(
            status_code=500,
            detail=_build_error_detail(
                code="SERVICE_NOT_INITIALIZED",
                message="Chat service does not support session deletion.",
            ),
        )
    deleted_messages = service.delete_session(session_id=session_id)
    return SessionDeleteResponse(session_id=session_id, deleted_messages=deleted_messages)


def _get_chat_service(request: Request) -> Any:
    """从应用状态中获取 chat service。"""
    service = getattr(request.app.state, "chat_service", None)
    if service is not None and hasattr(service, "chat") and hasattr(service, "session_store"):
        return service

    raise HTTPException(
        status_code=500,
        detail=_build_error_detail(
            code="SERVICE_NOT_INITIALIZED",
            message="Chat service is not initialized.",
        ),
    )

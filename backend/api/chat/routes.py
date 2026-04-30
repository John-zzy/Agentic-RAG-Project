from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request

from backend.api.chat.schemas import (
    ChatRequest,
    ChatResponse,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionTurnResponse,
)
from backend.api.chat.service import ChatService, ChatServiceError


router = APIRouter()


@router.get("/health")
def healthcheck() -> dict[str, str]:
    """健康检查接口，返回服务可用状态。"""
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """聊天接口：执行检索增强问答并返回回答。"""
    service = _get_chat_service(request)
    try:
        return service.chat(payload)
    except ChatServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "code": exc.code,
                "message": exc.message,
                "request_id": exc.request_id,
            },
        ) from exc


@router.post("/sessions", response_model=SessionCreateResponse)
def create_session(request: Request) -> SessionCreateResponse:
    """创建新会话并返回会话 ID。"""
    service = _get_chat_service(request)
    session_id = uuid4().hex
    service.session_store.create_session(session_id=session_id)
    return SessionCreateResponse(session_id=session_id)


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(
    session_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> SessionDetailResponse:
    """查询会话详情，按时间顺序返回最近对话。"""
    service = _get_chat_service(request)
    turns, total_turns = service.session_store.get_session_detail(
        session_id=session_id, limit=limit
    )
    return SessionDetailResponse(
        session_id=session_id,
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


def _get_chat_service(request: Request) -> ChatService:
    """从应用状态中获取 ChatService，不存在时抛出 500。"""
    service = getattr(request.app.state, "chat_service", None)
    if isinstance(service, ChatService):
        return service

    raise HTTPException(
        status_code=500,
        detail={
            "code": "SERVICE_NOT_INITIALIZED",
            "message": "Chat service is not initialized.",
            "request_id": "N/A",
        },
    )

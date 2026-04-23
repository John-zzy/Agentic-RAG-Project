from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.api.chat_service import ChatService, ChatServiceError
from backend.api.schemas import ChatRequest, ChatResponse


router = APIRouter()


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
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


def _get_chat_service(request: Request) -> ChatService:
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


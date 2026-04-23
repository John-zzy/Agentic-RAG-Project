from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.chat_service import ChatService, create_chat_service
from backend.api.routes import router as api_router
from backend.config.settings import settings


def create_app(chat_service: ChatService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.chat_service = chat_service or create_chat_service()
        yield

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()


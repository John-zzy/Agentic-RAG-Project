from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api.chat.routes import router as api_router
from backend.api.chat.service import ChatService, create_chat_service
from backend.config.settings import settings


def create_app(chat_service: ChatService | None = None) -> FastAPI:
    """创建并配置 FastAPI 应用。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期：注入配置与 ChatService。"""
        app.state.settings = settings
        app.state.chat_service = chat_service or create_chat_service()
        yield

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Ecommerce customer-service agent backend API.",
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.include_router(api_router)
    frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
    if frontend_dir.exists():
        app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")
    return app


app = create_app()

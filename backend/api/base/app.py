from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.chat.routes import router as api_router
from backend.api.chat.service import ChatService, create_chat_service
from backend.api.knowledge.routes import router as knowledge_document_router
from backend.config.settings import settings
from backend.knowledge.documents.service import KnowledgeDocumentService


def create_app(
    chat_service: ChatService | None = None,
    knowledge_document_service: KnowledgeDocumentService | None = None,
) -> FastAPI:
    """创建并配置 FastAPI 应用。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期：注入配置与 ChatService。"""
        app.state.settings = settings
        app.state.chat_service = chat_service or create_chat_service()
        app.state.knowledge_document_service = knowledge_document_service or KnowledgeDocumentService()
        yield

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Ecommerce customer-service agent backend API.",
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    app.include_router(knowledge_document_router)
    frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
    if frontend_dir.exists():
        app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")
    return app


app = create_app()

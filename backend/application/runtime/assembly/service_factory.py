from __future__ import annotations

from backend.application.runtime.service import (
    ActiveSceneChatService,
    SceneRegistry,
    build_default_scene_registry,
)
from backend.platform.config.settings import AppSettings, settings
from backend.platform.knowledge.documents import KnowledgeDocumentApplicationService, KnowledgeDocumentQueryService
from backend.platform.memory.base.session_store import SQLiteSessionStore
from backend.platform.memory.chat.prompt_context import PromptContextBuilder
from backend.platform.models.llm.client import ModelClient
from backend.platform.rag.retrieval.documents import DocumentRetrievalService
from backend.scenes.generic_assistant.definition import GenericAssistantBusinessExtension


def create_chat_service(
        app_settings: AppSettings | None = None,
        knowledge_service: object | None = None,
        document_retrieval_service: DocumentRetrievalService | None = None,
        generic_business_extensions: tuple[GenericAssistantBusinessExtension, ...] | None = None,
        include_default_business_extensions: bool = True,
        session_store: SQLiteSessionStore | None = None,
        context_builder: PromptContextBuilder | None = None,
        model: ModelClient | None = None,
        graph_runtime: object | None = None,
) -> ActiveSceneChatService:
    """聊天服务工厂函数，返回统一场景运行时服务。"""
    resolved_settings = app_settings or settings
    scene_registry = build_default_scene_registry(
        app_settings=resolved_settings,
        knowledge_service=knowledge_service,
        document_retrieval_service=document_retrieval_service,
        generic_business_extensions=generic_business_extensions,
        include_default_business_extensions=include_default_business_extensions,
    )
    return ActiveSceneChatService(
        scene_registry=scene_registry,
        app_settings=resolved_settings,
        knowledge_service=knowledge_service,
        session_store=session_store,
        context_builder=context_builder,
        model=model,
        graph_runtime=graph_runtime,
    )

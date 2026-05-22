from __future__ import annotations

from backend.platform.config.settings import AppSettings, settings
from backend.platform.rag.document_retrieval import DocumentRetrievalService
from backend.scenes.base import SceneDefinition
from backend.scenes.ecommerce.definition import (
    build_ecommerce_business_extension,
    build_ecommerce_scene_definition,
)
from backend.scenes.generic_assistant.definition import (
    GenericAssistantBusinessExtension,
    build_generic_assistant_scene_definition,
)


def build_default_business_extensions(
    *,
    app_settings: AppSettings | None = None,
    knowledge_service: object | None = None,
) -> tuple[GenericAssistantBusinessExtension, ...]:
    """构建默认注册到 generic scene 的业务扩展集合。"""
    resolved_settings = app_settings or settings
    return (
        build_ecommerce_business_extension(
            app_settings=resolved_settings,
            knowledge_service=knowledge_service,  # type: ignore[arg-type]
        ),
    )


def build_default_scene_definitions(
    *,
    app_settings: AppSettings | None = None,
    knowledge_service: object | None = None,
    document_retrieval_service: DocumentRetrievalService | None = None,
    generic_business_extensions: tuple[GenericAssistantBusinessExtension, ...] | None = None,
    include_default_business_extensions: bool = True,
) -> tuple[SceneDefinition, ...]:
    """组合默认 scene definitions，避免 runtime 写死业务扩展装配。"""
    resolved_settings = app_settings or settings
    resolved_extensions = (
        generic_business_extensions
        if generic_business_extensions is not None
        else (
            build_default_business_extensions(
                app_settings=resolved_settings,
                knowledge_service=knowledge_service,
            )
            if include_default_business_extensions
            else ()
        )
    )
    return (
        build_generic_assistant_scene_definition(
            app_settings=resolved_settings,
            business_extensions=resolved_extensions,
            document_retrieval_service=document_retrieval_service,
        ),
        build_ecommerce_scene_definition(
            app_settings=resolved_settings,
            knowledge_service=knowledge_service,
            document_retrieval_service=document_retrieval_service,
        ),
    )

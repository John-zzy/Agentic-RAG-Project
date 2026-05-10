from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backend.application.runtime.service import build_default_scene_registry, create_chat_service
from backend.platform.config.settings import AppSettings, settings


@dataclass(frozen=True)
class BootstrapSummary:
    """描述运行时启动时的关键装配结果。"""

    sqlite_path: Path
    active_scene: str
    scene_bootstrap_metrics: dict[str, int] = field(default_factory=dict)


def bootstrap_runtime(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: object | None = None,
) -> tuple[object, BootstrapSummary]:
    """初始化当前激活场景，并返回聊天服务与启动摘要。"""
    resolved_settings = app_settings or settings
    scene_registry = build_default_scene_registry(
        app_settings=resolved_settings,
        knowledge_service=knowledge_service,
    )
    active_definition = scene_registry.get_default_definition()
    bootstrap_result = active_definition.bootstrap() if active_definition.bootstrap else None
    chat_service = create_chat_service(
        app_settings=resolved_settings,
        knowledge_service=knowledge_service,
    )
    summary = BootstrapSummary(
        sqlite_path=resolved_settings.session.sqlite_path,
        active_scene=resolved_settings.app.active_scene,
        scene_bootstrap_metrics=bootstrap_result.metrics if bootstrap_result else {},
    )
    return chat_service, summary

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import uvicorn

from backend.api.base.app import create_app
from backend.api.chat.service import ChatService, create_chat_service
from backend.config.settings import AppSettings, settings
from backend.knowledge.ecommerce.loader import preload_knowledge_base
from backend.knowledge.ecommerce.service import create_knowledge_service
from backend.memory.base.session_store import SQLiteSessionStore


@dataclass(frozen=True)
class BootstrapSummary:
    sqlite_path: Path
    products_loaded: int
    reviews_loaded: int


def bootstrap_runtime(app_settings: AppSettings | None = None) -> tuple[ChatService, BootstrapSummary]:
    """初始化会话存储、知识库与聊天服务，并返回启动摘要。"""
    resolved_settings = app_settings or settings

    session_store = SQLiteSessionStore(app_settings=resolved_settings)
    knowledge_service = create_knowledge_service(app_settings=resolved_settings)
    load_summary = preload_knowledge_base(
        app_settings=resolved_settings,
        store=knowledge_service.store,
    )
    chat_service = create_chat_service(
        app_settings=resolved_settings,
        knowledge_service=knowledge_service,
        session_store=session_store,
    )

    summary = BootstrapSummary(
        sqlite_path=resolved_settings.session.sqlite_path,
        products_loaded=load_summary.products_loaded,
        reviews_loaded=load_summary.reviews_loaded,
    )
    return chat_service, summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数，支持覆盖 host/port。"""
    parser = argparse.ArgumentParser(description="Run AI RAG backend service.")
    parser.add_argument("--host", type=str, default=None, help="Override host from settings.")
    parser.add_argument("--port", type=int, default=None, help="Override port from settings.")
    return parser.parse_args()


def main() -> None:
    """服务启动入口：完成引导并启动 Uvicorn。"""
    args = parse_args()
    chat_service, summary = bootstrap_runtime(settings)
    app = create_app(chat_service=chat_service)

    host = args.host or settings.host
    port = args.port or settings.port

    print(
        "Bootstrap complete: "
        f"sqlite={summary.sqlite_path}, "
        f"products={summary.products_loaded}, "
        f"reviews={summary.reviews_loaded} \n"
        f"running on host {host}: {port}"
    )
    uvicorn.run(app=app, host=host, port=port)


if __name__ == "__main__":
    main()

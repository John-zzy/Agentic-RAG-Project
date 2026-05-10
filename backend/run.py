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

from backend.application.runtime.api.app import create_app
from backend.application.runtime import BootstrapSummary, SceneChatService, bootstrap_runtime
from backend.platform.config.settings import AppSettings, settings


def parse_args() -> argparse.Namespace:
    """解析命令行参数，支持覆盖 host/port。"""
    parser = argparse.ArgumentParser(description="Run AI RAG backend service.")
    parser.add_argument("--host", type=str, default=None, help="Override host from settings.")
    parser.add_argument("--port", type=int, default=None, help="Override port from settings.")
    return parser.parse_args()


def main() -> None:
    """服务启动入口。"""
    args = parse_args()
    chat_service, summary = bootstrap_runtime(settings)
    app = create_app(chat_service=chat_service)

    host = args.host or settings.host
    port = args.port or settings.port

    print(
        "Bootstrap complete: "
        f"scene={summary.active_scene}, "
        f"sqlite={summary.sqlite_path}, "
        f"metrics={summary.scene_bootstrap_metrics}\n"
        f"running on host {host}: {port}"
    )
    uvicorn.run(app=app, host=host, port=port)


if __name__ == "__main__":
    main()

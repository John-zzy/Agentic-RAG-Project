from __future__ import annotations

from pathlib import Path

from backend.platform.agent_runtime.chat_graph.runtime import (
    ChatGraphRuntime as PlatformChatGraphRuntime,
)
from backend.platform.config.settings import AppSettings
from backend.platform.workflow.langgraph.checkpointer import SQLiteLangGraphCheckpointer


class ChatGraphRuntime(PlatformChatGraphRuntime):
    """Application 装配层：只负责按配置创建 platform ChatGraphRuntime。"""

    @classmethod
    def from_settings(cls, app_settings: AppSettings) -> "ChatGraphRuntime":
        sqlite_path = Path(app_settings.data_dir) / "langgraph.db"
        return cls(checkpointer=SQLiteLangGraphCheckpointer(sqlite_path))

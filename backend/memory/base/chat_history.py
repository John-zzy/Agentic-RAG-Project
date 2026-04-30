from __future__ import annotations

from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from backend.memory.base.session_store import SQLiteSessionStore


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """将现有 SQLiteSessionStore 适配为 LangChain 消息历史接口。"""

    def __init__(
        self,
        session_id: str,
        *,
        store: SQLiteSessionStore | None = None,
    ) -> None:
        """基于指定会话 ID 暴露 LangChain 可直接消费的消息历史。"""
        self.session_id = session_id
        self.store = store or SQLiteSessionStore()

    @property
    def messages(self) -> list[BaseMessage]:
        """读取会话全部 LangChain message。"""
        return self.store.get_messages(self.session_id)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """批量写入消息，减少持久化往返次数。"""
        self.store.append_messages(self.session_id, list(messages))

    def clear(self) -> None:
        """清空当前会话对应的持久化消息和轮次。"""
        self.store.delete_session(self.session_id)

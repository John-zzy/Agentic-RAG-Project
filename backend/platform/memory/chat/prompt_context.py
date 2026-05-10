from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from backend.platform.memory.base.session_store import SessionTurn


class PromptContextBuilder:
    """负责按窗口大小裁剪轮次或 LangChain message 上下文。"""

    def __init__(self, window_size: int = 10, max_snippet_chars: int = 220) -> None:
        """初始化上下文裁剪器的窗口参数。"""
        self.window_size = window_size
        self.max_snippet_chars = max_snippet_chars

    def trim_turns(self, turns: Sequence[SessionTurn]) -> list[SessionTurn]:
        """仅保留窗口大小内的最近对话轮次。"""
        if len(turns) <= self.window_size:
            return list(turns)
        return list(turns[-self.window_size :])

    def trim_messages(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        """按轮次窗口裁剪 LangChain message，默认保留最近 N 轮对话。"""
        message_window = self.window_size * 2
        if len(messages) <= message_window:
            return list(messages)
        return list(messages[-message_window:])

    def turns_to_messages(self, turns: Sequence[SessionTurn]) -> list[BaseMessage]:
        """将现有轮次结构展开为 LangChain message 列表。"""
        messages: list[BaseMessage] = []
        for turn in self.trim_turns(turns):
            messages.extend(turn.to_messages())
        return messages

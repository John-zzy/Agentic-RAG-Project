from __future__ import annotations

from collections.abc import Sequence

from backend.knowledge._text_utils import truncate_snippet
from backend.memory.session_store import SessionTurn


class PromptContextBuilder:
    def __init__(self, window_size: int = 10, max_snippet_chars: int = 220) -> None:
        """初始化上下文裁剪器的窗口参数。"""
        self.window_size = window_size
        self.max_snippet_chars = max_snippet_chars

    def trim_turns(self, turns: Sequence[SessionTurn]) -> list[SessionTurn]:
        """仅保留窗口大小内的最近对话轮次。"""
        if len(turns) <= self.window_size:
            return list(turns)
        return list(turns[-self.window_size :])

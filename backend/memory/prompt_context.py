from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from backend.memory.session_store import SessionTurn


class PromptContextBuilder:
    def __init__(self, window_size: int = 10, max_snippet_chars: int = 220) -> None:
        self.window_size = window_size
        self.max_snippet_chars = max_snippet_chars

    def trim_turns(self, turns: Sequence[SessionTurn]) -> list[SessionTurn]:
        if len(turns) <= self.window_size:
            return list(turns)
        return list(turns[-self.window_size :])

    def build_prompt(
        self,
        user_message: str,
        history_turns: Sequence[SessionTurn],
        retrieval_snippets: Sequence[dict[str, Any]],
    ) -> str:
        trimmed_turns = self.trim_turns(history_turns)
        lines: list[str] = [
            "你是电商客服助手。请基于知识片段优先回答，缺少依据时明确说明不确定性。",
            "[History]",
        ]

        if not trimmed_turns:
            lines.append("(empty)")
        else:
            for turn in trimmed_turns:
                lines.append(f"User: {turn.user_message}")
                lines.append(f"Assistant: {turn.assistant_answer}")

        lines.append("[RetrievedKnowledge]")
        if not retrieval_snippets:
            lines.append("(empty)")
        else:
            for snippet in retrieval_snippets:
                citation_id = str(snippet.get("citation_id", "unknown"))
                namespace = str(snippet.get("namespace", "knowledge"))
                content = self._truncate(str(snippet.get("snippet", "")))
                lines.append(f"- [{namespace}:{citation_id}] {content}")

        lines.append("[UserMessage]")
        lines.append(user_message.strip())
        lines.append("[Answer]")
        return "\n".join(lines)

    def _truncate(self, text: str) -> str:
        normalized = text.replace("\n", " ").strip()
        if len(normalized) <= self.max_snippet_chars:
            return normalized
        return f"{normalized[: self.max_snippet_chars]}..."


from __future__ import annotations

MAX_SNIPPET_LENGTH: int = 220


def truncate_snippet(text: str, max_length: int = MAX_SNIPPET_LENGTH) -> str:
    """清理并截断文本片段，避免返回过长内容。"""
    normalized = text.replace("\n", " ").strip()
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[:max_length]}..."

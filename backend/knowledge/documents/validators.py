from __future__ import annotations

import re
from pathlib import Path


NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_namespace(namespace: str) -> str:
    """校验知识文档命名空间。"""
    if not namespace or namespace.strip() != namespace:
        raise ValueError("namespace must be a non-empty slug.")
    if ".." in namespace or "/" in namespace or "\\" in namespace:
        raise ValueError("namespace must not contain path separators or '..'.")
    if not NAMESPACE_PATTERN.fullmatch(namespace):
        raise ValueError("namespace must contain only lowercase letters, numbers, and underscores.")
    return namespace


def validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    """校验分块参数。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def validate_source_path(source_path: str | Path, data_root: str | Path) -> str:
    """校验源文件位于 data_root 下，并返回正斜杠相对路径。"""
    root = Path(data_root).resolve()
    raw_path = Path(source_path)
    raw_text = str(source_path)
    if ".." in raw_path.parts or raw_text.startswith(("..\\", "../")):
        raise ValueError("source_path must stay under data_root.")

    resolved_path = raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()
    try:
        relative_path = resolved_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("source_path must stay under data_root.") from exc
    return relative_path.as_posix()

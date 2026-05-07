from __future__ import annotations

import re
from pathlib import Path


NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_namespace(namespace: str) -> str:
    """校验文档管理命名空间，避免路径字符和异常标点进入索引元数据。"""
    if not namespace or namespace.strip() != namespace:
        raise ValueError("namespace must be a non-empty slug.")
    if ".." in namespace or "/" in namespace or "\\" in namespace:
        raise ValueError("namespace must not contain path separators or '..'.")
    if not NAMESPACE_PATTERN.fullmatch(namespace):
        raise ValueError("namespace must contain only lowercase letters, numbers, and underscores.")
    return namespace


def validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    """校验分块参数，确保窗口可以稳定向前推进。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def validate_source_path(source_path: str | Path, data_root: str | Path) -> str:
    """校验 JSON 源文件位于数据根目录内，并返回正斜杠相对路径。"""
    root = Path(data_root).resolve()
    raw_path = Path(source_path)
    raw_text = str(source_path)
    if ".." in raw_path.parts or raw_text.startswith(("..\\", "../")):
        raise ValueError("source_path must stay under data_root.")
    if raw_path.suffix.lower() != ".json":
        raise ValueError("source_path must be a JSON file.")

    resolved_path = raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()
    try:
        relative_path = resolved_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("source_path must stay under data_root.") from exc
    return relative_path.as_posix()

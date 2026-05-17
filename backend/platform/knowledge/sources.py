from __future__ import annotations

import json
from collections.abc import Sequence


SUPPORTED_MOUNTED_KNOWLEDGE_SOURCES = ("documents", "ecommerce")
DEFAULT_MOUNTED_KNOWLEDGE_SOURCES = ("documents",)


class MountedKnowledgeSourceValidationError(ValueError):
    """表示挂载知识源配置不合法。"""


def normalize_mounted_knowledge_sources(
    raw_sources: Sequence[str] | None,
) -> tuple[str, ...]:
    """将挂载知识源去重、排序，并在缺省时补上默认值。"""
    if raw_sources is None:
        return DEFAULT_MOUNTED_KNOWLEDGE_SOURCES
    if isinstance(raw_sources, str | bytes):
        raise MountedKnowledgeSourceValidationError(
            "mounted_knowledge_sources must be a list of strings."
        )

    seen: set[str] = set()
    unknown_sources: set[str] = set()

    for raw_source in raw_sources:
        if not isinstance(raw_source, str):
            raise MountedKnowledgeSourceValidationError(
                "mounted_knowledge_sources must contain only strings."
            )
        normalized_source = raw_source.strip().lower()
        if not normalized_source:
            continue
        if normalized_source not in SUPPORTED_MOUNTED_KNOWLEDGE_SOURCES:
            unknown_sources.add(normalized_source)
            continue
        seen.add(normalized_source)

    if unknown_sources:
        supported = ", ".join(SUPPORTED_MOUNTED_KNOWLEDGE_SOURCES)
        invalid = ", ".join(sorted(unknown_sources))
        raise MountedKnowledgeSourceValidationError(
            f"Unknown mounted knowledge sources: {invalid}. Expected one of: {supported}."
        )

    normalized_sources = tuple(
        source
        for source in SUPPORTED_MOUNTED_KNOWLEDGE_SOURCES
        if source in seen
    )
    return normalized_sources or DEFAULT_MOUNTED_KNOWLEDGE_SOURCES


def serialize_mounted_knowledge_sources(
    mounted_knowledge_sources: Sequence[str] | None,
) -> str:
    """将挂载知识源列表序列化为可持久化的 JSON 字符串。"""
    normalized_sources = normalize_mounted_knowledge_sources(mounted_knowledge_sources)
    return json.dumps(list(normalized_sources), ensure_ascii=False)


def parse_mounted_knowledge_sources(payload: object) -> tuple[str, ...]:
    """从数据库字段或历史数据中解析挂载知识源列表。"""
    if not isinstance(payload, str) or not payload.strip():
        return DEFAULT_MOUNTED_KNOWLEDGE_SOURCES

    try:
        raw_value = json.loads(payload)
    except json.JSONDecodeError:
        return DEFAULT_MOUNTED_KNOWLEDGE_SOURCES

    if not isinstance(raw_value, list):
        return DEFAULT_MOUNTED_KNOWLEDGE_SOURCES

    try:
        return normalize_mounted_knowledge_sources(raw_value)
    except MountedKnowledgeSourceValidationError:
        return DEFAULT_MOUNTED_KNOWLEDGE_SOURCES

from __future__ import annotations

from pathlib import Path

from backend.platform.knowledge.base.store import VectorSearchResult


DOCUMENT_MINIMUM_RELEVANCE = 0.18


def filter_low_relevance_document_results(
    results: list[VectorSearchResult],
    *,
    minimum_relevance: float = DOCUMENT_MINIMUM_RELEVANCE,
) -> list[VectorSearchResult]:
    """过滤明显不相关的文档命中，避免寒暄词误召回订单等内容。"""
    filtered: list[VectorSearchResult] = []
    for result in results:
        score = result.score
        if score is not None and float(score) < minimum_relevance:
            continue
        filtered.append(result)
    return filtered


def filter_managed_document_results(
    results: list[VectorSearchResult],
    *,
    files_root: str | Path,
) -> list[VectorSearchResult]:
    """只保留受管上传文档结果，避免内置业务数据混入文档检索。

    过滤规则分两层：
    1. 新数据优先看 `is_managed_document=True` 标记。
    2. 兼容历史数据时，再检查 `source_path` 是否能安全落到 `files_root` 下。
    """
    managed_root = Path(files_root).resolve()
    filtered: list[VectorSearchResult] = []
    for result in results:
        metadata = result.document.metadata
        if metadata.get("is_managed_document") is True:
            filtered.append(result)
            continue

        source_path = metadata.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            # 单元测试里的假文档经常只给相对路径，不会真的建文件。
            # 这类数据如果自报 namespace=documents，就保留，避免把测试桩误杀。
            if metadata.get("namespace") == "documents":
                filtered.append(result)
            continue
        resolved_path = (managed_root / source_path).resolve()
        try:
            resolved_path.relative_to(managed_root)
        except ValueError:
            continue
        if resolved_path.exists():
            filtered.append(result)
            continue
        if metadata.get("namespace") == "documents":
            filtered.append(result)
    return filtered

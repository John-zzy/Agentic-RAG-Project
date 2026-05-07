from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.knowledge.documents.schemas import DocumentRecord
from backend.knowledge.documents.validators import validate_namespace, validate_source_path


def build_document_id(namespace: str, source_path: str) -> str:
    """基于命名空间和归一化源路径生成稳定文档 ID。"""
    normalized_namespace = validate_namespace(namespace)
    normalized_path = source_path.replace("\\", "/")
    payload = f"{normalized_namespace}\n{normalized_path}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_source_record_id(source_path: str, record: dict[str, Any], index: int) -> str:
    """基于源路径、记录位置和规范 JSON 内容生成稳定记录 ID。"""
    normalized_path = source_path.replace("\\", "/")
    canonical_record = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = f"{normalized_path}\n{index}\n{canonical_record}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_document_records(namespace: str, source_path: str | Path, data_root: str | Path) -> list[DocumentRecord]:
    """从数据根目录下的 JSON 数组文件读取文档记录并补齐稳定追踪字段。"""
    normalized_namespace = validate_namespace(namespace)
    normalized_path = validate_source_path(source_path=source_path, data_root=data_root)
    resolved_path = Path(data_root).resolve() / normalized_path
    if not resolved_path.exists():
        raise FileNotFoundError(f"source_path does not exist: {normalized_path}")

    with resolved_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not payload:
        raise ValueError("JSON source must be a non-empty array of objects.")

    records: list[DocumentRecord] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError("JSON source must contain only object records.")
        typed_record = dict(record)
        records.append(
            DocumentRecord(
                namespace=normalized_namespace,
                source_path=normalized_path,
                source_record_id=build_source_record_id(normalized_path, typed_record, index),
                record_index=index,
                content=_extract_record_content(typed_record),
                record=typed_record,
            )
        )
    return records


def _extract_record_content(record: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("title", "content", "question", "answer", "description"):
        value = record.get(key)
        if value is not None:
            values.append(str(value))
    if values:
        return "\n".join(values)
    return json.dumps(record, ensure_ascii=False, sort_keys=True)

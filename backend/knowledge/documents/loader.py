from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from backend.knowledge.documents.schemas import DocumentRecord
from backend.knowledge.documents.validators import validate_namespace, validate_source_path

TEXT_SOURCE_EXTENSIONS = {".txt", ".md"}


def build_document_id(namespace: str, source_path: str) -> str:
    """基于命名空间和归一化源路径生成稳定文档 ID。"""
    normalized_namespace = validate_namespace(namespace)
    normalized_path = source_path.replace("\\", "/")
    payload = f"{normalized_namespace}\n{normalized_path}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_source_record_id(source_path: str, record: dict[str, Any], index: int) -> str:
    """基于源路径、记录位置和规范化内容生成稳定记录 ID。"""
    normalized_path = source_path.replace("\\", "/")
    canonical_record = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = f"{normalized_path}\n{index}\n{canonical_record}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_document_records(namespace: str, source_path: str | Path, data_root: str | Path) -> list[DocumentRecord]:
    """根据文件类型读取知识文档记录。"""
    normalized_namespace = validate_namespace(namespace)
    normalized_path = validate_source_path(source_path=source_path, data_root=data_root)
    resolved_path = Path(data_root).resolve() / normalized_path
    if not resolved_path.exists():
        raise FileNotFoundError(f"source_path does not exist: {normalized_path}")

    suffix = resolved_path.suffix.lower()
    if suffix == ".json":
        return _load_json_records(normalized_namespace, normalized_path, resolved_path)
    if suffix in TEXT_SOURCE_EXTENSIONS:
        return _load_text_record(normalized_namespace, normalized_path, resolved_path)
    if suffix == ".csv":
        return _load_csv_records(normalized_namespace, normalized_path, resolved_path)
    raise ValueError(f"Unsupported source file type: {suffix}")


def _load_json_records(namespace: str, normalized_path: str, resolved_path: Path) -> list[DocumentRecord]:
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
                namespace=namespace,
                source_path=normalized_path,
                source_record_id=build_source_record_id(normalized_path, typed_record, index),
                record_index=index,
                content=_extract_record_content(typed_record),
                record=typed_record,
            )
        )
    return records


def _load_text_record(namespace: str, normalized_path: str, resolved_path: Path) -> list[DocumentRecord]:
    content = resolved_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError("Text source must not be empty.")
    record = {
        "title": resolved_path.stem,
        "content": content,
        "source_type": resolved_path.suffix.lower().lstrip("."),
    }
    return [
        DocumentRecord(
            namespace=namespace,
            source_path=normalized_path,
            source_record_id=build_source_record_id(normalized_path, record, 0),
            record_index=0,
            content=content,
            record=record,
        )
    ]


def _load_csv_records(namespace: str, normalized_path: str, resolved_path: Path) -> list[DocumentRecord]:
    with resolved_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("CSV source must contain at least one data row.")

    records: list[DocumentRecord] = []
    for index, row in enumerate(rows):
        normalized_row = {str(key): "" if value is None else str(value) for key, value in row.items()}
        records.append(
            DocumentRecord(
                namespace=namespace,
                source_path=normalized_path,
                source_record_id=build_source_record_id(normalized_path, normalized_row, index),
                record_index=index,
                content=_extract_record_content(normalized_row),
                record=normalized_row,
            )
        )
    return records


def _extract_record_content(record: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("title", "content", "question", "answer", "description"):
        value = record.get(key)
        if value is not None and str(value).strip():
            values.append(str(value))
    if values:
        return "\n".join(values)
    return json.dumps(record, ensure_ascii=False, sort_keys=True)

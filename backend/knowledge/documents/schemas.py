from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """表示从文档源文件中读取的一条原始业务记录。"""

    namespace: str
    source_path: str
    source_record_id: str
    record_index: int
    content: str
    record: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """表示可写入向量库的文档切片及其追踪元数据。"""

    chunk_id: str
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

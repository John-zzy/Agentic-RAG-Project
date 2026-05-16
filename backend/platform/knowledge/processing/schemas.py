from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ProcessingRuleId = Literal[
    "trim_whitespace",
    "drop_empty_records",
    "dedupe_records",
    "strip_html_tags",
    "remove_markdown_boilerplate",
    "remove_url_lines",
]
ProcessingSourceType = Literal["json", "csv", "txt", "md", "pdf", "docx", "xlsx"]
ProcessingRuleLevel = Literal["record", "document"]
ProcessingWarningSeverity = Literal["info", "warning", "error"]


class ProcessingRuleDefinition(BaseModel):
    """预设处理规则定义。"""

    rule_id: ProcessingRuleId
    display_name: str
    description: str
    supported_source_types: list[ProcessingSourceType] = Field(default_factory=list)
    level: ProcessingRuleLevel = "record"


class ProcessingWarning(BaseModel):
    """预处理阶段的结构化警告。"""

    code: str
    message: str
    severity: ProcessingWarningSeverity = "warning"
    source_record_id: str | None = None
    record_index: int | None = None


class ProcessingStats(BaseModel):
    """处理统计摘要。"""

    raw_record_count: int = 0
    processed_record_count: int = 0
    removed_record_count: int = 0
    raw_char_count: int = 0
    processed_char_count: int = 0


class ProcessingSample(BaseModel):
    """预览中的样本记录。"""

    sample_index: int
    source_record_id: str
    record_index: int
    content: str
    content_hash: str
    applied_rules: list[ProcessingRuleId] = Field(default_factory=list)
    dropped: bool = False


class ProcessedDocumentRecord(BaseModel):
    """处理后的标准化记录。"""

    namespace: str
    source_path: str
    source_type: ProcessingSourceType
    source_record_id: str
    record_index: int
    raw_content: str
    processed_content: str
    applied_rules: list[ProcessingRuleId] = Field(default_factory=list)
    raw_content_hash: str
    processed_content_hash: str
    dropped: bool = False
    record: dict[str, object] = Field(default_factory=dict)
    warnings: list[ProcessingWarning] = Field(default_factory=list)


class ProcessedDocumentResult(BaseModel):
    """处理流水线输出，供预览和正式入库共用。"""

    namespace: str
    source_path: str
    source_type: ProcessingSourceType
    processing_rules: list[ProcessingRuleId] = Field(default_factory=list)
    processing_stats: ProcessingStats
    records: list[ProcessedDocumentRecord] = Field(default_factory=list)
    warnings: list[ProcessingWarning] = Field(default_factory=list)
    can_index: bool = True


class PreprocessPreview(BaseModel):
    """预处理预览响应。"""

    namespace: str
    source_path: str
    source_type: ProcessingSourceType
    chunk_size: int
    chunk_overlap: int
    supported_rules: list[ProcessingRuleDefinition] = Field(default_factory=list)
    selected_rules: list[ProcessingRuleDefinition] = Field(default_factory=list)
    processing_stats: ProcessingStats | None = None
    original_samples: list[ProcessingSample] = Field(default_factory=list)
    processed_samples: list[ProcessingSample] = Field(default_factory=list)
    can_index: bool = True
    warnings: list[ProcessingWarning] = Field(default_factory=list)

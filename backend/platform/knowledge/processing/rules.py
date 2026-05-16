from __future__ import annotations

from backend.platform.knowledge.processing.schemas import (
    ProcessingRuleDefinition,
    ProcessingRuleId,
    ProcessingSourceType,
)

SUPPORTED_PROCESSING_SOURCE_TYPES: tuple[ProcessingSourceType, ...] = ("json", "csv", "txt", "md")

PROCESSING_RULES: dict[ProcessingRuleId, ProcessingRuleDefinition] = {
    "trim_whitespace": ProcessingRuleDefinition(
        rule_id="trim_whitespace",
        display_name="Trim Whitespace",
        description="去除首尾空白并规范化多余空行。",
        supported_source_types=list(SUPPORTED_PROCESSING_SOURCE_TYPES),
        level="record",
    ),
    "drop_empty_records": ProcessingRuleDefinition(
        rule_id="drop_empty_records",
        display_name="Drop Empty Records",
        description="丢弃内容为空或只有空白字符的记录。",
        supported_source_types=list(SUPPORTED_PROCESSING_SOURCE_TYPES),
        level="record",
    ),
    "dedupe_records": ProcessingRuleDefinition(
        rule_id="dedupe_records",
        display_name="Dedupe Records",
        description="基于规范化内容去除重复记录。",
        supported_source_types=list(SUPPORTED_PROCESSING_SOURCE_TYPES),
        level="document",
    ),
    "strip_html_tags": ProcessingRuleDefinition(
        rule_id="strip_html_tags",
        display_name="Strip HTML Tags",
        description="移除文本中的 HTML 标签与脚本片段。",
        supported_source_types=list(SUPPORTED_PROCESSING_SOURCE_TYPES),
        level="record",
    ),
    "remove_markdown_boilerplate": ProcessingRuleDefinition(
        rule_id="remove_markdown_boilerplate",
        display_name="Remove Markdown Boilerplate",
        description="移除 Markdown 标题模板、目录和常见模板噪声。",
        supported_source_types=["md"],
        level="document",
    ),
    "remove_url_lines": ProcessingRuleDefinition(
        rule_id="remove_url_lines",
        display_name="Remove URL Lines",
        description="移除主要由 URL 组成的噪声行。",
        supported_source_types=list(SUPPORTED_PROCESSING_SOURCE_TYPES),
        level="record",
    ),
}


def get_processing_rule(rule_id: ProcessingRuleId) -> ProcessingRuleDefinition:
    return PROCESSING_RULES[rule_id]


def list_processing_rules(source_type: str | None = None) -> list[ProcessingRuleDefinition]:
    rules = list(PROCESSING_RULES.values())
    if source_type is None:
        return rules
    normalized_source_type = source_type.lower()
    return [rule for rule in rules if normalized_source_type in rule.supported_source_types]


def get_supported_rule_ids(source_type: str | None = None) -> list[ProcessingRuleId]:
    return [rule.rule_id for rule in list_processing_rules(source_type)]

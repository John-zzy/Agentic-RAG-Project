from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


ToolCapabilityType = Literal["retrieval", "action"]


class ToolContext(BaseModel):
    """封装工具调用上下文，便于 runtime 和审计层透传元数据。"""

    session_id: str | None = None
    request_id: str | None = None
    agent_name: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """统一的工具调用结果结构。"""

    records: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    tool_name: str
    success: bool
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @classmethod
    def ok(
        cls,
        *,
        tool_name: str,
        records: list[dict[str, Any]] | None = None,
        citations: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            tool_name=tool_name,
            success=True,
            records=records or [],
            citations=citations or [],
            confidence=confidence,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
        cls,
        *,
        tool_name: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            tool_name=tool_name,
            success=False,
            error=error,
            metadata=metadata or {},
        )


def build_structured_tool(
    *,
    name: str,
    description: str,
    capability_type: ToolCapabilityType,
    args_schema: type[BaseModel],
    func: Any,
) -> BaseTool:
    """构建结构化工具，作为平台层的统一工具协议入口。"""
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        args_schema=args_schema,
        metadata={"capability_type": capability_type},
    )


def get_tool_definition(tool: BaseTool) -> dict[str, Any]:
    """读取标准化工具定义。"""
    args_schema = getattr(tool, "args_schema", None)
    input_schema = (
        args_schema.model_json_schema()
        if isinstance(args_schema, type) and issubclass(args_schema, BaseModel)
        else None
    )
    return {
        "name": tool.name,
        "description": tool.description,
        "capability_type": (tool.metadata or {}).get("capability_type"),
        "input_schema": input_schema,
    }


@dataclass
class BaseJsonStore:
    """提供 JSON 文件读写的基础能力。"""

    data_dir: Path

    def _load_json_list(self, filename: str) -> list[dict[str, Any]]:
        path = self.data_dir / filename
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_json_list(self, filename: str, data: list[dict[str, Any]]) -> None:
        path = self.data_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "BaseJsonStore",
    "ToolCapabilityType",
    "ToolContext",
    "ToolResult",
    "build_structured_tool",
    "get_tool_definition",
]

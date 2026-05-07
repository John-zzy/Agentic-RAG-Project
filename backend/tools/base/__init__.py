from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


ToolCapabilityType = Literal["retrieval", "action"]


class ToolContext(BaseModel):
    """封装一次工具调用的上下文信息，供工具记录来源与调用环境。"""

    session_id: str | None = None
    request_id: str | None = None
    agent_name: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """统一描述工具调用结果，供 Agent、MCP 适配层和审计层复用。"""

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
        """构造成功结果，避免调用方重复拼装标准字段。"""
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
        """构造失败结果，统一错误返回结构。"""
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
    """构建可被 LangChain 或 LangGraph 直接消费的 StructuredTool。"""
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        args_schema=args_schema,
        metadata={"capability_type": capability_type},
    )


def get_tool_definition(tool: BaseTool) -> dict[str, Any]:
    """读取 LangChain tool 的注册信息，兼容现有注册表和协议适配层。"""
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
    """基础 JSON 文件存储类，提供通用读写能力，供各业务存储类复用。"""

    data_dir: Path

    def _load_json_list(self, filename: str) -> list[dict[str, Any]]:
        """读取 JSON 文件内容，文件不存在时返回空列表。"""
        path = self.data_dir / filename
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_json_list(self, filename: str, data: list[dict[str, Any]]) -> None:
        """持久化数据到 JSON 文件，自动创建父目录。"""
        path = self.data_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

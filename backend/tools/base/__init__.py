from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

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


class AgentTool(ABC):
    """所有 Agent 可调用工具的抽象基类，统一输入校验与元数据输出。"""

    name: str
    description: str
    capability_type: ToolCapabilityType
    input_model: type[BaseModel]

    def parse_input(self, tool_input: BaseModel | dict[str, Any]) -> BaseModel:
        """将原始输入标准化为工具声明的输入模型实例。"""
        if isinstance(tool_input, self.input_model):
            return tool_input
        if isinstance(tool_input, BaseModel):
            return self.input_model.model_validate(tool_input.model_dump())
        return self.input_model.model_validate(tool_input)

    def definition(self) -> dict[str, Any]:
        """返回工具的注册定义，供注册表或协议适配层读取。"""
        return {
            "name": self.name,
            "description": self.description,
            "capability_type": self.capability_type,
            "input_schema": self.input_model.model_json_schema(),
        }

    @abstractmethod
    def invoke(
        self,
        tool_input: BaseModel | dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """执行具体工具逻辑并返回统一的 ToolResult。"""
        raise NotImplementedError

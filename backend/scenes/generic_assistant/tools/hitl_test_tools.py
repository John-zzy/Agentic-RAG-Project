from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.platform.tools import SceneTool, ToolResult


class GenericHitlFakeWriteArgs(BaseModel):
    """generic HITL 写操作测试工具的入参。"""

    item_id: str = Field(description="要写入的测试对象 ID。")
    content: str = Field(description="要写入的测试内容。")


class GenericHitlFakeExternalApiArgs(BaseModel):
    """generic HITL 外部 API 测试工具的入参。"""

    endpoint: str = Field(description="测试用外部 API 地址。")
    payload: dict[str, Any] = Field(default_factory=dict, description="测试用请求内容。")


class GenericHitlFakeWriteTool(SceneTool):
    """只用于验证审批流程的 generic 写工具，不会被默认聊天主链调用。"""

    name = "generic_hitl_fake_write"
    description = "测试用 generic 写操作工具，必须在 approve 后才能执行。"
    capability_type = "action"
    args_schema = GenericHitlFakeWriteArgs

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, **kwargs: Any) -> ToolResult:
        """记录一次写操作调用，用于测试 approve 前不会执行。"""
        args = GenericHitlFakeWriteArgs(**kwargs)
        call = args.model_dump()
        self.calls.append(call)
        return ToolResult.ok(
            tool_name=self.name,
            records=[{"item_id": args.item_id, "content": args.content}],
            metadata={"side_effect": "local_write"},
        )


class GenericHitlFakeExternalApiTool(SceneTool):
    """只用于验证审批流程的 generic 外部 API 工具，不接真实外部服务。"""

    name = "generic_hitl_fake_external_api"
    description = "测试用 generic 外部 API 工具，必须在 approve 后才能执行。"
    capability_type = "action"
    args_schema = GenericHitlFakeExternalApiArgs

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, **kwargs: Any) -> ToolResult:
        """记录一次外部 API 调用，用于测试 reject 后不会执行。"""
        args = GenericHitlFakeExternalApiArgs(**kwargs)
        call = args.model_dump()
        self.calls.append(call)
        return ToolResult.ok(
            tool_name=self.name,
            records=[{"endpoint": args.endpoint, "payload": args.payload}],
            metadata={"side_effect": "external_api"},
        )

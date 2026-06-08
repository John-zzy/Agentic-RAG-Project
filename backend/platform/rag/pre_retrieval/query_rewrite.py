from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.platform.rag.contracts import RetrievalContext


class QueryRewrite(BaseModel):
    """描述查询改写输出，供后续轮次继续检索使用。"""

    query: str
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """结构化模型输出必须给出可检索 query，空白 query 不进入检索链路。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty.")
        return normalized


class QueryRewriter(RunnableSerializable[RetrievalContext, QueryRewrite], ABC):
    """定义查询改写契约，兼容 LangChain Runnable 调用约定。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def invoke(
        self,
        input: RetrievalContext,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> QueryRewrite:
        """根据当前状态生成下一轮检索查询。"""
        raise NotImplementedError

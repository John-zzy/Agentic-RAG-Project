from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from backend.platform.rag.contracts import RecallStrategy, RetrievalResult, RetrievalTool
from backend.platform.tools.base import SceneTool


def build_scene_structured_tool(tool: SceneTool) -> BaseTool:
    """把场景工具类适配为 LangChain StructuredTool。"""
    return StructuredTool.from_function(
        func=tool.invoke,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        metadata={
            "capability_type": tool.capability_type,
            "tool_class": type(tool).__name__,
        },
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


class RetrievalToolAdapter(RetrievalTool):
    """把独立工具类适配到 Agentic Retrieval 的 RetrievalTool 协议。"""

    name: str
    description: str
    tool: Any = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def retrieve(
        self,
        query: str,
        *,
        run_manager: Any | None = None,
        top_k: int | None = None,
        min_relevance_score: float | None = None,
        recall_strategy: RecallStrategy = "hybrid",
        rerank_enabled: bool = False,
        rerank_top_n: int | None = None,
    ) -> RetrievalResult:
        return self.tool.retrieve(
            query=query,
            run_manager=run_manager,
            top_k=top_k,
            min_relevance_score=min_relevance_score,
            recall_strategy=recall_strategy,
            rerank_enabled=rerank_enabled,
            rerank_top_n=rerank_top_n,
        )


def build_retrieval_tool(tool: Any) -> RetrievalTool:
    """把具备 retrieve() 的工具类适配为 Agentic RetrievalTool。"""
    return RetrievalToolAdapter(
        name=tool.name,
        description=tool.description,
        tool=tool,
    )

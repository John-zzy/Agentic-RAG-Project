from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import BaseModel, ConfigDict, Field

from backend.platform.rag.contracts import RetrievalContext, RetrievalNextAction


class SufficiencyDecision(BaseModel):
    """描述当前证据是否足够，以及不足时下一步应该采取的动作。"""

    is_sufficient: bool
    reason: str
    next_action: RetrievalNextAction
    confidence: float | None = None
    suggested_tool: str | None = None
    follow_up_question: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalDecisionLogEntry(BaseModel):
    round_index: int
    tool_name: str
    query: str
    rewritten_query: str | None = None
    result_count: int = 0
    result_success: bool
    result_confidence: float | None = None
    decision: RetrievalNextAction
    is_sufficient: bool
    reason: str
    suggested_tool: str | None = None
    follow_up_question: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SufficiencyJudge(RunnableSerializable[RetrievalContext, SufficiencyDecision], ABC):
    """定义证据充分性判断契约，兼容 LangChain Runnable 调用约定。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def invoke(
        self,
        input: RetrievalContext,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> SufficiencyDecision:
        """根据当前计划与累计结果判断是否继续检索。"""
        raise NotImplementedError

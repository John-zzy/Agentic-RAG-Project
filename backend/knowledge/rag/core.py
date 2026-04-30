from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field


RetrievalNextAction = Literal["finish", "rewrite", "switch_tool", "ask_user"]


class RetrievalRecord(BaseModel):
    """描述一次检索命中的标准化记录，供不同检索工具统一返回。"""

    record_id: str
    content: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalCitation(BaseModel):
    """描述回答可引用的证据片段，便于后续答案生成与调试复用。"""

    citation_id: str
    snippet: str
    source_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """统一描述一次检索工具执行结果，屏蔽底层检索源实现差异。"""

    tool_name: str
    query: str
    success: bool
    records: list[RetrievalRecord] = Field(default_factory=list)
    citations: list[RetrievalCitation] = Field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @classmethod
    def ok(
        cls,
        *,
        tool_name: str,
        query: str,
        records: Sequence[RetrievalRecord] | None = None,
        citations: Sequence[RetrievalCitation] | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RetrievalResult":
        """构造成功结果，避免调用方重复拼装标准字段。"""
        return cls(
            tool_name=tool_name,
            query=query,
            success=True,
            records=list(records or []),
            citations=list(citations or []),
            confidence=confidence,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
        cls,
        *,
        tool_name: str,
        query: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> "RetrievalResult":
        """构造失败结果，保留统一错误结构供编排器判断。"""
        return cls(
            tool_name=tool_name,
            query=query,
            success=False,
            error=error,
            metadata=metadata or {},
        )


class RetrievalPlan(BaseModel):
    """描述一轮 Agentic Retrieval 的输入、状态和可选工具集合。"""

    user_query: str
    active_query: str
    selected_tool: str
    round_index: int = Field(default=1, ge=1)
    max_rounds: int = Field(default=3, ge=1)
    candidate_tools: tuple[str, ...] = ()
    attempted_tools: tuple[str, ...] = ()
    previous_queries: tuple[str, ...] = ()
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def create_followup(
        self,
        *,
        active_query: str | None = None,
        selected_tool: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RetrievalPlan":
        """基于当前计划生成下一轮检索计划，供编排器有界重试时复用。"""
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return self.model_copy(
            update={
                "active_query": active_query or self.active_query,
                "selected_tool": selected_tool or self.selected_tool,
                "round_index": self.round_index + 1,
                "attempted_tools": (*self.attempted_tools, self.selected_tool),
                "previous_queries": (*self.previous_queries, self.active_query),
                "metadata": merged_metadata,
            }
        )


class SufficiencyDecision(BaseModel):
    """描述当前证据是否足够，以及不足时下一步应该采取的动作。"""

    is_sufficient: bool
    reason: str
    next_action: RetrievalNextAction
    confidence: float | None = None
    suggested_tool: str | None = None
    follow_up_question: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRewrite(BaseModel):
    """描述查询改写输出，供后续轮次继续检索使用。"""

    query: str
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTool(ABC):
    """定义通用检索工具契约，使不同检索源可接入统一编排流程。"""

    name: str
    description: str

    def definition(self) -> dict[str, Any]:
        """返回工具元数据，便于注册表或编排器读取。"""
        return {"name": self.name, "description": self.description}

    @abstractmethod
    def retrieve(self, plan: RetrievalPlan) -> RetrievalResult:
        """执行具体检索逻辑，并返回标准化检索结果。"""
        raise NotImplementedError


class SufficiencyJudge(ABC):
    """定义证据充分性判断契约，隔离领域策略与编排流程。"""

    @abstractmethod
    def judge(
        self,
        plan: RetrievalPlan,
        results: Sequence[RetrievalResult],
    ) -> SufficiencyDecision:
        """根据当前计划与累计结果判断是否继续检索。"""
        raise NotImplementedError


class QueryRewriter(ABC):
    """定义查询改写契约，用于证据不足时生成下一轮检索查询。"""

    @abstractmethod
    def rewrite(
        self,
        plan: RetrievalPlan,
        results: Sequence[RetrievalResult],
        decision: SufficiencyDecision | None = None,
    ) -> QueryRewrite:
        """根据当前计划、已有结果和判断结论生成改写查询。"""
        raise NotImplementedError

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import BaseModel, ConfigDict, Field


RetrievalNextAction = Literal["finish", "rewrite", "switch_tool", "ask_user"]


class RetrievalCitation(BaseModel):
    """描述回答可引用的证据片段，便于后续答案生成与调试复用。"""

    citation_id: str
    snippet: str
    source_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """统一描述一次检索工具执行结果，兼容 LangChain Document 输出。"""

    tool_name: str
    query: str
    records: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    citations: list[RetrievalCitation] = Field(default_factory=list)
    success: bool
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def ok(
        cls,
        *,
        tool_name: str,
        query: str,
        records: list[dict[str, Any]] | None = None,
        documents: list[Document] | None = None,
        citations: list[RetrievalCitation] | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RetrievalResult":
        """构造成功结果，避免调用方重复拼装标准字段。"""
        return cls(
            tool_name=tool_name,
            query=query,
            records=list(records or []),
            documents=list(documents or []),
            citations=list(citations or []),
            success=True,
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
    """描述当前检索轮次的输入、状态和可选工具集合。"""

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
        """基于当前计划生成下一轮检索计划，供有界重试时复用。"""
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


class RetrievalContext(BaseModel):
    """描述 Agentic Retrieval 当前状态，作为 LangChain runnable 的统一输入。"""

    plan: RetrievalPlan
    results: list[RetrievalResult] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RetrievalDecisionLogEntry(BaseModel):
    """缁熶竴璁板綍 Agentic Retrieval 姣忚疆鍐崇瓥锛屼究浜庤皟璇曞拰鍚庣画鍥炴斁銆?"""

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


class RetrievalTool(BaseRetriever, ABC):
    """定义 LangChain 风格的检索工具契约。"""

    name: str
    description: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def definition(self) -> dict[str, Any]:
        """返回工具元数据，便于注册表或编排器读取。"""
        return {"name": self.name, "description": self.description}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """适配 LangChain retriever 协议，直接返回 Document 列表。"""
        return self.retrieve(query=query, run_manager=run_manager).documents

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> RetrievalResult:
        """执行具体检索逻辑，并返回标准化检索结果。"""
        raise NotImplementedError


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

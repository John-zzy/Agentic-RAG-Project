from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from backend.platform.config.settings import AppSettings, settings
from backend.platform.knowledge.base.text import truncate_snippet
from backend.platform.knowledge.repositories import VectorStoreFactory
from backend.platform.rag.agentic import AgenticRetriever
from backend.platform.rag.core import (
    QueryRewrite,
    QueryRewriter,
    RetrievalCitation,
    RetrievalContext,
    RetrievalResult,
    RetrievalTool,
    SufficiencyDecision,
    SufficiencyJudge,
)
from backend.platform.rag.document_retrieval import (
    DocumentChunkRetrievalResult,
    DocumentRetrievalService,
)
from backend.platform.tools import ToolResult, build_structured_tool
from backend.scenes.base import (
    SceneBootstrapResult,
    SceneDefinition,
    SceneFallbackPolicy,
)


GENERIC_DOCUMENT_TOOL_NAME = "knowledge_document_search"
GENERIC_DOCUMENT_KNOWLEDGE_SOURCE = "documents"


GENERIC_ASSISTANT_SYSTEM_PROMPT = (
    "你是一名通用知识助手。"
    "请优先依据检索到的文档上下文回答问题，回答要清晰、克制。"
    "当证据不足时，明确说明不确定，并提示用户补充更具体的文档主题、术语或背景。"
)


class GenericKnowledgeDocumentSearchInput(BaseModel):
    """通用知识文档检索工具输入。"""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class GenericKnowledgeDocumentRetriever(BaseRetriever):
    """通用助手默认 retriever，只依赖文档知识库。"""

    document_retrieval_service: Any = Field(exclude=True)
    default_top_k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        """适配 LangChain retriever 协议。"""
        return self.search(query=query, top_k=self.default_top_k)

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """仅在已上传文档知识中检索证据。"""
        return self.document_retrieval_service.search(
            query=query,
            top_k=top_k or self.default_top_k,
        )


class GenericKnowledgeDocumentSearchTool(RetrievalTool):
    """通用知识文档检索工具，供 scene runtime 直接挂载。"""

    name: str = GENERIC_DOCUMENT_TOOL_NAME
    description: str = "Search semantically relevant uploaded knowledge documents."
    document_retrieval_service: Any = Field(exclude=True)
    default_top_k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def retrieve(self, query: str, *, run_manager: Any | None = None) -> RetrievalResult:
        """在上传文档分块中检索并返回标准化结果。"""
        del run_manager
        retrieval_results = self.document_retrieval_service.retrieve(query=query, top_k=self.default_top_k)
        records = [_build_document_record(result) for result in retrieval_results]
        citations = [
            RetrievalCitation(
                citation_id=record["citation_id"],
                snippet=record["snippet"],
                source_type=record["namespace"],
                metadata={
                    **record.get("metadata", {}),
                    "score": record.get("score"),
                    "vector_score": record.get("vector_score"),
                    "keyword_score": record.get("keyword_score"),
                    "vector_rank": record.get("vector_rank"),
                    "keyword_rank": record.get("keyword_rank"),
                    "matched_by": record.get("matched_by", []),
                },
            )
            for record in records
        ]
        documents = [
            Document(
                page_content=record["snippet"],
                metadata={
                    **record.get("metadata", {}),
                    "namespace": record["namespace"],
                    "citation_id": record["citation_id"],
                    "score": record.get("score"),
                    "vector_score": record.get("vector_score"),
                    "keyword_score": record.get("keyword_score"),
                    "vector_rank": record.get("vector_rank"),
                    "keyword_rank": record.get("keyword_rank"),
                    "matched_by": record.get("matched_by", []),
                },
            )
            for record in records
        ]
        confidence = _average_score(records)
        return RetrievalResult.ok(
            tool_name=self.name,
            query=query,
            records=records,
            documents=documents,
            citations=citations,
            confidence=confidence,
            metadata={"namespace": "documents", "result_count": len(records), "scene": "generic_assistant"},
        )


class GenericAssistantBusinessExtension(ABC):
    """generic scene 的业务扩展契约。"""

    knowledge_source: str

    def bootstrap(self) -> SceneBootstrapResult:
        """为扩展提供可选的预热入口，默认不执行任何初始化。"""
        return SceneBootstrapResult()

    @property
    @abstractmethod
    def retrieval_tool_names(self) -> tuple[str, ...]:
        """声明该扩展可暴露给 AgenticRetriever 的检索工具名。"""
        raise NotImplementedError

    @abstractmethod
    def build_retrieval_tools(self) -> tuple[RetrievalTool, ...]:
        """构建该扩展提供的 RetrievalTool 集合。"""
        raise NotImplementedError

    @abstractmethod
    def should_handoff(self, context: RetrievalContext) -> SufficiencyDecision | None:
        """决定 docs-first 主链是否应切换到该扩展。"""
        raise NotImplementedError

    @abstractmethod
    def resolve_followup(self, context: RetrievalContext) -> SufficiencyDecision | None:
        """在扩展接管后决定是否继续切换、改写或结束。"""
        raise NotImplementedError


class GenericAssistantSufficiencyJudge(SufficiencyJudge):
    """通用 docs-first 检索判断器。"""

    business_extensions: tuple[GenericAssistantBusinessExtension, ...] = Field(
        default_factory=tuple,
        exclude=True,
    )
    document_intent_keywords: tuple[str, ...] = (
        "文档",
        "说明",
        "手册",
        "指南",
        "faq",
        "知识库",
        "流程",
        "制度",
        "规则",
        "条款",
        "manual",
        "document",
        "docs",
        "guide",
        "policy",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def invoke(
        self,
        input: RetrievalContext,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> SufficiencyDecision:
        """优先评估文档证据，再按扩展顺序决定是否接管。"""
        del config, kwargs
        plan = input.plan
        result = input.results[-1]

        if plan.selected_tool == GENERIC_DOCUMENT_TOOL_NAME:
            handoff_decision = self._resolve_handoff(input)
            if handoff_decision is not None:
                return handoff_decision
            if result.records:
                return SufficiencyDecision(
                    is_sufficient=True,
                    next_action="finish",
                    reason="文档证据已足够支持当前回答。",
                    confidence=result.confidence,
                )
            return self._build_no_hit_decision(
                query=plan.user_query,
                round_index=plan.round_index,
                max_rounds=plan.max_rounds,
            )

        followup_decision = self._resolve_extension_followup(input)
        if followup_decision is not None:
            return followup_decision
        if result.records:
            return SufficiencyDecision(
                is_sufficient=True,
                next_action="finish",
                reason="当前证据已足够支持回答。",
                confidence=result.confidence,
            )
        return self._build_no_hit_decision(
            query=plan.user_query,
            round_index=plan.round_index,
            max_rounds=plan.max_rounds,
        )

    def _resolve_handoff(self, context: RetrievalContext) -> SufficiencyDecision | None:
        for extension in self.business_extensions:
            decision = extension.should_handoff(context)
            if decision is not None:
                return decision
        return None

    def _resolve_extension_followup(self, context: RetrievalContext) -> SufficiencyDecision | None:
        current_tool = context.plan.selected_tool
        for extension in self.business_extensions:
            if current_tool not in extension.retrieval_tool_names:
                continue
            decision = extension.resolve_followup(context)
            if decision is not None:
                return decision
        return None

    def _build_no_hit_decision(
        self,
        *,
        query: str,
        round_index: int,
        max_rounds: int,
    ) -> SufficiencyDecision:
        if not self._has_document_intent(query):
            return SufficiencyDecision(
                is_sufficient=False,
                next_action="ask_user",
                reason="当前问题缺少明确的文档查询意图，不进行文档查询改写。",
                follow_up_question="请补充更具体的文档主题、术语，或说明你希望查询的业务知识范围。",
            )
        if round_index >= max_rounds:
            return SufficiencyDecision(
                is_sufficient=False,
                next_action="ask_user",
                reason="在允许的检索轮次内没有找到足够相关的证据。",
                follow_up_question="请补充更具体的文档主题、术语，或说明你希望查询的业务知识范围。",
            )
        return SufficiencyDecision(
            is_sufficient=False,
            next_action="rewrite",
            reason="当前证据不足，先改写查询继续检索。",
        )

    def _has_document_intent(self, query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized:
            return False
        return any(keyword in normalized for keyword in self.document_intent_keywords)


class GenericAssistantQueryRewriter(QueryRewriter):
    """通用 docs-first 查询改写器。"""

    document_hint_keywords: tuple[str, ...] = (
        "文档",
        "说明",
        "手册",
        "指南",
        "faq",
        "知识库",
        "manual",
        "document",
        "docs",
        "guide",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def invoke(
        self,
        input: RetrievalContext,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> QueryRewrite:
        """把弱查询改写成更适合文档检索的中立表述。"""
        del config, kwargs
        query = input.plan.active_query.strip()
        lowered = query.lower()
        if any(keyword in lowered for keyword in self.document_hint_keywords):
            rewritten = f"{query} 相关内容"
        else:
            rewritten = f"{query} 相关文档 说明 手册 FAQ"
        return QueryRewrite(
            query=rewritten.strip(),
            reason="Broadened the query with generic document-oriented terms for the next retrieval round.",
            metadata={"original_query": query},
        )


def build_generic_assistant_scene_definition(
    app_settings: AppSettings | None = None,
    *,
    business_extensions: tuple[GenericAssistantBusinessExtension, ...] = (),
    document_retrieval_service: DocumentRetrievalService | None = None,
    max_rounds: int = 3,
) -> SceneDefinition:
    """构建通用知识助手场景定义。"""
    current_settings = app_settings or settings
    resolved_business_extensions = tuple(business_extensions)
    resolved_document_retrieval_service = document_retrieval_service or DocumentRetrievalService(
        app_settings=current_settings,
        vector_repository=VectorStoreFactory.create_document_chunk_vector_repository(current_settings),
        chunk_source=VectorStoreFactory.create_active_document_chunk_source(current_settings),
    )
    return SceneDefinition(
        scene="generic_assistant",
        name="Generic Knowledge Assistant",
        description="以用户上传文档为主，并可按会话挂载扩展到其他知识源的通用 RAG 助手。",
        build_retriever=lambda: _build_generic_agentic_retriever(
            document_retrieval_service=resolved_document_retrieval_service,
            business_extensions=resolved_business_extensions,
            max_rounds=max_rounds,
        ),
        build_tools=lambda: (
            build_generic_knowledge_document_tool(
                resolved_document_retrieval_service,
            ),
        ),
        candidate_retrieval_tools_resolver=lambda mounted_knowledge_sources: (
            _resolve_candidate_retrieval_tools(
                mounted_knowledge_sources,
                business_extensions=resolved_business_extensions,
            )
        ),
        system_prompt=GENERIC_ASSISTANT_SYSTEM_PROMPT,
        fallback_policy=SceneFallbackPolicy(
            no_hit_message="暂时没有检索到足够相关的文档知识。请补充更具体的主题、术语或文档范围，我再继续帮你查。"
        ),
        infer_complexity=infer_generic_assistant_complexity,
        bootstrap=lambda: _bootstrap_generic_scene(resolved_business_extensions),
        metadata={
            "supports_agentic_retrieval": True,
            "knowledge_sources": _resolve_supported_knowledge_sources(resolved_business_extensions),
            "business_extension_order": tuple(
                extension.knowledge_source for extension in resolved_business_extensions
            ),
            "default_agent": None,
            "prompt_style": "generic_knowledge_assistant",
        },
    )


def build_generic_knowledge_document_tool(
    document_retrieval_service: DocumentRetrievalService,
) -> BaseTool:
    """构建面向通用知识助手的文档检索工具。"""

    def knowledge_document_search(query: str, top_k: int = 5) -> ToolResult:
        retrieval_results = document_retrieval_service.retrieve(query=query, top_k=top_k)
        records = [_build_document_record(result) for result in retrieval_results]
        return ToolResult.ok(
            tool_name=GENERIC_DOCUMENT_TOOL_NAME,
            records=records,
            citations=[
                {
                    "citation_id": record["citation_id"],
                    "namespace": record["namespace"],
                    "snippet": record["snippet"],
                    "metadata": {
                        "score": record.get("score"),
                        "vector_score": record.get("vector_score"),
                        "keyword_score": record.get("keyword_score"),
                        "vector_rank": record.get("vector_rank"),
                        "keyword_rank": record.get("keyword_rank"),
                        "matched_by": record.get("matched_by", []),
                    },
                }
                for record in records
            ],
            confidence=_average_score(records),
            metadata={"namespace": "documents", "result_count": len(records), "scene": "generic_assistant"},
        )

    return build_structured_tool(
        name=GENERIC_DOCUMENT_TOOL_NAME,
        description="Search semantically relevant uploaded knowledge documents.",
        capability_type="retrieval",
        args_schema=GenericKnowledgeDocumentSearchInput,
        func=knowledge_document_search,
    )


def infer_generic_assistant_complexity(message: str) -> str:
    """按通用问答场景估算模型复杂度，避免平台层耦合业务关键词。"""
    normalized = message.strip().lower()
    complex_keywords = ("总结", "对比", "方案", "流程", "原因", "风险", "设计")
    moderate_keywords = ("解释", "说明", "如何", "为什么", "步骤", "文档", "知识库")

    if any(keyword in normalized for keyword in complex_keywords) or len(normalized) > 120:
        return "complex"
    if any(keyword in normalized for keyword in moderate_keywords) or len(normalized) > 40:
        return "moderate"
    return "simple"


def _build_generic_agentic_retriever(
    *,
    document_retrieval_service: DocumentRetrievalService,
    business_extensions: tuple[GenericAssistantBusinessExtension, ...],
    max_rounds: int,
) -> AgenticRetriever:
    """为通用场景构建文档优先的 AgenticRetriever。"""
    tools = _build_docs_first_retrieval_tools(
        document_retrieval_service=document_retrieval_service,
        business_extensions=business_extensions,
    )
    return AgenticRetriever(
        tools={tool.name: tool for tool in tools},
        default_tool=GENERIC_DOCUMENT_TOOL_NAME,
        sufficiency_judge=GenericAssistantSufficiencyJudge(
            business_extensions=business_extensions,
        ),
        query_rewriter=GenericAssistantQueryRewriter(),
        max_rounds=max_rounds,
    )


def _build_docs_first_retrieval_tools(
    *,
    document_retrieval_service: DocumentRetrievalService,
    business_extensions: tuple[GenericAssistantBusinessExtension, ...],
) -> tuple[RetrievalTool, ...]:
    """按 docs-only 默认边界组装主链工具，再按扩展顺序附加业务工具。"""
    tools: list[RetrievalTool] = [
        GenericKnowledgeDocumentSearchTool(
            document_retrieval_service=document_retrieval_service,
        )
    ]
    seen = {GENERIC_DOCUMENT_TOOL_NAME}
    for extension in business_extensions:
        for tool in extension.build_retrieval_tools():
            if tool.name in seen:
                continue
            tools.append(tool)
            seen.add(tool.name)
    return tuple(tools)


def _resolve_candidate_retrieval_tools(
    mounted_knowledge_sources: tuple[str, ...],
    *,
    business_extensions: tuple[GenericAssistantBusinessExtension, ...],
) -> tuple[str, ...]:
    """根据挂载知识源解析 generic scene 当前可用的候选检索工具。"""
    tool_names: list[str] = []
    seen: set[str] = set()

    if GENERIC_DOCUMENT_KNOWLEDGE_SOURCE in mounted_knowledge_sources:
        tool_names.append(GENERIC_DOCUMENT_TOOL_NAME)
        seen.add(GENERIC_DOCUMENT_TOOL_NAME)

    for extension in business_extensions:
        if extension.knowledge_source not in mounted_knowledge_sources:
            continue
        for tool_name in extension.retrieval_tool_names:
            if tool_name in seen:
                continue
            tool_names.append(tool_name)
            seen.add(tool_name)

    if not tool_names:
        raise ValueError("No retrieval tools available for mounted knowledge sources.")
    return tuple(tool_names)


def _resolve_supported_knowledge_sources(
    business_extensions: tuple[GenericAssistantBusinessExtension, ...],
) -> tuple[str, ...]:
    """汇总 generic scene 自身与已注册扩展支持的知识源。"""
    resolved = [GENERIC_DOCUMENT_KNOWLEDGE_SOURCE]
    seen = {GENERIC_DOCUMENT_KNOWLEDGE_SOURCE}
    for extension in business_extensions:
        if extension.knowledge_source in seen:
            continue
        resolved.append(extension.knowledge_source)
        seen.add(extension.knowledge_source)
    return tuple(resolved)


def _bootstrap_generic_scene(
    business_extensions: tuple[GenericAssistantBusinessExtension, ...],
) -> SceneBootstrapResult:
    """聚合已注册扩展的预热结果，避免 tool builder 承担 bootstrap 语义。"""
    metrics: dict[str, int] = {}
    for extension in business_extensions:
        result = extension.bootstrap()
        for metric_name, value in result.metrics.items():
            metrics[metric_name] = metrics.get(metric_name, 0) + value
    return SceneBootstrapResult(metrics=metrics)


def _build_document_record(result: DocumentChunkRetrievalResult) -> dict[str, Any]:
    """将文档知识检索结果映射为统一 record。"""
    snippet = truncate_snippet(result.document.content)
    return {
        "record_type": "document_chunk",
        "namespace": _resolve_document_namespace(result.document),
        "citation_id": _resolve_document_citation_id(result.document),
        "title": str(
            result.document.metadata.get("title")
            or result.document.metadata.get("source_path")
            or result.document.metadata.get("document_id")
            or result.document.id
        ),
        "snippet": snippet,
        "score": float(result.score) if result.score is not None else None,
        "vector_score": float(result.vector_score) if result.vector_score is not None else None,
        "keyword_score": float(result.keyword_score) if result.keyword_score is not None else None,
        "vector_rank": result.vector_rank,
        "keyword_rank": result.keyword_rank,
        "matched_by": list(result.matched_by),
        "metadata": result.document.metadata,
    }


def _resolve_document_namespace(document: Any) -> str:
    """优先保留文档知识源自己的 namespace。"""
    namespace = document.metadata.get("namespace")
    if isinstance(namespace, str) and namespace:
        return namespace
    return "documents"


def _resolve_document_citation_id(document: Any) -> str:
    """推导文档知识引用 ID。"""
    metadata = document.metadata
    return str(
        metadata.get("chunk_id")
        or metadata.get("document_id")
        or metadata.get("source_path")
        or metadata.get("id")
        or document.id
    )


def _average_score(records: list[dict[str, Any]]) -> float | None:
    """计算结果平均分，供工具和 retriever 元数据复用。"""
    scores = [float(score) for score in (record.get("score") for record in records) if isinstance(score, int | float)]
    if not scores:
        return None
    return sum(scores) / len(scores)

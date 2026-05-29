from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Any, Protocol, runtime_checkable

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict, Field

from backend.platform.config.settings import AppSettings, settings
from backend.platform.knowledge.repositories import VectorStoreFactory
from backend.platform.models.llm.client import model_client as default_model_client
from backend.platform.rag.orchestration.agentic import AgenticRetriever
from backend.platform.rag.contracts import (
    RetrievalContext,
    RetrievalTool,
)
from backend.platform.rag.orchestration.decisions import SufficiencyDecision, SufficiencyJudge
from backend.platform.rag.pre_retrieval.query_rewrite import QueryRewrite, QueryRewriter
from backend.platform.rag.pre_retrieval.query_rewrite_validator import QueryRewriteValidator
from backend.platform.rag.retrieval.documents import DocumentRetrievalService
from backend.platform.rag.retrieval.documents.filters import DOCUMENT_MINIMUM_RELEVANCE
from backend.platform.tools import build_retrieval_tool, build_scene_structured_tool
from backend.scenes.base import (
    SceneBootstrapResult,
    SceneDefinition,
    SceneFallbackPolicy,
    SceneRetrievalPolicy,
)
from backend.scenes.generic_assistant.tools import (
    GENERIC_DOCUMENT_KNOWLEDGE_SOURCE,
    GENERIC_DOCUMENT_TOOL_NAME,
    KnowledgeDocumentSearchTool,
)


GENERIC_ASSISTANT_RETRIEVAL_POLICY = SceneRetrievalPolicy(
    top_k=5,
    min_relevance_score=DOCUMENT_MINIMUM_RELEVANCE,
    recall_strategy="hybrid",
    no_hit_strategy="ask_user",
    rerank_enabled=False,
    rerank_top_n=None,
)


GENERIC_ASSISTANT_SYSTEM_PROMPT = (
    "你是一名通用知识助手。"
    "请优先依据检索到的文档上下文回答问题，回答要清晰、克制。"
    "当证据不足时，明确说明不确定，并提示用户补充更具体的文档主题、术语或背景。"
)


GENERIC_ASSISTANT_QUERY_REWRITE_PROMPT = PromptTemplate.from_template(
    """你是检索 query 改写器，只为下一轮知识库检索生成 query。

规则：
- 不要回答用户问题，不要输出解释性正文。
- 只输出一行 JSON，格式必须是：{{"query":"...","reason":"..."}}
- query 必须适合文档检索，保持中立、简洁、可搜索。
- 保留原问题中的实体名、版本号、错误码、英文缩写、数字 ID 和代码型 token。
- 如果原问题没有表达明确领域概念，不要添加无依据的通用文档词或业务概念。

用户原问题：
{original_query}

当前检索 query：
{active_query}

最近一轮检索结果摘要：
{retrieval_summary}
"""
)


@runtime_checkable
class QueryRewriteModelClient(Protocol):
    """定义 query rewrite 只需要依赖的模型客户端最小协议。"""

    def get_runnable(
        self,
        complexity: str = "simple",
        prompt_template: Any | None = None,
        *,
        output_parser: Any | None = None,
    ) -> Any:
        """返回可执行的模型 runnable。"""
        ...

    def invoke_runnable(
        self,
        runnable: Any,
        input: Any,
        *,
        config: Any | None = None,
    ) -> Any:
        """同步执行模型 runnable。"""
        ...


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
    non_retrieval_utterances: tuple[str, ...] = (
        "你好",
        "您好",
        "hello",
        "hi",
        "hey",
        "谢谢",
        "thanks",
        "thank you",
        "在吗",
        "你是谁",
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
        if not self._should_retry_retrieval(query):
            return SufficiencyDecision(
                is_sufficient=False,
                next_action="ask_user",
                reason="当前问题更像寒暄或非检索输入，不进行文档查询改写。",
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

    def _should_retry_retrieval(self, query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized:
            return False
        if normalized in self.non_retrieval_utterances:
            return False
        return True


class GenericAssistantQueryRewriter(QueryRewriter):
    """通用 docs-first 查询改写器。"""

    model_client: QueryRewriteModelClient = Field(
        default_factory=lambda: default_model_client,
        exclude=True,
    )
    query_rewrite_validator: QueryRewriteValidator = Field(
        default_factory=QueryRewriteValidator,
        exclude=True,
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def invoke(
        self,
        input: RetrievalContext,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> QueryRewrite:
        """用 LLM 生成下一轮检索 query，失败时保守回退到归一化原 query。"""
        del kwargs
        original_query = self._normalize_original_query(input.plan.active_query)
        preserved_tokens = self.query_rewrite_validator.extract_preserved_tokens(
            input.plan.user_query,
            input.plan.active_query,
        )
        try:
            runnable = self.model_client.get_runnable(
                complexity="simple",
                prompt_template=GENERIC_ASSISTANT_QUERY_REWRITE_PROMPT,
            )
            raw_output = self.model_client.invoke_runnable(
                runnable,
                self._build_prompt_variables(input, original_query=original_query),
                config=config,
            )
            parsed = self._parse_model_output(raw_output)
        except Exception as exc:
            return self._fallback_rewrite(
                original_query,
                reason="LLM query rewrite failed; using normalized original query.",
                fallback_reason=type(exc).__name__,
                preserved_tokens=preserved_tokens,
            )

        if parsed is None:
            return self._fallback_rewrite(
                original_query,
                reason="LLM query rewrite returned invalid JSON; using normalized original query.",
                fallback_reason="invalid_json_or_empty_query",
                preserved_tokens=preserved_tokens,
            )

        rewritten_query = self._normalize_rewritten_query(parsed["query"])
        if not rewritten_query:
            return self._fallback_rewrite(
                original_query,
                reason="LLM query rewrite returned an empty query; using normalized original query.",
                fallback_reason="empty_query",
                preserved_tokens=preserved_tokens,
            )

        unsafe_reason = self.query_rewrite_validator.resolve_unsafe_reason(
            original_query=original_query,
            rewritten_query=rewritten_query,
            preserved_tokens=preserved_tokens,
        )
        if unsafe_reason is not None:
            return self._fallback_rewrite(
                original_query,
                reason="LLM query rewrite was unsafe; using normalized original query.",
                fallback_reason=unsafe_reason,
                preserved_tokens=preserved_tokens,
            )

        return QueryRewrite(
            query=rewritten_query,
            reason=parsed.get("reason") or "LLM generated a focused retrieval query.",
            metadata={
                **self._build_metadata(
                    original_query=original_query,
                    fallback=False,
                    fallback_reason=None,
                    preserved_tokens=preserved_tokens,
                ),
            },
        )

    def _build_prompt_variables(
        self,
        context: RetrievalContext,
        *,
        original_query: str,
    ) -> dict[str, str]:
        """构造 prompt 变量，避免 prompt 模板直接感知 RetrievalContext 结构。"""
        return {
            "original_query": self._normalize_original_query(context.plan.user_query),
            "active_query": original_query,
            "retrieval_summary": self._summarize_latest_result(context),
        }

    def _summarize_latest_result(self, context: RetrievalContext) -> str:
        """压缩最近一轮检索结果，只给模型必要的改写背景。"""
        if not context.results:
            return "尚未执行检索。"
        result = context.results[-1]
        return (
            f"tool={result.tool_name}; "
            f"query={result.query}; "
            f"record_count={len(result.records)}; "
            f"document_count={len(result.documents)}; "
            f"success={result.success}; "
            f"error={result.error or 'none'}"
        )

    def _parse_model_output(self, raw_output: Any) -> dict[str, str] | None:
        """只接受包含非空 query 的 JSON 对象。"""
        text = str(raw_output).strip() if raw_output is not None else ""
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        # 只允许结构化 JSON 进入后续流程，避免解释性文本被当作检索 query。
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            return None
        reason = payload.get("reason")
        return {
            "query": query,
            "reason": reason.strip() if isinstance(reason, str) else "",
        }

    def _fallback_rewrite(
        self,
        original_query: str,
        *,
        reason: str,
        fallback_reason: str,
        preserved_tokens: tuple[str, ...],
    ) -> QueryRewrite:
        """构造保守 fallback，保证 query rewrite 失败不会中断聊天链路。"""
        return QueryRewrite(
            query=original_query,
            reason=reason,
            metadata=self._build_metadata(
                original_query=original_query,
                fallback=True,
                fallback_reason=fallback_reason,
                preserved_tokens=preserved_tokens,
            ),
        )

    def _build_metadata(
        self,
        *,
        original_query: str,
        fallback: bool,
        fallback_reason: str | None,
        preserved_tokens: tuple[str, ...],
    ) -> dict[str, Any]:
        """统一输出 rewrite 诊断 metadata，保持 trace 可解释。"""
        return {
            "original_query": original_query,
            "strategy": "llm_json",
            "fallback": fallback,
            "fallback_reason": fallback_reason,
            "preserved_tokens": list(preserved_tokens),
        }

    def _normalize_original_query(self, query: str) -> str:
        """归一化原始 query，作为 LLM 失败或不安全输出时的唯一 fallback。"""
        return re.sub(r"\s+", " ", query).strip()

    def _normalize_rewritten_query(self, query: str) -> str:
        """归一化模型输出 query，保持与原 query fallback 相同的空白处理。"""
        return re.sub(r"\s+", " ", query).strip()


def build_generic_assistant_scene_definition(
    app_settings: AppSettings | None = None,
    *,
    business_extensions: tuple[GenericAssistantBusinessExtension, ...] = (),
    document_retrieval_service: DocumentRetrievalService | None = None,
    retrieval_policy: SceneRetrievalPolicy = GENERIC_ASSISTANT_RETRIEVAL_POLICY,
    max_rounds: int = 3,
) -> SceneDefinition:
    """构建通用知识助手场景定义。"""
    current_settings = app_settings or settings
    resolved_retrieval_policy = retrieval_policy
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
            retrieval_policy=resolved_retrieval_policy,
            max_rounds=max_rounds,
        ),
        build_tools=lambda: (
            build_scene_structured_tool(
                _build_knowledge_document_search_tool(
                    document_retrieval_service=resolved_document_retrieval_service,
                    retrieval_policy=resolved_retrieval_policy,
                )
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
        retrieval_policy=resolved_retrieval_policy,
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
    retrieval_policy: SceneRetrievalPolicy,
    max_rounds: int,
) -> AgenticRetriever:
    """为通用场景构建文档优先的 AgenticRetriever。"""
    tools = _build_docs_first_retrieval_tools(
        document_retrieval_service=document_retrieval_service,
        business_extensions=business_extensions,
        retrieval_policy=retrieval_policy,
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
    retrieval_policy: SceneRetrievalPolicy,
) -> tuple[RetrievalTool, ...]:
    """按 docs-only 默认边界组装主链工具，再按扩展顺序附加业务工具。"""
    tools: list[RetrievalTool] = [
        build_retrieval_tool(
            _build_knowledge_document_search_tool(
                document_retrieval_service=document_retrieval_service,
                retrieval_policy=retrieval_policy,
            )
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


def _build_knowledge_document_search_tool(
    *,
    document_retrieval_service: DocumentRetrievalService,
    retrieval_policy: SceneRetrievalPolicy,
) -> KnowledgeDocumentSearchTool:
    """构建文档检索工具实例；scene 只注入策略，不承载工具业务逻辑。"""
    return KnowledgeDocumentSearchTool(
        document_retrieval_service=document_retrieval_service,
        default_top_k=retrieval_policy.top_k,
        default_min_relevance_score=retrieval_policy.min_relevance_score,
        default_recall_strategy=retrieval_policy.recall_strategy,
    )


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

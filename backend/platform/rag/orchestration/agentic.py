from __future__ import annotations

import logging
import inspect
from collections.abc import Mapping
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from backend.platform.rag.contracts import (
    RetrievalContext,
    RetrievalPlan,
    RetrievalResult,
    RetrievalTool,
)
from backend.platform.rag.orchestration.decisions import (
    RetrievalDecisionLogEntry,
    SufficiencyDecision,
    SufficiencyJudge,
)
from backend.platform.rag.post_retrieval.rerank import (
    DashScopeRetrievalReranker,
    RerankTrace,
    disabled_rerank_trace,
    remove_rerank_scores,
)
from backend.platform.agent_runtime.middleware.model_call import (
    SharedModelCallGuard,
    default_model_call_context,
)
from backend.platform.agent_runtime.middleware.trace import RuntimeTraceMiddleware
from backend.platform.rag.orchestration.retrieval_graph import build_agentic_rag_graph
from backend.platform.rag.orchestration.retrieval_graph.config import (
    AgenticRagGraphDependencies,
    build_agentic_rag_graph_context,
    build_agentic_rag_graph_config,
)
from backend.platform.rag.orchestration.retrieval_graph.projection import AgenticRagOutcomeProjector
from backend.platform.rag.orchestration.retrieval_graph.state import build_agentic_rag_graph_state
from backend.platform.rag.pre_retrieval.query_rewrite import QueryRewrite, QueryRewriter

logger = logging.getLogger(__name__)


class RetrievalRound(RetrievalContext):
    """描述单轮检索执行轨迹，便于编排器输出过程结果与调试信息。"""

    result: RetrievalResult
    decision: SufficiencyDecision
    rewrite: QueryRewrite | None = None


class AgenticRetrievalOutcome(RetrievalContext):
    """描述一次 Agentic Retrieval 会话的聚合结果与退出状态。"""

    success: bool
    rounds: list[RetrievalRound] = Field(default_factory=list)
    decision_log: list[RetrievalDecisionLogEntry] = Field(default_factory=list)
    final_plan: RetrievalPlan
    final_decision: SufficiencyDecision
    exit_reason: str
    follow_up_question: str | None = None


class AgenticRetriever(BaseRetriever):
    """基于 LangChain BaseRetriever 编排多轮检索。"""

    tools: dict[str, RetrievalTool] = Field(default_factory=dict)
    sufficiency_judge: SufficiencyJudge = Field(exclude=True)
    query_rewriter: QueryRewriter | None = Field(default=None, exclude=True)
    reranker: Any = Field(default_factory=DashScopeRetrievalReranker, exclude=True)
    default_tool: str | None = None
    max_rounds: int = Field(default=3, ge=1)
    attach_trace: bool = True
    model_call_guard: SharedModelCallGuard | None = Field(default=None, exclude=True)
    trace_middleware: RuntimeTraceMiddleware | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """适配 LangChain retriever 协议，对外返回最终聚合的 Document 列表。"""
        return self.retrieve_with_trace(query=query, run_manager=run_manager).documents

    def retrieve_with_trace(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        selected_tool: str | None = None,
        candidate_tools: tuple[str, ...] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        min_relevance_score: float | None = None,
        recall_strategy: str = "hybrid",
        rerank_enabled: bool = False,
        rerank_top_n: int | None = None,
    ) -> AgenticRetrievalOutcome:
        """执行多轮检索并返回完整轨迹，供 Agent 或 LangGraph 复用。"""
        self._validate_tools()
        resolved_candidate_tools = self._resolve_candidate_tools(candidate_tools)
        initial_tool = self._resolve_initial_tool(selected_tool, resolved_candidate_tools)
        logger.info(
            "Agentic retrieval started: query=%r, initial_tool=%s, candidate_tools=%s, max_rounds=%s",
            query,
            initial_tool,
            resolved_candidate_tools,
            self.max_rounds,
        )
        graph_state = build_agentic_rag_graph_state(
            query=query,
            selected_tool=initial_tool,
            candidate_tools=resolved_candidate_tools,
            filters=filters or {},
            top_k=top_k,
            min_relevance_score=min_relevance_score,
            recall_strategy=recall_strategy,
            rerank_enabled=rerank_enabled,
            rerank_top_n=rerank_top_n,
            max_rounds=self.max_rounds,
        )
        graph = build_agentic_rag_graph(
            AgenticRagGraphDependencies(
                retriever=self,
                run_manager=run_manager,
                model_call_guard=self._model_call_guard(),
                trace=self._trace_middleware(),
            ),
        )
        graph_config = build_agentic_rag_graph_config(
            retriever=self,
            run_manager=run_manager,
        )
        graph_context = build_agentic_rag_graph_context(
            retriever=self,
            run_manager=run_manager,
            runtime_context=default_model_call_context(
                session_id="agentic-rag",
                request_id=f"agentic-rag:{query[:64]}",
                scene="platform.rag",
                complexity="simple",
                workflow_metadata={
                    "run_id": f"agentic-rag:{query[:64]}",
                    "checkpoint_ns": "agentic_rag",
                    "metadata": {"graph": "agentic_rag_graph"},
                },
                request_metadata={
                    "operation": "agentic_retrieval",
                    "candidate_tools": resolved_candidate_tools,
                    "selected_tool": initial_tool,
                },
            ),
            model_call_guard=self._model_call_guard(),
            trace=self._trace_middleware(),
        )
        final_state = graph.invoke(
            graph_state,
            graph_config,
            context=graph_context,
        )
        logger.info(
            "Agentic retrieval finished: exit_reason=%s, rounds=%s, total_documents=%s",
            final_state.get("exit_reason"),
            len(final_state.get("rounds") or []),
            len(final_state.get("documents") or []),
        )
        return self._outcome_projector().project(final_state)

    def _outcome_projector(self) -> AgenticRagOutcomeProjector:
        return AgenticRagOutcomeProjector(
            outcome_factory=AgenticRetrievalOutcome,
            round_factory=RetrievalRound,
        )

    def _model_call_guard(self) -> SharedModelCallGuard:
        if self.model_call_guard is not None:
            return self.model_call_guard
        return SharedModelCallGuard(trace=self._trace_middleware())

    def _trace_middleware(self) -> RuntimeTraceMiddleware:
        if self.trace_middleware is not None:
            return self.trace_middleware
        trace = RuntimeTraceMiddleware()
        self.trace_middleware = trace
        return trace

    def _run_tool(
        self,
        plan: RetrievalPlan,
        run_manager: CallbackManagerForRetrieverRun | None,
    ) -> RetrievalResult:
        """执行当前轮次指定工具，并传递 LangChain 回调上下文。"""
        tool = self._get_tool(plan.selected_tool)
        child_manager = run_manager.get_child(tag=f"retrieval:{tool.name}") if run_manager else None
        logger.debug(
            "Executing retrieval tool: tool=%s, query=%r, round=%s",
            tool.name,
            plan.active_query,
            plan.round_index,
        )
        retrieve_kwargs: dict[str, Any] = {
            "query": plan.active_query,
            "run_manager": child_manager,
        }
        if self._supports_tool_argument(tool, "top_k"):
            retrieve_kwargs["top_k"] = plan.top_k
        if self._supports_tool_argument(tool, "min_relevance_score"):
            retrieve_kwargs["min_relevance_score"] = plan.min_relevance_score
        if self._supports_tool_argument(tool, "recall_strategy"):
            retrieve_kwargs["recall_strategy"] = plan.recall_strategy
        if self._supports_tool_argument(tool, "rerank_enabled"):
            retrieve_kwargs["rerank_enabled"] = plan.rerank_enabled
        if self._supports_tool_argument(tool, "rerank_top_n"):
            retrieve_kwargs["rerank_top_n"] = plan.rerank_top_n
        return tool.retrieve(**retrieve_kwargs)

    def _apply_rerank(self, *, plan: RetrievalPlan, result: RetrievalResult) -> RetrievalResult:
        """在 tool result 后、充分性判断前应用可替换 rerank 边界。"""
        if not plan.rerank_enabled:
            # 默认关闭时只补 trace，不改 records/documents/citations 的既有语义。
            trace = disabled_rerank_trace(result)
            sanitized = remove_rerank_scores(result)
            return sanitized.model_copy(
                update={
                    "metadata": {
                        **sanitized.metadata,
                        "rerank": trace.to_dict(),
                    }
                }
            )
        try:
            reranked, _trace = self.reranker.rerank(
                query=plan.active_query,
                result=result,
                top_n=plan.rerank_top_n,
            )
            return reranked
        except Exception as exc:
            # 外层兜底只负责保护编排稳定性，具体模型调用和映射逻辑仍由 adapter 承担。
            provider = getattr(self.reranker, "provider", "unknown")
            input_count = max(len(result.records), len(result.documents), len(result.citations))
            logger.warning(
                "Rerank adapter failed; preserving original retrieval order: provider=%s, reason=%s",
                provider,
                type(exc).__name__,
                exc_info=True,
            )
            trace = RerankTrace(
                enabled=True,
                provider=provider,
                model=None,
                applied=False,
                input_count=input_count,
                output_count=input_count,
                top_n=plan.rerank_top_n,
                fallback_reason=type(exc).__name__,
                error=str(exc).strip().splitlines()[0][:240] if str(exc).strip() else type(exc).__name__,
            )
            sanitized = remove_rerank_scores(result)
            return sanitized.model_copy(
                update={
                    "metadata": {
                        **sanitized.metadata,
                        "rerank": trace.to_dict(),
                    }
                }
            )

    def _rewrite_query(
        self,
        context: RetrievalContext,
        run_manager: CallbackManagerForRetrieverRun | None,
    ) -> QueryRewrite:
        """调用 LangChain runnable 风格的查询改写器。"""
        if self.query_rewriter is None:
            raise ValueError("Query rewriter is required when judge requests rewrite.")
        rewrite = self.query_rewriter.invoke(
            context,
            config=self._build_runnable_config(run_manager, tag="query_rewriter"),
        )
        if not rewrite.query.strip():
            raise ValueError("Query rewriter returned an empty query.")
        logger.debug(
            "Query rewrite completed: from=%r, to=%r, metadata=%s",
            context.plan.active_query,
            rewrite.query,
            rewrite.metadata,
        )
        return rewrite

    def _rewrite_already_attempted(self, plan: RetrievalPlan) -> bool:
        """判断本次检索会话是否已经执行过 query rewrite。"""
        return bool(plan.metadata.get("rewrite_attempted"))

    def _build_rewrite_limit_decision(self, decision: SufficiencyDecision) -> SufficiencyDecision:
        """将重复 rewrite 请求收口为 ask_user，避免 LLM 改写循环。"""
        return decision.model_copy(
            update={
                "is_sufficient": False,
                "next_action": "ask_user",
                "reason": (
                    f"{decision.reason} Query rewrite already attempted once; "
                    "stopping retrieval rewrite loop."
                ),
            }
        )

    def _judge(
        self,
        context: RetrievalContext,
        run_manager: CallbackManagerForRetrieverRun | None,
    ) -> SufficiencyDecision:
        """调用 LangChain runnable 风格的充分性判断器。"""
        decision = self.sufficiency_judge.invoke(
            context,
            config=self._build_runnable_config(run_manager, tag="sufficiency_judge"),
        )
        logger.debug(
            "Sufficiency judge completed: tool=%s, action=%s, sufficient=%s",
            context.plan.selected_tool,
            decision.next_action,
            decision.is_sufficient,
        )
        return decision

    def _resolve_next_tool(self, plan: RetrievalPlan, decision: SufficiencyDecision) -> str:
        """解析下一轮工具，始终限制在本轮候选工具集合内。"""
        attempted = set(plan.attempted_tools) | {plan.selected_tool}

        if (
            decision.suggested_tool
            and decision.suggested_tool in plan.candidate_tools
            and decision.suggested_tool not in attempted
        ):
            self._get_tool(decision.suggested_tool)
            return decision.suggested_tool

        for tool_name in plan.candidate_tools:
            if tool_name not in attempted:
                self._get_tool(tool_name)
                return tool_name

        raise ValueError("No alternative retrieval tool available for switch_tool decision.")

    def _resolve_candidate_tools(
        self,
        candidate_tools: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """规范化候选工具列表，确保只包含已注册工具。"""
        if candidate_tools is None:
            return tuple(self.tools.keys())

        resolved: list[str] = []
        seen: set[str] = set()
        for tool_name in candidate_tools:
            if tool_name in seen:
                continue
            self._get_tool(tool_name)
            seen.add(tool_name)
            resolved.append(tool_name)

        if not resolved:
            raise ValueError("At least one candidate retrieval tool must be provided.")
        return tuple(resolved)

    def _resolve_initial_tool(
        self,
        selected_tool: str | None,
        candidate_tools: tuple[str, ...],
    ) -> str:
        """确定首轮工具，优先显式指定，其次默认工具，最后取第一个候选工具。"""
        if selected_tool is not None:
            if selected_tool not in candidate_tools:
                raise ValueError(
                    f"Selected retrieval tool '{selected_tool}' is not in candidate tools."
                )
            self._get_tool(selected_tool)
            logger.debug("Resolved initial retrieval tool from explicit selection: %s", selected_tool)
            return selected_tool

        if self.default_tool and self.default_tool in candidate_tools:
            self._get_tool(self.default_tool)
            logger.debug("Resolved initial retrieval tool from default tool: %s", self.default_tool)
            return self.default_tool

        first_tool = candidate_tools[0]
        self._get_tool(first_tool)
        logger.debug("Resolved initial retrieval tool from first candidate: %s", first_tool)
        return first_tool

    def _get_tool(self, tool_name: str) -> RetrievalTool:
        """按名称解析检索工具，不存在时抛出显式错误。"""
        tool = self.tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Retrieval tool '{tool_name}' is not registered.")
        return tool

    def _supports_tool_argument(self, tool: RetrievalTool, argument_name: str) -> bool:
        """兼容旧 RetrievalTool 实现，避免一次性改动所有 scene。"""
        parameters = inspect.signature(tool.retrieve).parameters
        return argument_name in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    def _validate_tools(self) -> None:
        """校验工具集合、默认工具和最大轮次配置是否有效。"""
        if not self.tools:
            raise ValueError("At least one retrieval tool must be provided.")
        if self.default_tool and self.default_tool not in self.tools:
            raise ValueError(f"Default retrieval tool '{self.default_tool}' is not registered.")

    def _merge_documents(
        self,
        existing: list[Document],
        incoming: list[Document],
    ) -> list[Document]:
        """按文档标识聚合去重，保留首次出现顺序。"""
        merged = list(existing)
        seen = {self._document_key(document) for document in existing}
        for document in incoming:
            key = self._document_key(document)
            if key in seen:
                continue
            seen.add(key)
            merged.append(document)
        return merged

    def _finalize_documents(
        self,
        documents: list[Document],
        rounds: list[RetrievalRound],
        exit_reason: str,
    ) -> list[Document]:
        """按需将检索轨迹写入文档 metadata，便于 LangChain 下游链路调试。"""
        if not self.attach_trace:
            return documents

        trace = [
            {
                "round_index": round_trace.plan.round_index,
                "tool_name": round_trace.result.tool_name,
                "query": round_trace.result.query,
                "rewritten_query": round_trace.rewrite.query if round_trace.rewrite else None,
                "decision": round_trace.decision.next_action,
                "reason": round_trace.decision.reason,
                "result_success": round_trace.result.success,
                "result_count": len(round_trace.result.records),
            }
            for round_trace in rounds
        ]
        return [
            document.model_copy(
                update={
                    "metadata": {
                        **document.metadata,
                        "agentic_retrieval": {
                            "trace": trace,
                            "exit_reason": exit_reason,
                        },
                    }
                }
            )
            for document in documents
        ]

    def _build_runnable_config(
        self,
        run_manager: CallbackManagerForRetrieverRun | None,
        *,
        tag: str,
    ) -> dict[str, Any] | None:
        """为内部 runnable 构造 LangChain config，保证 tracing 不断链。"""
        if run_manager is None:
            return None
        return {"callbacks": [run_manager.get_child(tag=tag)]}

    def _document_key(self, document: Document) -> tuple[str, str]:
        """为去重计算稳定键，优先使用 metadata 中的 citation_id/source。"""
        citation_id = str(document.metadata.get("citation_id") or document.id or document.page_content)
        source = str(document.metadata.get("namespace") or document.metadata.get("source") or "knowledge")
        return source, citation_id

    def _build_decision_log_entry(
        self,
        round_trace: RetrievalRound,
        *,
        rewritten_query: str | None,
        exit_reason: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RetrievalDecisionLogEntry:
        """将单轮执行轨迹转换为稳定的决策日志记录。"""
        metadata = dict(round_trace.decision.metadata)
        metadata["exit_reason"] = exit_reason
        if extra_metadata:
            metadata.update(extra_metadata)
        return RetrievalDecisionLogEntry(
            round_index=round_trace.plan.round_index,
            tool_name=round_trace.result.tool_name,
            query=round_trace.result.query,
            rewritten_query=rewritten_query,
            result_count=len(round_trace.result.records),
            result_success=round_trace.result.success,
            result_confidence=round_trace.result.confidence,
            decision=round_trace.decision.next_action,
            is_sufficient=round_trace.decision.is_sufficient,
            reason=round_trace.decision.reason,
            suggested_tool=round_trace.decision.suggested_tool,
            follow_up_question=round_trace.decision.follow_up_question,
            metadata=metadata,
        )

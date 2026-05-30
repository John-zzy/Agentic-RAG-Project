from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from inspect import Parameter, signature
import logging
from pathlib import Path
import re
from typing import Any, Literal, Protocol
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory

from backend.application.runtime.api.chat.prompts import build_rag_answer_prompt_template
from backend.application.runtime.api.chat.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    RetrievalTrace,
    RetrievalTraceRound,
    RetrievalTraceTopChunk,
)
from backend.platform.config.settings import AppSettings, settings
from backend.platform.knowledge.sources import (
    DEFAULT_MOUNTED_KNOWLEDGE_SOURCES,
    normalize_mounted_knowledge_sources,
)
from backend.platform.rag.retrieval.documents import DocumentRetrievalService
from backend.scenes.base import SceneDefinition, SceneRetrievalPolicy
from backend.platform.knowledge.base.text import truncate_snippet
from backend.platform.memory.base.chat_history import SQLiteChatMessageHistory
from backend.platform.memory.base.session_store import (
    SQLiteSessionStore,
    SessionRecord,
)
from backend.platform.memory.chat.prompt_context import PromptContextBuilder
from backend.platform.models.base.router import TaskComplexity
from backend.platform.models.llm.client import ModelClient, model_client
from backend.scenes.generic_assistant.definition import GenericAssistantBusinessExtension
from backend.scenes.registry import build_default_scene_definitions

logger = logging.getLogger(__name__)

RuntimeFinalDecision = Literal[
    "answer_with_evidence",
    "ask_user",
    "max_rounds_reached",
    "no_evidence",
    "retrieval_failed",
]
AnswerMode = Literal["evidence_answer", "follow_up", "fallback"]


class RetrievalChainModel(Protocol):
    """定义运行时依赖的最小模型构建协议。"""

    def get_runnable(
        self,
        complexity: TaskComplexity = "simple",
        prompt_template: Any | None = None,
        *,
        output_parser: Any | None = None,
    ) -> Any:
        """返回可供 runtime 执行的 LCEL runnable。"""
        ...

    def invoke_runnable(
        self,
        runnable: Any,
        input: Any,
        *,
        config: Any | None = None,
    ) -> Any:
        """同步执行 runnable。"""
        ...

    def stream_runnable(
        self,
        runnable: Any,
        input: Any,
        *,
        config: Any | None = None,
    ) -> Iterator[Any]:
        """流式执行 runnable。"""
        ...


class ChatServiceError(RuntimeError):
    """封装可返回给 API 层的业务错误。"""

    def __init__(self, *, status_code: int, code: str, message: str, request_id: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        super().__init__(message)


@dataclass(frozen=True)
class SceneMetadata:
    """描述统一聊天响应需要返回的场景元数据。"""

    scene: str
    agent: str | None = None


@dataclass(frozen=True)
class RetrievalExecutionResult:
    """封装一次检索执行后的文档与来源工具信息。"""

    documents: list[Document]
    tool_event: dict[str, Any]
    retrieval_trace: RetrievalTrace
    # 这些字段承载 RetrievalExecutor 已归一化的 outcome 层语义。
    success: bool | None = None
    final_decision: RuntimeFinalDecision | None = None
    follow_up_question: str | None = None


@dataclass(frozen=True)
class PreparedChatTurn:
    """封装同步和流式路径共享的聊天准备结果。"""

    session_id: str
    request_id: str
    timestamp: str
    user_message: str
    documents: list[Document]
    tool_event: dict[str, Any]
    retrieval_trace: RetrievalTrace
    citations: list[Citation]
    knowledge_used: bool
    scene_metadata: SceneMetadata
    complexity: TaskComplexity | None
    # 同步与 SSE 后续共用的回答分支元数据
    final_decision: RuntimeFinalDecision | None = None
    follow_up_question: str | None = None
    answer_mode: AnswerMode = "fallback"


@dataclass(frozen=True)
class RuntimeRetrievalDecision:
    """封装 application runtime 对检索 outcome 的归一化判断。"""

    success: bool
    exit_reason: str | None
    final_decision: RuntimeFinalDecision
    follow_up_question: str | None = None


@dataclass(frozen=True)
class ChatStreamEvent:
    """描述一条待编码为 SSE 的聊天流事件。"""

    event: str
    data: dict[str, Any]


class RetrievalExecutor:
    """负责执行 retriever 并返回统一的文档结果。"""

    def __init__(self, *, scene_definition: SceneDefinition, retriever: Any) -> None:
        self._scene_definition = scene_definition
        self._retriever = retriever

    def retrieve(
        self,
        message: str,
        *,
        mounted_knowledge_sources: tuple[str, ...],
    ) -> RetrievalExecutionResult:
        policy = self._scene_definition.retrieval_policy
        policy_summary = self._build_policy_summary(policy)
        candidate_tools = self._scene_definition.resolve_candidate_retrieval_tools(
            mounted_knowledge_sources
        )
        logger.info(
            "Starting retrieval for scene=%s: message=%r, mounted_knowledge_sources=%s, candidate_tools=%s",
            self._scene_definition.scene,
            message,
            mounted_knowledge_sources,
            candidate_tools,
        )
        if hasattr(self._retriever, "retrieve_with_trace"):
            retrieve_with_trace = self._retriever.retrieve_with_trace
            retrieval_kwargs = self._build_supported_policy_kwargs(retrieve_with_trace, policy)
            outcome: Any = self._retriever.retrieve_with_trace(  # type: ignore[attr-defined]
                message,
                candidate_tools=candidate_tools,
                **retrieval_kwargs,
            )
            outcome_documents = list(getattr(outcome, "documents", []))
            outcome_rounds = list(getattr(outcome, "rounds", []))
            runtime_decision = self._normalize_agentic_outcome(
                outcome,
                documents=outcome_documents,
            )
            logger.info(
                "Agentic retrieval completed for scene=%s: exit_reason=%s, final_decision=%s, rounds=%s, documents=%s",
                self._scene_definition.scene,
                runtime_decision.exit_reason,
                runtime_decision.final_decision,
                len(outcome_rounds),
                len(outcome_documents),
            )
            rounds = [self._build_agentic_round_trace(round_trace) for round_trace in outcome_rounds]
            retrieval_trace = self._build_retrieval_trace(
                original_query=message,
                final_query=self._resolve_outcome_final_query(outcome, message),
                rewritten_query=self._last_rewritten_query(rounds),
                candidate_tools=candidate_tools,
                exit_reason=runtime_decision.exit_reason,
                rounds=rounds,
                documents=outcome_documents,
                success=runtime_decision.success,
                final_decision=runtime_decision.final_decision,
                follow_up_question=runtime_decision.follow_up_question,
            )
            return RetrievalExecutionResult(
                documents=outcome_documents,
                tool_event={
                    "stage": "retrieval",
                    "mode": "agentic",
                    "retrieval_policy": policy_summary,
                    "candidate_tools": list(candidate_tools),
                    "documents": len(outcome_documents),
                    "exit_reason": runtime_decision.exit_reason,
                    "success": runtime_decision.success,
                    "final_decision": runtime_decision.final_decision,
                    "follow_up_question": runtime_decision.follow_up_question,
                    "rounds": [round_trace.model_dump() for round_trace in rounds],
                },
                retrieval_trace=retrieval_trace,
                success=runtime_decision.success,
                final_decision=runtime_decision.final_decision,
                follow_up_question=runtime_decision.follow_up_question,
            )
        if hasattr(self._retriever, "search"):
            search = self._retriever.search
            retrieval_kwargs = self._build_supported_policy_kwargs(search, policy)
            documents = list(search(query=message, **retrieval_kwargs))  # type: ignore[attr-defined]
            logger.info(
                "Retriever search completed for scene=%s: documents=%s",
                self._scene_definition.scene,
                len(documents),
            )
            round_trace = self._build_simple_round_trace(
                round_index=1,
                tool_name="search",
                query=message,
                reason="search completed",
                documents=documents,
            )
            runtime_decision = self._normalize_simple_retrieval_result(
                documents=documents,
                exit_reason="finished_by_search",
            )
            return RetrievalExecutionResult(
                documents=documents,
                tool_event={
                    "stage": "retrieval",
                    "mode": "search",
                    "retrieval_policy": policy_summary,
                    "candidate_tools": list(candidate_tools),
                    "documents": len(documents),
                    "exit_reason": runtime_decision.exit_reason,
                    "success": runtime_decision.success,
                    "final_decision": runtime_decision.final_decision,
                    "follow_up_question": runtime_decision.follow_up_question,
                    "rounds": [round_trace.model_dump()],
                },
                retrieval_trace=self._build_retrieval_trace(
                    original_query=message,
                    final_query=message,
                    rewritten_query=None,
                    candidate_tools=candidate_tools,
                    exit_reason=runtime_decision.exit_reason,
                    rounds=[round_trace],
                    documents=documents,
                    success=runtime_decision.success,
                    final_decision=runtime_decision.final_decision,
                    follow_up_question=runtime_decision.follow_up_question,
                ),
                success=runtime_decision.success,
                final_decision=runtime_decision.final_decision,
                follow_up_question=runtime_decision.follow_up_question,
            )
        if isinstance(self._retriever, BaseRetriever):
            documents = list(self._retriever.invoke(message))
            logger.info(
                "BaseRetriever invoke completed for scene=%s: documents=%s",
                self._scene_definition.scene,
                len(documents),
            )
            round_trace = self._build_simple_round_trace(
                round_index=1,
                tool_name=type(self._retriever).__name__,
                query=message,
                reason="retriever invoke completed",
                documents=documents,
            )
            runtime_decision = self._normalize_simple_retrieval_result(
                documents=documents,
                exit_reason="finished_by_retriever",
            )
            return RetrievalExecutionResult(
                documents=documents,
                tool_event={
                    "stage": "retrieval",
                    "mode": "base_retriever",
                    "retrieval_policy": policy_summary,
                    "candidate_tools": list(candidate_tools),
                    "documents": len(documents),
                    "exit_reason": runtime_decision.exit_reason,
                    "success": runtime_decision.success,
                    "final_decision": runtime_decision.final_decision,
                    "follow_up_question": runtime_decision.follow_up_question,
                    "rounds": [round_trace.model_dump()],
                },
                retrieval_trace=self._build_retrieval_trace(
                    original_query=message,
                    final_query=message,
                    rewritten_query=None,
                    candidate_tools=candidate_tools,
                    exit_reason=runtime_decision.exit_reason,
                    rounds=[round_trace],
                    documents=documents,
                    success=runtime_decision.success,
                    final_decision=runtime_decision.final_decision,
                    follow_up_question=runtime_decision.follow_up_question,
                ),
                success=runtime_decision.success,
                final_decision=runtime_decision.final_decision,
                follow_up_question=runtime_decision.follow_up_question,
            )
        raise TypeError("Retriever does not support document retrieval.")

    def _normalize_agentic_outcome(
        self,
        outcome: Any,
        *,
        documents: list[Document],
    ) -> RuntimeRetrievalDecision:
        """从 Agentic outcome 中兼容读取字段，并归一化为 runtime 决策。"""
        raw_final_decision = getattr(outcome, "final_decision", None)
        exit_reason = self._resolve_optional_str(getattr(outcome, "exit_reason", None))
        success = self._resolve_outcome_success(
            getattr(outcome, "success", None),
            has_documents=len(documents) > 0,
        )
        follow_up_question = self._resolve_follow_up_question(
            outcome=outcome,
            raw_final_decision=raw_final_decision,
        )
        final_decision = self._normalize_runtime_final_decision(
            success=success,
            exit_reason=exit_reason,
            raw_final_decision=raw_final_decision,
            has_documents=len(documents) > 0,
        )
        return RuntimeRetrievalDecision(
            success=success,
            exit_reason=exit_reason,
            final_decision=final_decision,
            follow_up_question=follow_up_question,
        )

    def _resolve_outcome_success(self, value: Any, *, has_documents: bool) -> bool:
        """兼容缺少 success 的旧式 outcome，有文档时按成功候选处理。"""
        if isinstance(value, bool):
            return value
        if value is None:
            return has_documents
        return bool(value)

    def _normalize_simple_retrieval_result(
        self,
        *,
        documents: list[Document],
        exit_reason: str,
    ) -> RuntimeRetrievalDecision:
        """为旧 search/BaseRetriever 分支补齐与 Agentic 分支一致的 runtime 决策。"""
        success = True
        final_decision = self._normalize_runtime_final_decision(
            success=success,
            exit_reason=exit_reason,
            raw_final_decision=None,
            has_documents=len(documents) > 0,
        )
        return RuntimeRetrievalDecision(
            success=success,
            exit_reason=exit_reason,
            final_decision=final_decision,
            follow_up_question=None,
        )

    def _normalize_runtime_final_decision(
        self,
        *,
        success: bool,
        exit_reason: str | None,
        raw_final_decision: Any,
        has_documents: bool,
    ) -> RuntimeFinalDecision:
        """把平台层检索动作转换成 `/chat` 可消费的最终业务语义。"""
        next_action = getattr(raw_final_decision, "next_action", None)
        is_sufficient = bool(getattr(raw_final_decision, "is_sufficient", False))

        if exit_reason == "max_rounds_reached":
            return "max_rounds_reached"
        if next_action == "ask_user" or exit_reason == "ask_user":
            return "ask_user"
        if not success:
            return "retrieval_failed"
        # 旧式 outcome/search/BaseRetriever 没有 final_decision，此时先按是否有文档映射。
        if raw_final_decision is None:
            return "answer_with_evidence" if has_documents else "no_evidence"
        # 归一化阶段只能看到文档；最终是否采纳仍由 effective citations 门控。
        if is_sufficient and next_action == "finish" and has_documents:
            return "answer_with_evidence"
        return "no_evidence"

    def _resolve_follow_up_question(
        self,
        *,
        outcome: Any,
        raw_final_decision: Any,
    ) -> str | None:
        """按 outcome 优先、final_decision 兜底的顺序解析追问文本。"""
        for value in (
            getattr(outcome, "follow_up_question", None),
            getattr(raw_final_decision, "follow_up_question", None),
        ):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _build_agentic_round_trace(self, round_trace: Any) -> RetrievalTraceRound:
        document_trace = self._extract_document_retrieval_trace(round_trace.result.metadata)
        raw_candidates_count = self._resolve_count(
            document_trace.get("raw_candidates_count"),
            fallback=len(round_trace.result.records),
        )
        filtered_candidates_count = self._resolve_count(
            document_trace.get("filtered_candidates_count"),
            fallback=len(round_trace.result.documents),
        )
        rerank_trace = round_trace.result.metadata.get("rerank")
        top_k_chunks = self._round_top_chunks(
            document_trace=document_trace,
            rerank_trace=rerank_trace,
            documents=round_trace.result.documents,
        )
        return RetrievalTraceRound(
            round_index=round_trace.plan.round_index,
            tool_name=round_trace.result.tool_name,
            query=round_trace.result.query,
            rewritten_query=round_trace.rewrite.query if round_trace.rewrite is not None else None,
            decision=round_trace.decision.next_action,
            is_sufficient=round_trace.decision.is_sufficient,
            reason=round_trace.decision.reason,
            result_count=len(round_trace.result.records),
            document_count=len(round_trace.result.documents),
            success=round_trace.result.success,
            error=round_trace.result.error,
            raw_candidates_count=raw_candidates_count,
            filtered_candidates_count=filtered_candidates_count,
            top_k_chunks=top_k_chunks,
            rerank=rerank_trace,
        )

    def _build_simple_round_trace(
        self,
        *,
        round_index: int,
        tool_name: str,
        query: str,
        reason: str,
        documents: list[Document],
    ) -> RetrievalTraceRound:
        return RetrievalTraceRound(
            round_index=round_index,
            tool_name=tool_name,
            query=query,
            rewritten_query=None,
            decision="finish",
            is_sufficient=len(documents) > 0,
            reason=reason,
            result_count=len(documents),
            document_count=len(documents),
            success=True,
            error=None,
            raw_candidates_count=len(documents),
            filtered_candidates_count=len(documents),
            top_k_chunks=self._top_chunks_from_documents(documents),
            rerank=None,
        )

    def _build_retrieval_trace(
        self,
        *,
        original_query: str,
        final_query: str,
        rewritten_query: str | None,
        candidate_tools: tuple[str, ...],
        exit_reason: str | None,
        rounds: list[RetrievalTraceRound],
        documents: list[Document],
        success: bool | None = None,
        final_decision: RuntimeFinalDecision | None = None,
        follow_up_question: str | None = None,
    ) -> RetrievalTrace:
        raw_candidates_count = sum(round_trace.raw_candidates_count or 0 for round_trace in rounds)
        filtered_candidates_count = sum(
            round_trace.filtered_candidates_count or 0 for round_trace in rounds
        )
        return RetrievalTrace(
            original_query=original_query,
            final_query=final_query,
            rewritten_query=rewritten_query,
            tool_call_count=len(rounds),
            candidate_tools=list(candidate_tools),
            exit_reason=exit_reason,
            final_decision=final_decision,
            success=success,
            follow_up_question=follow_up_question,
            raw_candidates_count=raw_candidates_count,
            filtered_candidates_count=filtered_candidates_count,
            top_k_chunks=self._top_chunks_from_documents(documents),
            citations=[],
            knowledge_used=False,
            rounds=rounds,
        )

    def _extract_document_retrieval_trace(self, metadata: dict[str, Any]) -> dict[str, Any]:
        trace = metadata.get("document_retrieval_trace")
        if isinstance(trace, dict):
            return trace
        return {}

    def _coerce_top_chunks(self, value: Any) -> list[RetrievalTraceTopChunk]:
        if not isinstance(value, list):
            return []
        chunks: list[RetrievalTraceTopChunk] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                chunks.append(RetrievalTraceTopChunk.model_validate(item))
            except ValueError:
                continue
        return chunks

    def _round_top_chunks(
        self,
        *,
        document_trace: dict[str, Any],
        rerank_trace: Any,
        documents: list[Document],
    ) -> list[RetrievalTraceTopChunk]:
        if isinstance(rerank_trace, dict) and rerank_trace.get("enabled") is True:
            # 步骤 1：ReRank 边界开启后，轮次 trace 使用重排/截断后的最终证据顺序。
            return self._top_chunks_from_documents(documents)
        return self._coerce_top_chunks(document_trace.get("top_k_chunks"))

    def _top_chunks_from_documents(self, documents: list[Document]) -> list[RetrievalTraceTopChunk]:
        chunks: list[RetrievalTraceTopChunk] = []
        for rank, document in enumerate(documents, start=1):
            metadata = document.metadata
            citation_id = str(metadata.get("citation_id") or metadata.get("chunk_id") or "unknown")
            chunks.append(
                RetrievalTraceTopChunk(
                    rank=rank,
                    citation_id=citation_id,
                    document_id=self._resolve_optional_str(metadata.get("document_id")),
                    chunk_id=self._resolve_optional_str(metadata.get("chunk_id")),
                    chunk_index=self._resolve_int(metadata.get("chunk_index")),
                    source_name=self._resolve_source_name(
                        citation_id=citation_id,
                        document_id=self._resolve_optional_str(metadata.get("document_id")),
                        metadata=metadata,
                    ),
                    source_path=self._resolve_optional_str(metadata.get("source_path")),
                    score=self._resolve_float(metadata.get("score")),
                    vector_score=self._resolve_float(metadata.get("vector_score")),
                    keyword_score=self._resolve_float(metadata.get("keyword_score")),
                    vector_rank=self._resolve_int(metadata.get("vector_rank")),
                    keyword_rank=self._resolve_int(metadata.get("keyword_rank")),
                    rerank_score=self._resolve_float(metadata.get("rerank_score")),
                    matched_by=self._resolve_matched_by(metadata.get("matched_by")),
                )
            )
        return chunks

    def _last_rewritten_query(self, rounds: list[RetrievalTraceRound]) -> str | None:
        for round_trace in reversed(rounds):
            if round_trace.rewritten_query:
                return round_trace.rewritten_query
        return None

    def _resolve_outcome_final_query(self, outcome: Any, fallback: str) -> str:
        final_plan = getattr(outcome, "final_plan", None)
        active_query = getattr(final_plan, "active_query", None)
        if isinstance(active_query, str) and active_query:
            return active_query
        plan = getattr(outcome, "plan", None)
        active_query = getattr(plan, "active_query", None)
        if isinstance(active_query, str) and active_query:
            return active_query
        return fallback

    def _resolve_source_name(
        self,
        *,
        citation_id: str,
        document_id: str | None,
        metadata: dict[str, Any],
    ) -> str:
        source_path = self._resolve_optional_str(metadata.get("source_path"))
        if source_path:
            return Path(source_path).name
        if document_id:
            return document_id
        for field_name in ("title", "name", "product_name", "order_id", "product_id", "review_id"):
            resolved = self._resolve_optional_str(metadata.get(field_name))
            if resolved:
                return resolved
        return citation_id

    def _resolve_optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, int | float):
            return str(value)
        return None

    def _resolve_float(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return float(value)
        return None

    def _resolve_int(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    def _resolve_count(self, value: Any, *, fallback: int) -> int:
        resolved = self._resolve_int(value)
        if resolved is None or resolved < 0:
            return fallback
        return resolved

    def _resolve_matched_by(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if isinstance(item, str)]
        return []

    def _build_policy_summary(self, policy: SceneRetrievalPolicy) -> dict[str, Any]:
        """返回可暴露到事件流的 scene 检索策略摘要。"""
        return {
            "top_k": policy.top_k,
            "min_relevance_score": policy.min_relevance_score,
            "recall_strategy": policy.recall_strategy,
            "no_hit_strategy": policy.no_hit_strategy,
            "rerank_enabled": policy.rerank_enabled,
            "rerank_top_n": policy.rerank_top_n,
        }

    def _build_supported_policy_kwargs(
        self,
        callable_obj: Any,
        policy: SceneRetrievalPolicy,
    ) -> dict[str, Any]:
        try:
            parameters = signature(callable_obj).parameters
        except (TypeError, ValueError):
            return {}

        # 运行时只负责把 scene policy 适配给当前 retriever 支持的参数名。
        accepts_kwargs = any(
            parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        policy_kwargs = {
            "top_k": policy.top_k,
            "min_relevance_score": policy.min_relevance_score,
            "minimum_relevance": policy.min_relevance_score,
            "recall_strategy": policy.recall_strategy,
            "rerank_enabled": policy.rerank_enabled,
            "rerank_top_n": policy.rerank_top_n,
        }
        return {
            name: value
            for name, value in policy_kwargs.items()
            if value is not None and (accepts_kwargs or name in parameters)
        }


class CitationMapper:
    """负责将检索文档映射为 API citations 与回答上下文。"""

    def citations_from_documents(self, documents: list[Document]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()

        for rank, doc in enumerate(documents, start=1):
            metadata = doc.metadata
            namespace = str(metadata.get("namespace", "knowledge"))
            citation_id = str(metadata.get("citation_id") or metadata.get("chunk_id") or "unknown")
            key = self._build_citation_key(namespace=namespace, metadata=metadata, citation_id=citation_id)
            if key in seen:
                continue
            seen.add(key)

            score = metadata.get("score")
            normalized_score = float(score) if isinstance(score, int | float) else None
            snippet = truncate_snippet(doc.page_content)
            if snippet:
                citations.append(
                    self._build_citation(
                        index=len(citations) + 1,
                        rank=rank,
                        namespace=namespace,
                        citation_id=citation_id,
                        snippet=snippet,
                        score=normalized_score,
                        metadata=metadata,
                    )
                )

        return citations

    def build_answer_documents(self, documents: list[Document]) -> list[Document]:
        citations = self.citations_from_documents(documents)
        citation_map = {
            self._build_citation_key_from_values(
                namespace=citation.namespace,
                chunk_id=citation.chunk_id,
                citation_id=citation.citation_id,
                document_id=citation.document_id,
            ): citation
            for citation in citations
        }

        formatted_documents: list[Document] = []
        for document in documents:
            namespace = str(document.metadata.get("namespace", "knowledge"))
            citation_id = str(document.metadata.get("citation_id") or document.metadata.get("chunk_id") or "unknown")
            key = self._build_citation_key(
                namespace=namespace,
                metadata=document.metadata,
                citation_id=citation_id,
            )
            citation = citation_map.get(key)
            if citation is None:
                continue
            header = (
                f"[{citation.index}] "
                f"来源类型：{citation.source_kind}；"
                f"来源名称：{citation.source_name}；"
                f"来源路径：{citation.source_path or 'N/A'}；"
                f"分块：{citation.chunk_id or 'N/A'}"
            )
            formatted_documents.append(
                document.model_copy(
                    update={
                        "page_content": f"{header}\n{document.page_content}",
                    }
                )
            )
        return formatted_documents or documents

    def ensure_answer_citation_markers(self, answer: str, citations: list[Citation]) -> str:
        if not citations:
            return answer
        if re.search(r"\[\d+\]", answer):
            return answer
        markers = "".join(f"[{citation.index}]" for citation in citations)
        return f"{answer}\n\n参考来源：{markers}"

    def _build_citation(
        self,
        *,
        index: int,
        rank: int,
        namespace: str,
        citation_id: str,
        snippet: str,
        score: float | None,
        metadata: dict[str, Any],
    ) -> Citation:
        source_kind = self._resolve_source_kind(namespace=namespace, metadata=metadata)
        source_path = self._resolve_source_path(metadata)
        document_id = self._resolve_optional_str(metadata.get("document_id"))
        chunk_id = self._resolve_optional_str(metadata.get("chunk_id")) or (
            citation_id if source_kind == "document_chunk" else None
        )
        chunk_index = self._resolve_int(metadata.get("chunk_index"))
        source_name = self._resolve_source_name(
            source_kind=source_kind,
            citation_id=citation_id,
            source_path=source_path,
            document_id=document_id,
            metadata=metadata,
        )
        return Citation(
            index=index,
            citation_id=citation_id,
            namespace=namespace,
            source_kind=source_kind,
            source_name=source_name,
            source_path=source_path,
            document_id=document_id,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            snippet=snippet,
            score=score,
            vector_score=self._resolve_float(metadata.get("vector_score")),
            keyword_score=self._resolve_float(metadata.get("keyword_score")),
            vector_rank=self._resolve_int(metadata.get("vector_rank")),
            keyword_rank=self._resolve_int(metadata.get("keyword_rank")),
            rerank_score=self._resolve_float(metadata.get("rerank_score")),
            matched_by=self._resolve_matched_by(metadata.get("matched_by")),
            rank=rank,
        )

    def _build_citation_key(
        self,
        *,
        namespace: str,
        metadata: dict[str, Any],
        citation_id: str,
    ) -> tuple[str, str]:
        return self._build_citation_key_from_values(
            namespace=namespace,
            chunk_id=self._resolve_optional_str(metadata.get("chunk_id")),
            citation_id=citation_id,
            document_id=self._resolve_optional_str(metadata.get("document_id")),
        )

    def _build_citation_key_from_values(
        self,
        *,
        namespace: str,
        chunk_id: str | None,
        citation_id: str | None,
        document_id: str | None,
    ) -> tuple[str, str]:
        """用显式字段生成 citation lookup key，避免按列表位置错配。"""
        return namespace, chunk_id or citation_id or document_id or "unknown"

    def _resolve_source_kind(self, *, namespace: str, metadata: dict[str, Any]) -> str:
        if metadata.get("chunk_id") is not None or metadata.get("document_id") is not None:
            return "document_chunk"
        source_kind_map = {
            "products": "product",
            "reviews": "review",
            "orders": "order",
            "inventory": "inventory",
            "product_detail": "product_detail",
            "documents": "document_chunk",
        }
        return source_kind_map.get(namespace, namespace)

    def _resolve_source_name(
        self,
        *,
        source_kind: str,
        citation_id: str,
        source_path: str | None,
        document_id: str | None,
        metadata: dict[str, Any],
    ) -> str:
        if source_kind == "document_chunk":
            if source_path:
                return Path(source_path).name
            if document_id:
                return document_id
        for field_name in ("title", "name", "product_name", "order_id", "product_id", "review_id"):
            resolved = self._resolve_optional_str(metadata.get(field_name))
            if resolved:
                return resolved
        return citation_id

    def _resolve_source_path(self, metadata: dict[str, Any]) -> str | None:
        source_path = self._resolve_optional_str(metadata.get("source_path"))
        if source_path:
            return source_path
        for field_name in ("product_id", "review_id", "order_id"):
            resolved = self._resolve_optional_str(metadata.get(field_name))
            if resolved:
                return resolved
        return None

    def _resolve_optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, int | float):
            return str(value)
        return None

    def _resolve_int(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    def _resolve_float(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _resolve_matched_by(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if isinstance(item, str) and item]


class SceneRegistry:
    """维护可用场景定义，并解析当前激活场景。"""

    def __init__(self, definitions: list[SceneDefinition], default_scene: str) -> None:
        self._definitions = {definition.scene: definition for definition in definitions}
        self._default_scene = default_scene
        if default_scene not in self._definitions:
            supported = ", ".join(sorted(self._definitions))
            raise ValueError(
                f"Unknown active scene '{default_scene}'. Expected one of: {supported}."
            )

    @property
    def default_scene(self) -> str:
        """返回默认场景标识。"""
        return self._default_scene

    def list_definitions(self) -> tuple[SceneDefinition, ...]:
        """返回全部已注册场景定义。"""
        return tuple(self._definitions.values())

    def is_supported(self, scene: str) -> bool:
        """检查场景是否已注册。"""
        return scene in self._definitions

    def get_definition(self, scene: str) -> SceneDefinition:
        """按场景标识返回场景定义。"""
        return self._definitions[scene]

    def get_default_definition(self) -> SceneDefinition:
        """返回默认场景定义。"""
        return self.get_definition(self._default_scene)


class ChatService:
    """执行单个场景下的检索、生成和会话持久化流程。"""

    def __init__(
        self,
        *,
        scene_definition: SceneDefinition,
        app_settings: AppSettings | None = None,
        session_store: SQLiteSessionStore | None = None,
        context_builder: PromptContextBuilder | None = None,
        model: RetrievalChainModel | None = None,
    ) -> None:
        """初始化场景聊天服务依赖。"""
        self.settings = app_settings or settings
        self.scene_definition = scene_definition
        self.session_store = session_store or SQLiteSessionStore(self.settings)
        self.context_builder = context_builder or PromptContextBuilder(
            window_size=self.settings.session.window_size
        )
        self.model = model or model_client
        self._rag_answer_template = build_rag_answer_prompt_template(
            system_prompt=scene_definition.system_prompt
        )
        self._retriever = scene_definition.build_retriever()
        self._retrieval_executor = RetrievalExecutor(
            scene_definition=scene_definition,
            retriever=self._retriever,
        )
        self._citation_mapper = CitationMapper()
        self._answer_base_runnables: dict[TaskComplexity, Any] = {}

    def chat(self, payload: ChatRequest) -> ChatResponse:
        """执行一次完整对话流程，并返回统一结构。"""
        prepared = self._prepare_chat_turn(payload)
        answer, citations = self._generate_answer(prepared)
        self._persist_turn(prepared=prepared, answer=answer, citations=citations)
        return self._build_chat_response(
            prepared=prepared,
            answer=answer,
            citations=citations,
        )

    def chat_stream(self, payload: ChatRequest) -> Iterator[ChatStreamEvent]:
        """执行一次流式对话流程，并产出结构化事件。"""
        prepared = self._prepare_chat_turn(payload)
        yield ChatStreamEvent(
            event="start",
            data={
                "session_id": prepared.session_id,
                "request_id": prepared.request_id,
                "knowledge_used": prepared.knowledge_used,
                "scene": prepared.scene_metadata.scene,
                "agent": prepared.scene_metadata.agent,
            },
        )
        yield ChatStreamEvent(event="history", data=self._build_history_event(prepared))
        yield ChatStreamEvent(event="tool", data=self._build_tool_event(prepared))

        if prepared.answer_mode != "evidence_answer":
            answer, citations = self._build_non_evidence_answer(prepared)
            yield ChatStreamEvent(event="chunk", data={"delta": answer})
        else:
            answer_parts: list[str] = []
            for chunk in self._stream_model_answer(prepared):
                answer_parts.append(chunk)
                yield ChatStreamEvent(event="chunk", data={"delta": chunk})
            answer, citations = self._finalize_streamed_answer(prepared, answer_parts)

        self._persist_turn(prepared=prepared, answer=answer, citations=citations)
        response = self._build_chat_response(
            prepared=prepared,
            answer=answer,
            citations=citations,
        )
        yield ChatStreamEvent(event="done", data=response.model_dump())

    def _ensure_session_ready(
        self,
        *,
        session_id: str,
        timestamp: str,
        request_id: str,
        scene: str,
    ) -> None:
        """创建或续期当前会话。"""
        self.session_store.cleanup_expired_sessions(now=timestamp)
        session = self.session_store.get_session(session_id)
        if session is None:
            self.session_store.create_session(
                session_id=session_id,
                scene=scene,
                mounted_knowledge_sources=DEFAULT_MOUNTED_KNOWLEDGE_SOURCES,
                now=timestamp,
            )
            return
        if session.status == "expired":
            raise ChatServiceError(
                status_code=409,
                code="SESSION_EXPIRED",
                message="Session has expired. Please create a new session before continuing.",
                request_id=request_id,
            )
        if session.scene != scene:
            raise ChatServiceError(
                status_code=409,
                code="SCENE_SESSION_MISMATCH",
                message="Session is bound to a different scene. Please create a new session for this scene.",
                request_id=request_id,
            )
        self.session_store.touch_session(session_id=session_id, now=timestamp)

    def _scene_metadata(self) -> SceneMetadata:
        """从场景定义中提取响应元数据。"""
        default_agent = self.scene_definition.metadata.get("default_agent")
        return SceneMetadata(
            scene=self.scene_definition.scene,
            agent=str(default_agent) if isinstance(default_agent, str) else None,
        )

    def _prepare_chat_turn(self, payload: ChatRequest) -> PreparedChatTurn:
        """准备一次对话执行所需的共享上下文。"""
        request_id = uuid4().hex
        session_id = payload.session_id or uuid4().hex
        timestamp = datetime.now(UTC).isoformat()
        resolved_scene = self.scene_definition.scene

        self._ensure_session_ready(
            session_id=session_id,
            timestamp=timestamp,
            request_id=request_id,
            scene=resolved_scene,
        )
        session = self.session_store.get_session(session_id)
        mounted_knowledge_sources = (
            session.mounted_knowledge_sources
            if session is not None
            else DEFAULT_MOUNTED_KNOWLEDGE_SOURCES
        )
        retrieval_result = self._retrieval_executor.retrieve(
            payload.message,
            mounted_knowledge_sources=mounted_knowledge_sources,
        )
        candidate_documents = retrieval_result.documents
        candidate_citations = self._citation_mapper.citations_from_documents(candidate_documents)
        knowledge_used = self._can_answer_with_evidence(
            final_decision=retrieval_result.final_decision,
            citations=candidate_citations,
        )
        final_decision = self._resolve_prepared_final_decision(
            final_decision=retrieval_result.final_decision,
            knowledge_used=knowledge_used,
        )
        documents = candidate_documents if knowledge_used else []
        citations = candidate_citations if knowledge_used else []
        retrieval_trace = self._build_prepared_retrieval_trace(
            retrieval_trace=retrieval_result.retrieval_trace,
            citations=citations,
            knowledge_used=knowledge_used,
            final_decision=final_decision,
        )
        return PreparedChatTurn(
            session_id=session_id,
            request_id=request_id,
            timestamp=timestamp,
            user_message=payload.message,
            documents=documents,
            tool_event=retrieval_result.tool_event,
            retrieval_trace=retrieval_trace,
            citations=citations,
            knowledge_used=knowledge_used,
            scene_metadata=self._scene_metadata(),
            complexity=(
                self.scene_definition.infer_complexity(payload.message)
                if knowledge_used
                else None
            ),
            final_decision=final_decision,
            follow_up_question=retrieval_result.follow_up_question,
            answer_mode=self._resolve_answer_mode(
                final_decision=final_decision,
                knowledge_used=knowledge_used,
            ),
        )

    def _can_answer_with_evidence(
        self,
        *,
        final_decision: RuntimeFinalDecision | None,
        citations: list[Citation],
    ) -> bool:
        """只允许最终决策和有效引用同时满足时进入证据回答链。"""
        return final_decision == "answer_with_evidence" and len(citations) > 0

    def _build_prepared_retrieval_trace(
        self,
        *,
        retrieval_trace: RetrievalTrace,
        citations: list[Citation],
        knowledge_used: bool,
        final_decision: RuntimeFinalDecision | None,
    ) -> RetrievalTrace:
        """构造最终响应 trace，并保留轮次级诊断信息。"""
        return retrieval_trace.model_copy(
            update={
                "citations": citations,
                "knowledge_used": knowledge_used,
                "final_decision": final_decision,
                # 顶层 top_k_chunks 只代表最终采纳证据，非证据分支清空但保留 rounds。
                "top_k_chunks": retrieval_trace.top_k_chunks if knowledge_used else [],
            }
        )

    def _resolve_prepared_final_decision(
        self,
        *,
        final_decision: RuntimeFinalDecision | None,
        knowledge_used: bool,
    ) -> RuntimeFinalDecision | None:
        """将无有效 citation 的证据候选收敛为 no_evidence，保证 trace 解释最终分支。"""
        if final_decision == "answer_with_evidence" and not knowledge_used:
            return "no_evidence"
        return final_decision

    def _resolve_answer_mode(
        self,
        *,
        final_decision: RuntimeFinalDecision | None,
        knowledge_used: bool,
    ) -> AnswerMode:
        """根据最终决策选择回答分支，供 JSON 与 SSE 共用。"""
        if knowledge_used:
            return "evidence_answer"
        if final_decision == "ask_user":
            return "follow_up"
        return "fallback"

    def _generate_answer(self, prepared: PreparedChatTurn) -> tuple[str, list[Citation]]:
        """根据准备结果生成最终答案。"""
        if prepared.answer_mode != "evidence_answer":
            return self._build_non_evidence_answer(prepared)
        return self._invoke_answer_template(prepared=prepared)

    def _invoke_answer_template(
        self,
        *,
        prepared: PreparedChatTurn,
    ) -> tuple[str, list[Citation]]:
        """调用模型链生成答案，并返回答案与引用。"""
        runnable = self._get_answer_runnable(prepared)
        try:
            answer = self.model.invoke_runnable(
                runnable,
                self._build_answer_variables(prepared),
                config=self._build_runnable_config(prepared.session_id),
            )
        except ValueError as exc:
            if str(exc) == "Model returned empty content":
                raise ChatServiceError(
                    status_code=502,
                    code="MODEL_EMPTY_RESPONSE",
                    message="Model returned empty response.",
                    request_id=prepared.request_id,
                ) from exc
            raise ChatServiceError(
                status_code=502,
                code="MODEL_INVOCATION_FAILED",
                message="Model invocation failed. Please retry later.",
                request_id=prepared.request_id,
            ) from exc
        except Exception as exc:
            raise ChatServiceError(
                status_code=502,
                code="MODEL_INVOCATION_FAILED",
                message="Model invocation failed. Please retry later.",
                request_id=prepared.request_id,
            ) from exc

        return self._finalize_answer_text(answer, prepared.citations)

    def _stream_model_answer(self, prepared: PreparedChatTurn) -> Iterator[str]:
        """对最终答案生成阶段执行流式调用。"""
        runnable = self._get_answer_runnable(prepared)
        try:
            for chunk in self.model.stream_runnable(
                runnable,
                self._build_answer_variables(prepared),
                config=self._build_runnable_config(prepared.session_id),
            ):
                yield str(chunk)
        except ValueError as exc:
            message = str(exc)
            if message == "Model returned empty streaming content":
                raise ChatServiceError(
                    status_code=502,
                    code="MODEL_EMPTY_RESPONSE",
                    message="Model returned empty response.",
                    request_id=prepared.request_id,
                ) from exc
            raise ChatServiceError(
                status_code=502,
                code="MODEL_INVOCATION_FAILED",
                message="Model invocation failed. Please retry later.",
                request_id=prepared.request_id,
            ) from exc
        except Exception as exc:
            raise ChatServiceError(
                status_code=502,
                code="MODEL_INVOCATION_FAILED",
                message="Model invocation failed. Please retry later.",
                request_id=prepared.request_id,
            ) from exc

    def _finalize_streamed_answer(
        self,
        prepared: PreparedChatTurn,
        answer_parts: list[str],
    ) -> tuple[str, list[Citation]]:
        """将流式片段拼接为最终权威答案。"""
        joined_answer = "".join(answer_parts).strip()
        if not joined_answer:
            raise ChatServiceError(
                status_code=502,
                code="MODEL_EMPTY_RESPONSE",
                message="Model returned empty response.",
                request_id=prepared.request_id,
            )
        return self._finalize_answer_text(joined_answer, prepared.citations)

    def _build_non_evidence_answer(self, prepared: PreparedChatTurn) -> tuple[str, list[Citation]]:
        """构造不携带引用的追问或降级回答。"""
        if prepared.answer_mode == "follow_up":
            return self._build_follow_up_answer(prepared), []
        return self._build_fallback_answer(prepared), []

    def _build_follow_up_answer(self, prepared: PreparedChatTurn) -> str:
        """解析 ask_user 分支追问文案，缺失时回退到 scene no-hit 文案。"""
        if prepared.follow_up_question:
            return prepared.follow_up_question
        return self._build_fallback_answer(prepared)

    def _build_fallback_answer(self, prepared: PreparedChatTurn) -> str:
        """构造无命中或降级时的 fallback 回答。"""
        policy = self.scene_definition.retrieval_policy
        return self.scene_definition.fallback_policy.message_for_strategy(policy.no_hit_strategy)

    def _finalize_answer_text(
        self,
        answer: str,
        citations: list[Citation],
    ) -> tuple[str, list[Citation]]:
        """统一补齐 citation markers，并返回最终答案与引用。"""
        final_answer = self._citation_mapper.ensure_answer_citation_markers(answer.strip(), citations)
        return final_answer, citations

    def _build_answer_variables(self, prepared: PreparedChatTurn) -> dict[str, Any]:
        """构造最终回答模板需要的变量。"""
        return {
            "context": self._citation_mapper.build_answer_documents(prepared.documents),
            "input": prepared.user_message,
        }

    def _get_answer_base_runnable(self, complexity: TaskComplexity) -> Any:
        """为给定复杂度构建不携带请求上下文的基础回答 runnable。"""
        cached = self._answer_base_runnables.get(complexity)
        if cached is not None:
            return cached

        runnable = self.model.get_runnable(
            complexity=complexity,
            prompt_template=self._rag_answer_template,
        )
        self._answer_base_runnables[complexity] = runnable
        return runnable

    def _get_answer_runnable(self, prepared: PreparedChatTurn) -> RunnableWithMessageHistory:
        """为当前请求构建带消息历史的回答 runnable。"""
        base_runnable = self._get_answer_base_runnable(prepared.complexity or "simple")

        def history_factory(session_id: str) -> SQLiteChatMessageHistory:
            return self._get_session_history(
                session_id,
                request_id=prepared.request_id,
                timestamp=prepared.timestamp,
            )

        runnable = RunnableWithMessageHistory(
            base_runnable,
            history_factory,
            input_messages_key="input",
            history_messages_key="history",
        )
        return runnable

    def _get_session_history(
        self,
        session_id: str,
        *,
        request_id: str,
        timestamp: str,
    ) -> SQLiteChatMessageHistory:
        """解析指定会话的 LangChain message history。"""
        return SQLiteChatMessageHistory(
            session_id,
            store=self.session_store,
            request_id=request_id,
            timestamp=timestamp,
            message_limit=self.settings.session.window_size * 2,
            message_transform=self.context_builder.trim_messages,
        )

    def _build_runnable_config(self, session_id: str) -> dict[str, Any]:
        """构造 RunnableWithMessageHistory 所需的 configurable config。"""
        return {"configurable": {"session_id": session_id}}

    def _build_history_event(self, prepared: PreparedChatTurn) -> dict[str, Any]:
        """构造本轮模型调用前的历史消息快照事件。"""
        messages = self._get_session_history(
            prepared.session_id,
            request_id=prepared.request_id,
            timestamp=prepared.timestamp,
        ).messages
        return {
            "session_id": prepared.session_id,
            "request_id": prepared.request_id,
            "window_size": self.settings.session.window_size,
            "message_count": len(messages),
            "messages": [self._serialize_history_message(message) for message in messages],
        }

    def _build_tool_event(self, prepared: PreparedChatTurn) -> dict[str, Any]:
        """构造 retrieval/tool 阶段的结构化事件。"""
        return {
            **prepared.tool_event,
            "rounds": [round_trace.model_dump() for round_trace in prepared.retrieval_trace.rounds],
            "session_id": prepared.session_id,
            "request_id": prepared.request_id,
            "final_decision": prepared.final_decision,
            "follow_up_question": prepared.follow_up_question,
            "answer_mode": prepared.answer_mode,
            "knowledge_used": prepared.knowledge_used,
            "citations": [citation.model_dump() for citation in prepared.citations],
            "retrieval_trace": prepared.retrieval_trace.model_dump(),
        }

    def _persist_turn(
        self,
        *,
        prepared: PreparedChatTurn,
        answer: str,
        citations: list[Citation],
    ) -> None:
        """以既有语义写入最终对话轮次。"""
        self.session_store.append_turn(
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            user_message=prepared.user_message,
            assistant_answer=answer,
            retrieval_snippets=[citation.model_dump() for citation in citations],
            timestamp=prepared.timestamp,
            persist_messages=not prepared.knowledge_used,
        )

    def _build_chat_response(
        self,
        *,
        prepared: PreparedChatTurn,
        answer: str,
        citations: list[Citation],
    ) -> ChatResponse:
        """统一构造聊天响应。"""
        return ChatResponse(
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            answer=answer,
            knowledge_used=prepared.knowledge_used,
            scene=prepared.scene_metadata.scene,
            agent=prepared.scene_metadata.agent,
            citations=citations,
            retrieval_trace=prepared.retrieval_trace.model_copy(
                update={
                    "citations": citations,
                    "knowledge_used": prepared.knowledge_used,
                    "top_k_chunks": prepared.retrieval_trace.top_k_chunks
                    if prepared.knowledge_used
                    else [],
                }
            ),
        )

    def _serialize_history_message(self, message: BaseMessage) -> dict[str, Any]:
        """将 LangChain message 归一化为稳定的 SSE payload。"""
        return {
            "type": message.type,
            "content": message.content,
        }


class ActiveSceneChatService:
    """统一 `/chat` 入口，通过会话绑定场景分发请求。"""

    def __init__(
        self,
        *,
        scene_registry: SceneRegistry,
        app_settings: AppSettings | None = None,
        knowledge_service: object | None = None,
        session_store: SQLiteSessionStore | None = None,
        context_builder: PromptContextBuilder | None = None,
        model: RetrievalChainModel | None = None,
    ) -> None:
        """初始化运行时依赖，并缓存当前激活场景服务。"""
        del knowledge_service
        self.settings = app_settings or settings
        self.scene_registry = scene_registry
        self.session_store = session_store or SQLiteSessionStore(self.settings)
        self.context_builder = context_builder or PromptContextBuilder(
            window_size=self.settings.session.window_size
        )
        self.model = model or model_client
        self._scene_services: dict[str, ChatService] = {}

    def chat(self, payload: ChatRequest) -> ChatResponse:
        """将请求转发给会话绑定的场景。"""
        scene = self.resolve_session_scene(payload.session_id)
        return self._get_scene_service(scene).chat(payload)

    def chat_stream(self, payload: ChatRequest) -> Iterator[ChatStreamEvent]:
        """将流式请求转发给会话绑定的场景。"""
        scene = self.resolve_session_scene(payload.session_id)
        yield from self._get_scene_service(scene).chat_stream(payload)

    def list_scenes(self) -> tuple[SceneDefinition, ...]:
        """列出所有可用场景定义。"""
        return self.scene_registry.list_definitions()

    def default_scene(self) -> str:
        """返回默认场景标识。"""
        return self.scene_registry.default_scene

    def validate_scene(self, scene: str) -> str:
        """校验并返回合法场景标识。"""
        if not self.scene_registry.is_supported(scene):
            supported = ", ".join(
                definition.scene for definition in self.scene_registry.list_definitions()
            )
            raise ValueError(f"Unknown scene '{scene}'. Expected one of: {supported}.")
        return scene

    def create_session(
        self,
        scene: str | None = None,
        mounted_knowledge_sources: list[str] | tuple[str, ...] | None = None,
    ) -> SessionRecord:
        """创建绑定场景的新会话，并保存规范化后的挂载知识源。"""
        resolved_scene = self.validate_scene(scene or self.default_scene())
        resolved_sources = self.validate_mounted_knowledge_sources(mounted_knowledge_sources)
        session_id = uuid4().hex
        return self.session_store.create_session(
            session_id=session_id,
            scene=resolved_scene,
            mounted_knowledge_sources=resolved_sources,
        )

    def validate_mounted_knowledge_sources(
        self,
        mounted_knowledge_sources: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """校验并规范化会话挂载知识源。"""
        return normalize_mounted_knowledge_sources(mounted_knowledge_sources)

    def default_mounted_knowledge_sources(self) -> tuple[str, ...]:
        """返回系统默认挂载的知识源列表。"""
        return DEFAULT_MOUNTED_KNOWLEDGE_SOURCES

    def resolve_session_scene(self, session_id: str | None) -> str:
        """解析会话绑定场景；无会话时返回默认场景。"""
        if not session_id:
            return self.default_scene()
        session = self.session_store.get_session(session_id)
        if session is None:
            raise ChatServiceError(
                status_code=404,
                code="SESSION_NOT_FOUND",
                message="Session was not found. Please create a new session before continuing.",
                request_id="N/A",
            )
        return self.validate_scene(session.scene)

    def _get_scene_service(self, scene: str) -> ChatService:
        """按场景懒加载 ChatService。"""
        cached = self._scene_services.get(scene)
        if cached is not None:
            return cached

        service = ChatService(
            scene_definition=self.scene_registry.get_definition(scene),
            app_settings=self.settings,
            session_store=self.session_store,
            context_builder=self.context_builder,
            model=self.model,
        )
        self._scene_services[scene] = service
        return service


SceneChatService = ActiveSceneChatService


def build_default_scene_registry(
    *,
    app_settings: AppSettings | None = None,
    knowledge_service: object | None = None,
    document_retrieval_service: DocumentRetrievalService | None = None,
    generic_business_extensions: tuple[GenericAssistantBusinessExtension, ...] | None = None,
    include_default_business_extensions: bool = True,
) -> SceneRegistry:
    """构建默认场景注册表。"""
    resolved_settings = app_settings or settings
    definitions = list(
        build_default_scene_definitions(
            app_settings=resolved_settings,
            knowledge_service=knowledge_service,
            document_retrieval_service=document_retrieval_service,
            generic_business_extensions=generic_business_extensions,
            include_default_business_extensions=include_default_business_extensions,
        )
    )
    return SceneRegistry(definitions=definitions, default_scene=resolved_settings.app.active_scene)


def create_chat_service(
    app_settings: AppSettings | None = None,
    knowledge_service: object | None = None,
    document_retrieval_service: DocumentRetrievalService | None = None,
    generic_business_extensions: tuple[GenericAssistantBusinessExtension, ...] | None = None,
    include_default_business_extensions: bool = True,
    session_store: SQLiteSessionStore | None = None,
    context_builder: PromptContextBuilder | None = None,
    model: ModelClient | None = None,
) -> ActiveSceneChatService:
    """聊天服务工厂函数，返回统一场景运行时服务。"""
    resolved_settings = app_settings or settings
    scene_registry = build_default_scene_registry(
        app_settings=resolved_settings,
        knowledge_service=knowledge_service,
        document_retrieval_service=document_retrieval_service,
        generic_business_extensions=generic_business_extensions,
        include_default_business_extensions=include_default_business_extensions,
    )
    return ActiveSceneChatService(
        scene_registry=scene_registry,
        app_settings=resolved_settings,
        knowledge_service=knowledge_service,
        session_store=session_store,
        context_builder=context_builder,
        model=model,
    )

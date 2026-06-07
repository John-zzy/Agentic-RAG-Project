from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict

from langchain_core.documents import Document

from backend.platform.rag.contracts import RetrievalPlan, RetrievalResult
from backend.platform.rag.orchestration.decisions import (
    RetrievalDecisionLogEntry,
    SufficiencyDecision,
)
from backend.platform.rag.pre_retrieval.query_rewrite import QueryRewrite


class AgenticRagGraphState(TypedDict, total=False):
    """Agentic RAG retrieval graph 的可序列化状态。"""

    query: str
    active_query: str
    rewritten_query: NotRequired[str | None]
    selected_tool: NotRequired[str | None]
    candidate_tools: NotRequired[tuple[str, ...]]
    attempted_tools: NotRequired[tuple[str, ...]]
    candidate_docs: NotRequired[list[dict[str, Any]]]
    tool_observation: NotRequired[dict[str, Any] | None]
    retrieval_trace: NotRequired[dict[str, Any]]
    knowledge_used: NotRequired[bool]
    citations: NotRequired[list[dict[str, Any]]]
    final_decision: NotRequired[SufficiencyDecision | None]
    final_decision_label: NotRequired[str | None]
    follow_up_question: NotRequired[str | None]
    plan: NotRequired[RetrievalPlan]
    results: NotRequired[list[RetrievalResult]]
    documents: NotRequired[list[Document]]
    current_result: NotRequired[RetrievalResult | None]
    current_decision: NotRequired[SufficiencyDecision | None]
    current_rewrite: NotRequired[QueryRewrite | None]
    rounds: NotRequired[list[dict[str, Any]]]
    decision_log: NotRequired[list[RetrievalDecisionLogEntry]]
    final_plan: NotRequired[RetrievalPlan | None]
    success: NotRequired[bool]
    exit_reason: NotRequired[str | None]
    route_next_action: NotRequired[str | None]
    metadata: NotRequired[dict[str, Any]]
    filters: NotRequired[dict[str, Any]]
    top_k: NotRequired[int | None]
    min_relevance_score: NotRequired[float | None]
    recall_strategy: NotRequired[str]
    rerank_enabled: NotRequired[bool]
    rerank_top_n: NotRequired[int | None]
    max_rounds: NotRequired[int]


def build_agentic_rag_graph_state(
    *,
    query: str,
    selected_tool: str | None = None,
    candidate_tools: Sequence[str] | None = None,
    filters: Mapping[str, Any] | None = None,
    top_k: int | None = None,
    min_relevance_score: float | None = None,
    recall_strategy: str = "hybrid",
    rerank_enabled: bool = False,
    rerank_top_n: int | None = None,
    max_rounds: int = 3,
    metadata: Mapping[str, Any] | None = None,
) -> AgenticRagGraphState:
    """创建一份初始图状态，后续节点只做增量回写。"""
    if not query:
        raise ValueError("query is required for Agentic RAG graph state.")
    return {
        "query": query,
        "active_query": query,
        "selected_tool": selected_tool,
        "candidate_tools": tuple(candidate_tools or ()),
        "attempted_tools": (),
        "candidate_docs": [],
        "tool_observation": None,
        "retrieval_trace": {},
        "knowledge_used": False,
        "citations": [],
        "final_decision": None,
        "final_decision_label": None,
        "follow_up_question": None,
        "results": [],
        "documents": [],
        "current_result": None,
        "current_decision": None,
        "current_rewrite": None,
        "rounds": [],
        "decision_log": [],
        "final_plan": None,
        "success": False,
        "exit_reason": None,
        "route_next_action": None,
        "metadata": dict(metadata or {}),
        "filters": dict(filters or {}),
        "top_k": top_k,
        "min_relevance_score": min_relevance_score,
        "recall_strategy": recall_strategy,
        "rerank_enabled": rerank_enabled,
        "rerank_top_n": rerank_top_n,
        "max_rounds": max_rounds,
    }


def build_round_snapshot(
    *,
    plan: RetrievalPlan,
    result: RetrievalResult,
    decision: SufficiencyDecision,
    results: Sequence[RetrievalResult],
    documents: Sequence[Document],
    rewrite: QueryRewrite | None = None,
) -> dict[str, Any]:
    """把单轮检索事实和可观测 trace 一次性封装。"""
    document_trace = result.metadata.get("document_retrieval_trace")
    document_trace = document_trace if isinstance(document_trace, Mapping) else {}
    return {
        "plan": plan.model_dump(),
        "results": [item.model_dump() for item in results],
        "documents": [_document_to_payload(document) for document in documents],
        "result": result.model_dump(),
        "decision": decision.model_dump(),
        "rewrite": rewrite.model_dump() if rewrite else None,
        "trace": {
            "round_index": plan.round_index,
            "tool_name": result.tool_name,
            "query": result.query,
            "rewritten_query": rewrite.query if rewrite else None,
            "decision": decision.next_action,
            "reason": decision.reason,
            "is_sufficient": decision.is_sufficient,
            "result_success": result.success,
            "result_count": len(result.records),
            "document_count": len(result.documents),
            "raw_candidates_count": document_trace.get("raw_candidates_count", len(result.records)),
            "filtered_candidates_count": document_trace.get(
                "filtered_candidates_count",
                len(result.documents),
            ),
            "citations": _citations_to_dicts(result.citations),
            "top_k_chunks": document_trace.get("top_k_chunks", []),
            "rerank": result.metadata.get("rerank"),
        },
    }


def build_decision_log_entry(
    *,
    round_snapshot: Mapping[str, Any],
    exit_reason: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RetrievalDecisionLogEntry:
    """把一轮 trace 归一化为可审计的决策日志。"""
    decision = round_snapshot["decision"]
    result = round_snapshot["result"]
    metadata = dict(decision.get("metadata") or {})
    metadata["exit_reason"] = exit_reason
    if extra_metadata:
        metadata.update(extra_metadata)
    return RetrievalDecisionLogEntry(
        round_index=int(round_snapshot["trace"]["round_index"]),
        tool_name=str(result["tool_name"]),
        query=str(result["query"]),
        rewritten_query=round_snapshot["trace"].get("rewritten_query"),
        result_count=len(result.get("records") or []),
        result_success=bool(result.get("success")),
        result_confidence=result.get("confidence"),
        decision=str(decision["next_action"]),
        is_sufficient=bool(decision["is_sufficient"]),
        reason=str(decision["reason"]),
        suggested_tool=decision.get("suggested_tool"),
        follow_up_question=decision.get("follow_up_question"),
        metadata=metadata,
    )


def build_retrieval_trace_snapshot(
    *,
    query: str,
    final_query: str,
    success: bool,
    final_decision_label: str,
    knowledge_used: bool,
    citations: Sequence[Mapping[str, Any]],
    rounds: Sequence[Mapping[str, Any]],
    decision_log: Sequence[RetrievalDecisionLogEntry | Mapping[str, Any]],
    exit_reason: str,
    follow_up_question: str | None,
    candidate_tools: Sequence[str],
) -> dict[str, Any]:
    """把最终状态压成对外可读的 retrieval_trace。"""
    return {
        "original_query": query,
        "final_query": final_query,
        "success": success,
        "final_decision": final_decision_label,
        "knowledge_used": knowledge_used,
        "citations": _citations_to_dicts(citations),
        "rounds": [dict(round_snapshot["trace"]) for round_snapshot in rounds],
        "decision_log": [
            item.model_dump() if hasattr(item, "model_dump") else dict(item)
            for item in decision_log
        ],
        "exit_reason": exit_reason,
        "follow_up_question": follow_up_question,
        "candidate_tools": list(candidate_tools),
        "tool_call_count": len(rounds),
    }


def build_tool_observation_snapshot(
    *,
    tool_name: str,
    result: RetrievalResult,
    retrieval_trace: Mapping[str, Any],
    final_decision_label: str,
    knowledge_used: bool,
    follow_up_question: str | None,
    success: bool,
    citations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """把子图结果包装成稳定的工具观察快照。"""
    documents = _documents_to_payload(result.documents)
    citations = _citations_to_dicts(citations or ([] if not knowledge_used else result.citations))
    return {
        "tool_name": tool_name,
        "success": success,
        "output": {
            "query": result.query,
            "records": [dict(record) for record in result.records],
            "documents": documents,
            "knowledge_used": knowledge_used,
            "confidence": result.confidence,
            "final_decision": final_decision_label,
            "follow_up_question": follow_up_question,
        },
        "result_summary": _build_summary(
            tool_name=tool_name,
            success=success,
            record_count=len(result.records),
            final_decision=final_decision_label,
            error=result.error,
        ),
        "citations": citations,
        "trace": {"retrieval_trace": dict(retrieval_trace)},
        "retryable": final_decision_label == "retrieval_failed",
        "requires_user": final_decision_label == "ask_user",
        "user_prompt": follow_up_question if final_decision_label == "ask_user" else None,
        "error": result.error if not success else None,
        "metadata": {
            "final_decision": final_decision_label,
            "knowledge_used": knowledge_used,
        },
    }


def build_document_citations(documents: Sequence[Document]) -> list[dict[str, Any]]:
    """当结果没有显式 citations 时，用文档兜底生成可引用片段。"""
    return [
        {
            "citation_id": _document_citation_id(document),
            "snippet": document.page_content[:500],
            "source_type": str(
                document.metadata.get("namespace")
                or document.metadata.get("source_type")
                or "knowledge"
            ),
            "metadata": dict(document.metadata),
        }
        for document in documents
    ]


def _citations_to_dicts(citations: Sequence[Any]) -> list[dict[str, Any]]:
    return [_citation_to_dict(citation) for citation in citations]


def _citation_to_dict(citation: Any) -> dict[str, Any]:
    if hasattr(citation, "model_dump"):
        return citation.model_dump()
    payload = dict(citation)
    payload.setdefault("metadata", {})
    return payload


def _documents_to_payload(documents: Sequence[Document]) -> list[dict[str, Any]]:
    return [
        {
            "page_content": document.page_content,
            "metadata": dict(document.metadata),
        }
        for document in documents
    ]


def _document_to_payload(document: Document) -> dict[str, Any]:
    return {
        "page_content": document.page_content,
        "metadata": dict(document.metadata),
    }


def _document_citation_id(document: Document) -> str:
    return str(
        document.metadata.get("citation_id")
        or document.metadata.get("chunk_id")
        or document.metadata.get("document_id")
        or document.metadata.get("source")
        or document.id
        or "unknown"
    )


def _build_summary(
    *,
    tool_name: str,
    success: bool,
    record_count: int,
    final_decision: str,
    error: str | None,
) -> str:
    if success:
        return f"{tool_name} returned {record_count} evidence record(s)."
    if error:
        return f"{tool_name} failed: {error}"
    return f"{tool_name} finished with decision: {final_decision}."


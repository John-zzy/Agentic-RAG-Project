from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from backend.platform.agent_runtime.contracts import ToolExecutionMetadata, ToolObservation
from backend.platform.agent_runtime.validation import validate_tool_input
from backend.platform.rag.contracts import RecallStrategy, RetrievalCitation, RetrievalResult
from backend.platform.rag.orchestration.agentic import AgenticRetrievalOutcome


RAGToolKind = Literal["native_rag", "agentic_rag"]
NATIVE_RAG_TOOL_NAME = "native_rag_search"
AGENTIC_RAG_TOOL_NAME = "agentic_rag_search"
RAG_FINAL_DECISION_ANSWER = "answer_with_evidence"
RAG_FINAL_DECISION_ASK_USER = "ask_user"
RAG_FINAL_DECISION_FAILED = "retrieval_failed"
RAG_FINAL_DECISION_NO_EVIDENCE = "no_evidence"


class RAGToolInput(BaseModel):
    """顶层 Agent 调用 RAG 工具时使用的统一输入。"""

    query: str = Field(min_length=1)
    selected_tool: str | None = None
    candidate_tools: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = Field(default=None, ge=1)
    min_relevance_score: float | None = None
    recall_strategy: RecallStrategy = "hybrid"
    rerank_enabled: bool = False
    rerank_top_n: int | None = Field(default=None, ge=1)


class NativeRagToolAdapter:
    """将 Native RAG 封装为 ToolExecutor 可调用工具。"""

    kind: RAGToolKind = "native_rag"

    def __init__(
        self,
        *,
        retriever: Any,
        name: str = NATIVE_RAG_TOOL_NAME,
        args_schema: type[BaseModel] | None = None,
        description: str = "Run native RAG retrieval.",
        candidate_tools: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.retriever = retriever
        self.args_schema = args_schema or RAGToolInput
        self.description = description
        self.candidate_tools = candidate_tools

    def invoke(
        self,
        input_payload: Mapping[str, Any] | None = None,
        *,
        execution: ToolExecutionMetadata | None = None,
    ) -> ToolObservation:
        payload = validate_tool_input(
            tool_name=self.name,
            input_payload=input_payload,
            args_schema=self.args_schema,
        )
        query = str(payload.pop("query"))
        result = self._retrieve(query=query, payload=payload)
        return retrieval_result_to_observation(
            adapter_name=self.name,
            result=result,
            execution=execution,
            candidate_tools=_resolve_candidate_tools(payload, self.candidate_tools),
        )

    def execute(
        self,
        input_payload: Mapping[str, Any] | None = None,
        *,
        context: Any | None = None,
        execution: ToolExecutionMetadata | None = None,
    ) -> ToolObservation:
        del context
        return self.invoke(input_payload, execution=execution)

    def _retrieve(self, *, query: str, payload: Mapping[str, Any]) -> RetrievalResult:
        retrieval_tool = _resolve_native_retrieval_tool(
            self.retriever,
            payload=payload,
            default_candidate_tools=self.candidate_tools,
        )
        retrieval_kwargs = _retrieval_kwargs(payload)
        if retrieval_tool is not None:
            return retrieval_tool.retrieve(
                **_supported_kwargs(retrieval_tool.retrieve, query=query, **retrieval_kwargs)
            )
        if hasattr(self.retriever, "retrieve"):
            retrieve = self.retriever.retrieve
            return retrieve(**_supported_kwargs(retrieve, query=query, **retrieval_kwargs))
        if hasattr(self.retriever, "search"):
            search = self.retriever.search
            documents = list(search(**_supported_kwargs(search, query=query, **retrieval_kwargs)))
            return _retrieval_result_from_documents(
                tool_name="search",
                query=query,
                documents=documents,
            )
        if isinstance(self.retriever, BaseRetriever):
            documents = list(self.retriever.invoke(query))
            return _retrieval_result_from_documents(
                tool_name=type(self.retriever).__name__,
                query=query,
                documents=documents,
            )
        raise TypeError("Retriever does not support native RAG execution.")


class AgenticRagToolAdapter:
    """将 Agentic Retrieval 封装为单个顶层 Agent 工具调用。"""

    kind: RAGToolKind = "agentic_rag"

    def __init__(
        self,
        *,
        retriever: Any,
        name: str = AGENTIC_RAG_TOOL_NAME,
        args_schema: type[BaseModel] | None = None,
        description: str = "Run agentic RAG retrieval.",
        candidate_tools: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.retriever = retriever
        self.args_schema = args_schema or RAGToolInput
        self.description = description
        self.candidate_tools = candidate_tools

    def invoke(
        self,
        input_payload: Mapping[str, Any] | None = None,
        *,
        execution: ToolExecutionMetadata | None = None,
    ) -> ToolObservation:
        payload = validate_tool_input(
            tool_name=self.name,
            input_payload=input_payload,
            args_schema=self.args_schema,
        )
        query = str(payload.pop("query"))
        payload["candidate_tools"] = tuple(
            _resolve_candidate_tools(payload, self.candidate_tools)
        )
        retrieve_with_trace = self.retriever.retrieve_with_trace
        outcome = retrieve_with_trace(
            **_supported_kwargs(
                retrieve_with_trace,
                query=query,
                **_strip_none_values(payload),
            )
        )
        return agentic_outcome_to_observation(
            adapter_name=self.name,
            outcome=outcome,
            execution=execution,
        )

    def execute(
        self,
        input_payload: Mapping[str, Any] | None = None,
        *,
        context: Any | None = None,
        execution: ToolExecutionMetadata | None = None,
    ) -> ToolObservation:
        del context
        return self.invoke(input_payload, execution=execution)


# 保留全大写 RAG 命名，便于后续代码按常量风格导入。
NativeRAGToolAdapter = NativeRagToolAdapter
AgenticRAGToolAdapter = AgenticRagToolAdapter
RAGToolAdapter = NativeRagToolAdapter | AgenticRagToolAdapter


def build_rag_tool_adapters(
    *,
    retriever: Any,
    candidate_tools: tuple[str, ...] = (),
) -> tuple[NativeRagToolAdapter | AgenticRagToolAdapter, ...]:
    """按 retriever 能力生成 ToolExecutor 可注册的 RAG 工具。"""
    tools: list[NativeRagToolAdapter | AgenticRagToolAdapter] = [
        NativeRagToolAdapter(retriever=retriever, candidate_tools=candidate_tools)
    ]
    if hasattr(retriever, "retrieve_with_trace"):
        tools.append(
            AgenticRagToolAdapter(retriever=retriever, candidate_tools=candidate_tools)
        )
    return tuple(tools)


def retrieval_result_to_observation(
    *,
    adapter_name: str,
    result: RetrievalResult,
    execution: ToolExecutionMetadata | None = None,
    candidate_tools: tuple[str, ...] = (),
) -> ToolObservation:
    """将 Native RAG RetrievalResult 归一化为 ToolObservation。"""
    citations = _result_citations(result) if result.success else []
    knowledge_used = bool(result.success and citations)
    final_decision = _resolve_native_final_decision(result=result, knowledge_used=knowledge_used)
    retrieval_trace = _base_retrieval_trace(
        query=result.query,
        final_query=result.query,
        success=result.success,
        final_decision=final_decision,
        knowledge_used=knowledge_used,
        citations=citations,
        records=result.records,
        metadata=result.metadata,
    )
    retrieval_trace.update(
        {
            "tool_call_count": 1,
            "candidate_tools": list(candidate_tools),
            "rounds": [_native_round_trace(result)],
        }
    )
    # success 表示工具调用本身完成；是否真的采纳证据由 knowledge_used/final_decision 表达。
    # no_evidence 不能被当成工具失败，否则顶层 ReAct/Plan 会误入 failed。
    tool_success = bool(result.success)
    return ToolObservation(
        tool_name=adapter_name,
        success=tool_success,
        output={
            "query": result.query,
            "records": list(result.records),
            "documents": _documents_to_payload(result.documents),
            "knowledge_used": knowledge_used,
            "confidence": result.confidence,
            "final_decision": final_decision,
        },
        result_summary=_build_rag_summary(
            adapter_name=adapter_name,
            success=tool_success,
            record_count=len(result.records),
            final_decision=final_decision,
            error=result.error,
        ),
        citations=citations,
        trace={"retrieval_trace": retrieval_trace},
        retryable=not result.success,
        error=result.error if not result.success else None,
        execution=execution,
        metadata={
            "source_tool_name": result.tool_name,
            "final_decision": final_decision,
            "knowledge_used": knowledge_used,
        },
    )


def agentic_outcome_to_observation(
    *,
    adapter_name: str,
    outcome: AgenticRetrievalOutcome,
    execution: ToolExecutionMetadata | None = None,
) -> ToolObservation:
    """将 Agentic Retrieval 多轮结果压成一个顶层工具观察。"""
    final_decision = _resolve_agentic_final_decision(outcome)
    requires_user = final_decision == RAG_FINAL_DECISION_ASK_USER
    outcome_documents = _outcome_documents(outcome)
    outcome_success = _outcome_success(outcome, documents=outcome_documents)
    citations = _outcome_citations(outcome) if outcome_success else []
    knowledge_used = bool(
        outcome_success
        and final_decision == RAG_FINAL_DECISION_ANSWER
        and (citations or outcome_documents)
    )
    if outcome_success and knowledge_used and not citations:
        citations = _document_citations(outcome_documents)
    retrieval_error = _agentic_error(outcome)
    rounds = [_round_to_trace(round_trace) for round_trace in _outcome_rounds(outcome)]
    decision_log = [
        entry.model_dump() if hasattr(entry, "model_dump") else dict(entry)
        for entry in list(getattr(outcome, "decision_log", []) or [])
    ]
    outcome_results = _outcome_results(outcome)
    retrieval_trace = _base_retrieval_trace(
        query=_outcome_user_query(outcome),
        final_query=_outcome_final_query(outcome),
        success=outcome_success,
        final_decision=final_decision,
        knowledge_used=knowledge_used,
        citations=citations,
        records=[record for result in outcome_results for record in result.records],
        metadata={},
    )
    retrieval_trace.update(
        {
            "rounds": rounds,
            "decision_log": decision_log,
            "exit_reason": getattr(outcome, "exit_reason", None),
            "follow_up_question": getattr(outcome, "follow_up_question", None),
            "candidate_tools": list(_outcome_candidate_tools(outcome)),
            "tool_call_count": len(rounds),
        }
    )
    tool_success = final_decision != RAG_FINAL_DECISION_FAILED
    return ToolObservation(
        tool_name=adapter_name,
        success=tool_success,
        output={
            "query": _outcome_user_query(outcome),
            "final_query": _outcome_final_query(outcome),
            "documents": _documents_to_payload(outcome_documents),
            "knowledge_used": knowledge_used,
            "final_decision": final_decision,
            "follow_up_question": getattr(outcome, "follow_up_question", None),
        },
        result_summary=_build_rag_summary(
            adapter_name=adapter_name,
            success=tool_success,
            record_count=sum(len(result.records) for result in outcome_results),
            final_decision=final_decision,
            error=None,
        ),
        citations=citations,
        trace={
            "retrieval_trace": retrieval_trace,
        },
        retryable=final_decision == RAG_FINAL_DECISION_FAILED,
        requires_user=requires_user,
        user_prompt=getattr(outcome, "follow_up_question", None) if requires_user else None,
        error=(
            retrieval_error or _final_decision_reason(outcome) or RAG_FINAL_DECISION_FAILED
            if final_decision == RAG_FINAL_DECISION_FAILED
            else None
        ),
        execution=execution,
        metadata={
            "final_decision": final_decision,
            "knowledge_used": knowledge_used,
            "exit_reason": getattr(outcome, "exit_reason", None),
        },
    )


def _resolve_native_final_decision(
    *,
    result: RetrievalResult,
    knowledge_used: bool,
) -> str:
    if not result.success:
        return RAG_FINAL_DECISION_FAILED
    if knowledge_used:
        return RAG_FINAL_DECISION_ANSWER
    return RAG_FINAL_DECISION_NO_EVIDENCE


def _resolve_agentic_final_decision(outcome: AgenticRetrievalOutcome) -> str:
    final_decision = getattr(outcome, "final_decision", None)
    next_action = getattr(final_decision, "next_action", None)
    exit_reason = getattr(outcome, "exit_reason", None)
    if exit_reason == "max_rounds_reached":
        return "max_rounds_reached"
    if next_action == "ask_user" or exit_reason == "ask_user":
        return RAG_FINAL_DECISION_ASK_USER
    if bool(getattr(outcome, "success", False)) and bool(
        getattr(final_decision, "is_sufficient", False)
    ):
        return RAG_FINAL_DECISION_ANSWER
    if _agentic_error(outcome):
        return RAG_FINAL_DECISION_FAILED
    if not _outcome_success(outcome, documents=_outcome_documents(outcome)):
        return RAG_FINAL_DECISION_FAILED
    if _outcome_documents(outcome):
        return RAG_FINAL_DECISION_ANSWER
    return RAG_FINAL_DECISION_ANSWER


def _base_retrieval_trace(
    *,
    query: str,
    final_query: str,
    success: bool,
    final_decision: str,
    knowledge_used: bool,
    citations: list[dict[str, Any]],
    records: list[dict[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    nested_trace = (
        metadata.get("retrieval_trace")
        or metadata.get("document_retrieval_trace")
        or {}
    )
    trace = dict(nested_trace) if isinstance(nested_trace, Mapping) else {}
    trace.update(
        {
            "original_query": query,
            "final_query": final_query,
            "success": success,
            "final_decision": final_decision,
            "knowledge_used": knowledge_used,
            "citations": citations,
            "raw_candidates_count": trace.get("raw_candidates_count", len(records)),
            "filtered_candidates_count": trace.get("filtered_candidates_count", len(records)),
        }
    )
    return trace


def _native_round_trace(result: RetrievalResult) -> dict[str, Any]:
    return {
        "round_index": 1,
        "tool_name": result.tool_name,
        "query": result.query,
        "rewritten_query": None,
        "decision": "finish" if result.success else "retrieval_failed",
        "reason": result.error or "native retrieval completed",
        "is_sufficient": bool(result.success and (result.records or result.documents)),
        "result_success": result.success,
        "result_count": len(result.records),
        "citations": _citations_to_dicts(result.citations) if result.success else [],
    }


def _round_to_trace(round_trace: Any) -> dict[str, Any]:
    result = round_trace.result
    decision = round_trace.decision
    rewrite = getattr(round_trace, "rewrite", None)
    document_trace = result.metadata.get("document_retrieval_trace")
    document_trace = document_trace if isinstance(document_trace, Mapping) else {}
    return {
        "round_index": round_trace.plan.round_index,
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
    }


def _outcome_citations(outcome: AgenticRetrievalOutcome) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in _outcome_results(outcome):
        for citation in _citations_to_dicts(result.citations):
            citation_id = str(citation.get("citation_id") or citation)
            if citation_id in seen:
                continue
            seen.add(citation_id)
            citations.append(citation)
    if citations:
        return citations
    return _document_citations(_outcome_documents(outcome))


def _agentic_error(outcome: AgenticRetrievalOutcome) -> str | None:
    for result in _outcome_results(outcome):
        error = getattr(result, "error", None)
        if error:
            return str(error)
    return None


def _outcome_results(outcome: AgenticRetrievalOutcome) -> list[RetrievalResult]:
    return list(getattr(outcome, "results", []) or [])


def _outcome_rounds(outcome: AgenticRetrievalOutcome) -> list[Any]:
    return list(getattr(outcome, "rounds", []) or [])


def _outcome_documents(outcome: AgenticRetrievalOutcome) -> list[Document]:
    return list(getattr(outcome, "documents", []) or [])


def _outcome_success(
    outcome: AgenticRetrievalOutcome,
    *,
    documents: list[Document],
) -> bool:
    value = getattr(outcome, "success", None)
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(documents)
    return bool(value)


def _outcome_user_query(outcome: AgenticRetrievalOutcome) -> str:
    plan = getattr(outcome, "plan", None)
    query = getattr(plan, "user_query", None)
    if isinstance(query, str) and query:
        return query
    return str(getattr(outcome, "query", "") or "")


def _outcome_final_query(outcome: AgenticRetrievalOutcome) -> str:
    final_plan = getattr(outcome, "final_plan", None)
    active_query = getattr(final_plan, "active_query", None)
    if isinstance(active_query, str) and active_query:
        return active_query
    plan = getattr(outcome, "plan", None)
    active_query = getattr(plan, "active_query", None)
    if isinstance(active_query, str) and active_query:
        return active_query
    return _outcome_user_query(outcome)


def _outcome_candidate_tools(outcome: AgenticRetrievalOutcome) -> tuple[str, ...]:
    final_plan = getattr(outcome, "final_plan", None)
    candidate_tools = getattr(final_plan, "candidate_tools", None)
    if isinstance(candidate_tools, tuple | list):
        return tuple(str(tool_name) for tool_name in candidate_tools)
    return ()


def _final_decision_reason(outcome: AgenticRetrievalOutcome) -> str | None:
    final_decision = getattr(outcome, "final_decision", None)
    reason = getattr(final_decision, "reason", None)
    return str(reason) if reason else None


def _result_citations(result: RetrievalResult) -> list[dict[str, Any]]:
    citations = _citations_to_dicts(result.citations)
    if citations:
        return citations
    return _document_citations(result.documents)


def _document_citations(documents: list[Document]) -> list[dict[str, Any]]:
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


def _citations_to_dicts(citations: list[Any]) -> list[dict[str, Any]]:
    return [_citation_to_dict(citation) for citation in citations]


def _citation_to_dict(citation: Any) -> dict[str, Any]:
    if isinstance(citation, RetrievalCitation):
        return citation.model_dump()
    if hasattr(citation, "model_dump"):
        return citation.model_dump()
    payload = dict(citation)
    payload.setdefault("metadata", {})
    return payload


def _documents_to_payload(documents: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "page_content": document.page_content,
            "metadata": dict(document.metadata),
        }
        for document in documents
    ]


def _document_citation_id(document: Document) -> str:
    return str(
        document.metadata.get("citation_id")
        or document.metadata.get("chunk_id")
        or document.metadata.get("document_id")
        or document.metadata.get("source")
        or document.id
        or "unknown"
    )


def _retrieval_result_from_documents(
    *,
    tool_name: str,
    query: str,
    documents: list[Document],
) -> RetrievalResult:
    return RetrievalResult.ok(
        tool_name=tool_name,
        query=query,
        records=[
            {
                "citation_id": _document_citation_id(document),
                "content": document.page_content,
                "metadata": dict(document.metadata),
            }
            for document in documents
        ],
        documents=documents,
        metadata={"result_count": len(documents)},
    )


def _resolve_native_retrieval_tool(
    retriever: Any,
    *,
    payload: Mapping[str, Any],
    default_candidate_tools: tuple[str, ...],
) -> Any | None:
    tools = getattr(retriever, "tools", None)
    if not isinstance(tools, dict) or not tools:
        return None
    candidate_tools = _resolve_candidate_tools(payload, default_candidate_tools)
    selected_tool = (
        payload.get("selected_tool")
        or getattr(retriever, "default_tool", None)
        or (candidate_tools[0] if candidate_tools else None)
    )
    if selected_tool is None:
        return None
    selected_tool_name = str(selected_tool)
    if candidate_tools and selected_tool_name not in candidate_tools:
        raise ValueError(f"Selected retrieval tool is not allowed: {selected_tool_name}.")
    tool = tools.get(selected_tool_name)
    if tool is None:
        raise ValueError(f"Retrieval tool is not registered: {selected_tool_name}.")
    return tool


def _resolve_candidate_tools(
    payload: Mapping[str, Any],
    default_candidate_tools: tuple[str, ...],
) -> tuple[str, ...]:
    candidate_tools = payload.get("candidate_tools")
    if isinstance(candidate_tools, list | tuple) and candidate_tools:
        return tuple(str(tool_name) for tool_name in candidate_tools)
    return default_candidate_tools


def _strip_none_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _retrieval_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"selected_tool", "candidate_tools", "filters"}
    return {
        key: value
        for key, value in _strip_none_values(payload).items()
        if key not in ignored
    }


def _supported_kwargs(callable_obj: Any, **kwargs: Any) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_var_kwargs:
        return {key: value for key, value in kwargs.items() if value is not None}
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters and value is not None
    }


def _build_rag_summary(
    *,
    adapter_name: str,
    success: bool,
    record_count: int,
    final_decision: str,
    error: str | None,
) -> str:
    if success:
        return f"{adapter_name} returned {record_count} evidence record(s)."
    if error:
        return f"{adapter_name} failed: {error}"
    return f"{adapter_name} finished with decision: {final_decision}."

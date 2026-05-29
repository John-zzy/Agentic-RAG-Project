from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from backend.platform.knowledge.base.text import truncate_snippet
from backend.platform.rag.contracts import RecallStrategy, RetrievalCitation, RetrievalResult
from backend.platform.rag.retrieval.documents import DocumentChunkRetrievalResult
from backend.platform.tools import SceneTool, ToolResult


GENERIC_DOCUMENT_TOOL_NAME = "knowledge_document_search"
GENERIC_DOCUMENT_KNOWLEDGE_SOURCE = "documents"
DOCUMENT_ANSWER_CONTEXT_NEIGHBOR_RADIUS = 1


class KnowledgeDocumentSearchInput(BaseModel):
    """通用知识文档检索工具输入。"""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class KnowledgeDocumentSearchTool(SceneTool):
    """通用知识文档检索工具，统一服务 StructuredTool 与 Agentic Retrieval。"""

    name = GENERIC_DOCUMENT_TOOL_NAME
    description = "Search semantically relevant uploaded knowledge documents."
    capability_type = "retrieval"
    args_schema = KnowledgeDocumentSearchInput

    def __init__(
        self,
        *,
        document_retrieval_service: Any,
        default_top_k: int = 5,
        default_min_relevance_score: float | None = None,
        default_recall_strategy: RecallStrategy = "hybrid",
    ) -> None:
        self._document_retrieval_service = document_retrieval_service
        self._default_top_k = default_top_k
        self._default_min_relevance_score = default_min_relevance_score
        self._default_recall_strategy = default_recall_strategy

    def invoke(self, query: str, top_k: int = 5) -> ToolResult:
        """StructuredTool 入口复用同一个检索实现，避免重复业务逻辑。"""
        retrieval_result = self.retrieve(
            query=query,
            top_k=top_k,
            min_relevance_score=self._default_min_relevance_score,
            recall_strategy=self._default_recall_strategy,
        )
        return _to_tool_result(retrieval_result)

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
        """在上传文档分块中检索并返回标准化结果。"""
        del run_manager, rerank_enabled, rerank_top_n
        resolved_top_k = top_k or self._default_top_k
        resolved_min_relevance = (
            min_relevance_score
            if min_relevance_score is not None
            else self._default_min_relevance_score
        )
        resolved_recall_strategy = recall_strategy or self._default_recall_strategy
        retrieval_trace: dict[str, Any] | None = None

        # 优先读取带 trace 的检索结果，保证调试页和 eval 能复用底层召回证据。
        if hasattr(self._document_retrieval_service, "retrieve_with_trace"):
            traced_result = self._document_retrieval_service.retrieve_with_trace(
                query=query,
                top_k=resolved_top_k,
                minimum_relevance=resolved_min_relevance,
                recall_strategy=resolved_recall_strategy,
            )
            retrieval_results = traced_result.results
            retrieval_trace = traced_result.trace.model_dump()
        else:
            retrieval_results = self._document_retrieval_service.retrieve(
                query=query,
                top_k=resolved_top_k,
                minimum_relevance=resolved_min_relevance,
                recall_strategy=resolved_recall_strategy,
            )

        records = _build_document_records(
            retrieval_results,
            document_retrieval_service=self._document_retrieval_service,
        )
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
                page_content=record["content"],
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
        return RetrievalResult.ok(
            tool_name=self.name,
            query=query,
            records=records,
            documents=documents,
            citations=citations,
            confidence=_average_score(records),
            metadata={
                "namespace": "documents",
                "result_count": len(records),
                "scene": "generic_assistant",
                "document_retrieval_trace": retrieval_trace,
            },
        )


def _to_tool_result(retrieval_result: RetrievalResult) -> ToolResult:
    """将 Agentic Retrieval 结果映射为通用 ToolResult。"""
    if not retrieval_result.success:
        return ToolResult.fail(
            tool_name=retrieval_result.tool_name,
            error=retrieval_result.error or "Unknown retrieval error.",
            metadata=retrieval_result.metadata,
        )
    return ToolResult.ok(
        tool_name=retrieval_result.tool_name,
        records=retrieval_result.records,
        citations=[
            {
                "citation_id": citation.citation_id,
                "namespace": citation.source_type,
                "snippet": citation.snippet,
                "metadata": citation.metadata,
            }
            for citation in retrieval_result.citations
        ],
        confidence=retrieval_result.confidence,
        metadata=retrieval_result.metadata,
    )


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
        "content": result.document.content,
        "score": float(result.score) if result.score is not None else None,
        "vector_score": float(result.vector_score) if result.vector_score is not None else None,
        "keyword_score": float(result.keyword_score) if result.keyword_score is not None else None,
        "vector_rank": result.vector_rank,
        "keyword_rank": result.keyword_rank,
        "matched_by": list(result.matched_by),
        "metadata": result.document.metadata,
    }


def _build_document_records(
    results: list[DocumentChunkRetrievalResult],
    *,
    document_retrieval_service: Any,
) -> list[dict[str, Any]]:
    """保留 citation 摘要，同时为回答上下文补齐相邻分块。"""
    records = [_build_document_record(result) for result in results]
    neighbor_context = _build_neighbor_context_by_chunk_id(
        results,
        document_retrieval_service=document_retrieval_service,
    )
    if not neighbor_context:
        return records

    for record in records:
        citation_id = str(record.get("citation_id") or "")
        expanded_content = neighbor_context.get(citation_id)
        if expanded_content:
            record["content"] = expanded_content
    return records


def _build_neighbor_context_by_chunk_id(
    results: list[DocumentChunkRetrievalResult],
    *,
    document_retrieval_service: Any,
) -> dict[str, str]:
    chunk_source = getattr(document_retrieval_service, "chunk_source", None)
    list_chunks = getattr(chunk_source, "list_active_document_chunks", None)
    if not callable(list_chunks):
        return {}

    try:
        chunks = list_chunks(limit=None)
    except TypeError:
        chunks = list_chunks()
    if not chunks:
        return {}

    chunks_by_position: dict[tuple[str, int], Any] = {}
    for chunk in chunks:
        document_id = _resolve_document_id(chunk)
        chunk_index = _resolve_chunk_index(chunk)
        if document_id is None or chunk_index is None:
            continue
        chunks_by_position[(document_id, chunk_index)] = chunk

    context_by_chunk_id: dict[str, str] = {}
    for result in results:
        document_id = _resolve_document_id(result.document)
        chunk_index = _resolve_chunk_index(result.document)
        citation_id = _resolve_document_citation_id(result.document)
        if document_id is None or chunk_index is None:
            continue
        window: list[Any] = []
        for neighbor_index in range(
            chunk_index - DOCUMENT_ANSWER_CONTEXT_NEIGHBOR_RADIUS,
            chunk_index + DOCUMENT_ANSWER_CONTEXT_NEIGHBOR_RADIUS + 1,
        ):
            neighbor = chunks_by_position.get((document_id, neighbor_index))
            if neighbor is not None:
                window.append(neighbor)
        if len(window) <= 1:
            continue
        context_by_chunk_id[citation_id] = "\n\n".join(str(chunk.content) for chunk in window)
    return context_by_chunk_id


def _resolve_document_namespace(document: Any) -> str:
    """优先保留文档知识源自己的 namespace。"""
    namespace = document.metadata.get("namespace")
    if isinstance(namespace, str) and namespace:
        return namespace
    return "documents"


def _resolve_document_id(document: Any) -> str | None:
    document_id = document.metadata.get("document_id")
    if isinstance(document_id, str) and document_id:
        return document_id
    return None


def _resolve_chunk_index(document: Any) -> int | None:
    chunk_index = document.metadata.get("chunk_index")
    if isinstance(chunk_index, bool):
        return None
    if isinstance(chunk_index, int):
        return chunk_index
    if isinstance(chunk_index, float) and chunk_index.is_integer():
        return int(chunk_index)
    if isinstance(chunk_index, str):
        try:
            return int(chunk_index)
        except ValueError:
            return None
    return None


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

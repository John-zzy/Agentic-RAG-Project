from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from langchain_core.documents import Document

from backend.platform.models.llm.rerank import get_rerank_wrapper
from backend.platform.rag.contracts import RetrievalCitation, RetrievalResult


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankTrace:
    """描述一次 rerank 边界执行结果，供 trace 和 eval 读取。"""

    enabled: bool
    provider: str | None
    model: str | None
    applied: bool
    input_count: int
    output_count: int
    top_n: int | None = None
    fallback_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "applied": self.applied,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "top_n": self.top_n,
            "fallback_reason": self.fallback_reason,
            "error": self.error,
        }


class RetrievalReranker(Protocol):
    """平台级 rerank 协议，编排层只依赖该协议完成重排。"""

    provider: str

    def rerank(
        self,
        *,
        query: str,
        result: RetrievalResult,
        top_n: int | None = None,
    ) -> tuple[RetrievalResult, RerankTrace]:
        """返回 rerank 后的结果及安全 trace。"""
        ...


@dataclass(frozen=True)
class _RankedCandidate:
    """保存模型返回的候选索引和分数，供三类证据同步排序。"""

    index: int
    rerank_score: float


class DashScopeRetrievalReranker:
    """通过模型层创建的 LangChain DashScopeRerank 执行真实重排。"""

    provider = "dashscope"

    def __init__(self, *, wrapper_factory: Callable[[], Any] = get_rerank_wrapper) -> None:
        self._wrapper_factory = wrapper_factory

    def rerank(
        self,
        *,
        query: str,
        result: RetrievalResult,
        top_n: int | None = None,
    ) -> tuple[RetrievalResult, RerankTrace]:
        input_count = len(result.documents)
        if input_count == 0:
            return self._fallback_result(
                result=result,
                input_count=input_count,
                top_n=top_n,
                reason="empty_candidates",
            )

        model_name: str | None = None
        try:
            # 步骤 1：只通过模型层工厂获得 wrapper，编排层不读取模型配置或环境变量。
            wrapper = self._wrapper_factory()
            model_name = self._resolve_model_name(wrapper)
            ranked_candidates = self._call_rerank_model(
                wrapper=wrapper,
                query=query,
                documents=result.documents,
                top_n=top_n,
            )
        except Exception as exc:
            return self._fallback_result(
                result=result,
                input_count=input_count,
                top_n=top_n,
                reason=type(exc).__name__,
                model=model_name,
                exc=exc,
            )

        if not ranked_candidates:
            return self._fallback_result(
                result=result,
                input_count=input_count,
                output_count=0,
                top_n=top_n,
                reason="empty_rerank_result",
                model=model_name,
            )

        trace = RerankTrace(
            enabled=True,
            provider=self.provider,
            model=model_name,
            applied=True,
            input_count=input_count,
            output_count=len(ranked_candidates),
            top_n=top_n,
        )
        reranked = result.model_copy(
            update={
                "records": self._rerank_records(result, ranked_candidates),
                "documents": self._rerank_documents(result, ranked_candidates),
                "citations": self._rerank_citations(result, ranked_candidates),
                "metadata": {
                    **result.metadata,
                    "rerank": trace.to_dict(),
                },
            }
        )
        return reranked, trace

    def _call_rerank_model(
        self,
        *,
        wrapper: Any,
        query: str,
        documents: Sequence[Document],
        top_n: int | None,
    ) -> list[_RankedCandidate]:
        if top_n is None:
            raw_results = wrapper.rerank(documents, query)
        else:
            raw_results = wrapper.rerank(documents, query, top_n=top_n)

        # 步骤 2：严格解析 LangChain DashScopeRerank 返回的 index 和 relevance_score。
        ranked_candidates: list[_RankedCandidate] = []
        for raw_result in raw_results:
            index = int(raw_result["index"])
            if index < 0 or index >= len(documents):
                raise ValueError(f"Rerank result index out of range: {index}")
            ranked_candidates.append(
                _RankedCandidate(
                    index=index,
                    rerank_score=float(raw_result["relevance_score"]),
                )
            )
        if top_n is not None:
            # 步骤 3：adapter 自身执行最终截断，避免外部 wrapper 行为变化影响证据数量。
            return ranked_candidates[:top_n]
        return ranked_candidates

    def _resolve_model_name(self, wrapper: Any) -> str | None:
        for attribute_name in ("model", "model_name"):
            value = getattr(wrapper, attribute_name, None)
            if isinstance(value, str) and value:
                return value
        return None

    def _rerank_records(
        self,
        result: RetrievalResult,
        ranked_candidates: list[_RankedCandidate],
    ) -> list[dict[str, Any]]:
        # 步骤 4：records、documents、citations 使用同一组模型索引同步重排。
        return [
            {
                **result.records[candidate.index],
                "rerank_score": candidate.rerank_score,
                "metadata": self._metadata_with_rerank_score(
                    result.records[candidate.index].get("metadata"),
                    candidate.rerank_score,
                ),
            }
            for candidate in ranked_candidates
            if candidate.index < len(result.records)
        ]

    def _rerank_documents(
        self,
        result: RetrievalResult,
        ranked_candidates: list[_RankedCandidate],
    ) -> list[Document]:
        return [
            result.documents[candidate.index].model_copy(
                update={
                    "metadata": {
                        **result.documents[candidate.index].metadata,
                        "rerank_score": candidate.rerank_score,
                    }
                }
            )
            for candidate in ranked_candidates
        ]

    def _rerank_citations(
        self,
        result: RetrievalResult,
        ranked_candidates: list[_RankedCandidate],
    ) -> list[RetrievalCitation]:
        return [
            result.citations[candidate.index].model_copy(
                update={
                    "metadata": {
                        **result.citations[candidate.index].metadata,
                        "rerank_score": candidate.rerank_score,
                    }
                }
            )
            for candidate in ranked_candidates
            if candidate.index < len(result.citations)
        ]

    def _metadata_with_rerank_score(self, metadata: Any, rerank_score: float) -> dict[str, Any]:
        if isinstance(metadata, dict):
            return {**metadata, "rerank_score": rerank_score}
        return {"rerank_score": rerank_score}

    def _fallback_result(
        self,
        *,
        result: RetrievalResult,
        input_count: int,
        top_n: int | None,
        reason: str,
        output_count: int | None = None,
        model: str | None = None,
        exc: Exception | None = None,
    ) -> tuple[RetrievalResult, RerankTrace]:
        if exc is None:
            logger.info("Rerank skipped or fell back: reason=%s, input_count=%s", reason, input_count)
        else:
            logger.warning(
                "Rerank model call failed; preserving original retrieval order: reason=%s",
                reason,
                exc_info=True,
            )
        trace = RerankTrace(
            enabled=True,
            provider=self.provider,
            model=model,
            applied=False,
            input_count=input_count,
            output_count=input_count if output_count is None else output_count,
            top_n=top_n,
            fallback_reason=reason,
            error=self._safe_error_summary(exc),
        )
        sanitized = remove_rerank_scores(result)
        return sanitized.model_copy(
            update={
                "metadata": {
                    **sanitized.metadata,
                    "rerank": trace.to_dict(),
                }
            }
        ), trace

    def _safe_error_summary(self, exc: Exception | None) -> str | None:
        if exc is None:
            return None
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return message[:240]


class IdentityRetrievalReranker:
    """保持既有排序的 identity reranker，仅负责统一截断和 trace。"""

    provider = "identity"

    def rerank(
        self,
        *,
        query: str,
        result: RetrievalResult,
        top_n: int | None = None,
    ) -> tuple[RetrievalResult, RerankTrace]:
        del query
        input_count = max(len(result.records), len(result.documents), len(result.citations))
        sanitized = remove_rerank_scores(result)
        retained_ids = self._retained_citation_ids(result=result, top_n=top_n)
        trace = RerankTrace(
            enabled=True,
            provider=self.provider,
            model=None,
            applied=False,
            input_count=input_count,
            output_count=len(retained_ids),
            top_n=top_n,
        )
        reranked = sanitized.model_copy(
            update={
                "records": self._truncate_records(sanitized.records, retained_ids=retained_ids),
                "documents": self._truncate_documents(sanitized.documents, retained_ids=retained_ids),
                "citations": self._truncate_citations(sanitized.citations, retained_ids=retained_ids),
                "metadata": {
                    **sanitized.metadata,
                    "rerank": trace.to_dict(),
                },
            }
        )
        return reranked, trace

    def _retained_citation_ids(
        self,
        *,
        result: RetrievalResult,
        top_n: int | None,
    ) -> list[str]:
        # 以 citation_id 作为保留轴，确保 records/documents/citations 截断后仍能一一对齐。
        citation_ids: list[str] = []
        for record in result.records:
            citation_id = record.get("citation_id")
            if citation_id is not None:
                citation_ids.append(str(citation_id))
        if not citation_ids:
            for citation in result.citations:
                citation_ids.append(citation.citation_id)
        if not citation_ids:
            for document in result.documents:
                citation_ids.append(self._document_citation_id(document))
        if top_n is not None:
            return citation_ids[:top_n]
        return citation_ids

    def _truncate_records(
        self,
        records: list[dict[str, object]],
        *,
        retained_ids: list[str],
    ) -> list[dict[str, object]]:
        if not retained_ids:
            return []
        retained = set(retained_ids)
        return [record for record in records if str(record.get("citation_id")) in retained]

    def _truncate_documents(
        self,
        documents: list[Document],
        *,
        retained_ids: list[str],
    ) -> list[Document]:
        if not retained_ids:
            return []
        retained = set(retained_ids)
        return [document for document in documents if self._document_citation_id(document) in retained]

    def _truncate_citations(
        self,
        citations: list[RetrievalCitation],
        *,
        retained_ids: list[str],
    ) -> list[RetrievalCitation]:
        if not retained_ids:
            return []
        retained = set(retained_ids)
        return [citation for citation in citations if citation.citation_id in retained]

    def _document_citation_id(self, document: Document) -> str:
        return str(
            document.metadata.get("citation_id")
            or document.metadata.get("chunk_id")
            or document.metadata.get("document_id")
            or document.id
            or document.page_content
        )


def disabled_rerank_trace(result: RetrievalResult) -> RerankTrace:
    count = max(len(result.records), len(result.documents), len(result.citations))
    return RerankTrace(
        enabled=False,
        provider=None,
        model=None,
        applied=False,
        input_count=count,
        output_count=count,
        top_n=None,
    )


def remove_rerank_scores(result: RetrievalResult) -> RetrievalResult:
    """移除未应用 rerank 时可能残留的保留字段，避免 API 暴露伪造分数。"""
    return result.model_copy(
        update={
            "records": [_record_without_rerank_score(record) for record in result.records],
            "documents": [
                document.model_copy(
                    update={"metadata": _metadata_without_rerank_score(document.metadata)}
                )
                for document in result.documents
            ],
            "citations": [
                citation.model_copy(
                    update={"metadata": _metadata_without_rerank_score(citation.metadata)}
                )
                for citation in result.citations
            ],
        }
    )


def _record_without_rerank_score(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {key: value for key, value in record.items() if key != "rerank_score"}
    metadata = cleaned.get("metadata")
    if isinstance(metadata, dict):
        cleaned["metadata"] = _metadata_without_rerank_score(metadata)
    return cleaned


def _metadata_without_rerank_score(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "rerank_score"}

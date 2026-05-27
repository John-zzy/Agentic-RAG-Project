from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document

from backend.platform.rag.core import RetrievalCitation, RetrievalResult


@dataclass(frozen=True)
class RerankTrace:
    """描述一次 rerank 边界执行结果，供 trace 和 eval 读取。"""

    enabled: bool
    provider: str | None
    applied: bool
    input_count: int
    output_count: int
    top_n: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "applied": self.applied,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "top_n": self.top_n,
        }


class RetrievalReranker(Protocol):
    """平台级 rerank 协议，后续可替换为真实 rerank provider。"""

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
        retained_ids = self._retained_citation_ids(result=result, top_n=top_n)
        reranked = result.model_copy(
            update={
                "records": self._truncate_records(result.records, retained_ids=retained_ids),
                "documents": self._truncate_documents(result.documents, retained_ids=retained_ids),
                "citations": self._truncate_citations(result.citations, retained_ids=retained_ids),
                "metadata": {
                    **result.metadata,
                    "rerank": RerankTrace(
                        enabled=True,
                        provider=self.provider,
                        applied=False,
                        input_count=input_count,
                        output_count=len(retained_ids),
                        top_n=top_n,
                    ).to_dict(),
                },
            }
        )
        return reranked, RerankTrace(
            enabled=True,
            provider=self.provider,
            applied=False,
            input_count=input_count,
            output_count=len(retained_ids),
            top_n=top_n,
        )

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
        applied=False,
        input_count=count,
        output_count=count,
        top_n=None,
    )

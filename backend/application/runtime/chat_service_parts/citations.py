from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from langchain_core.documents import Document

from backend.application.runtime.api.chat.schemas import Citation
from backend.platform.knowledge.base.text import truncate_snippet
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



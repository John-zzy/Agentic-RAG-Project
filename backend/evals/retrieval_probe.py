from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_core.documents import Document

from backend.platform.config.settings import AppSettings, settings
from backend.platform.knowledge.repositories import VectorStoreFactory
from backend.platform.rag.orchestration.agentic import AgenticRetriever
from backend.platform.rag.retrieval.documents import DocumentRetrievalService
from backend.scenes.base import SceneDefinition
from backend.scenes.generic_assistant.definition import build_generic_assistant_scene_definition


SAFE_RANKED_ITEM_FIELDS = (
    "rank",
    "source_doc",
    "document_id",
    "chunk_id",
    "chunk_index",
    "score",
    "matched_by",
)


@dataclass(frozen=True)
class RetrievalProbe:
    """Eval-only retrieval probe that serializes ranked results by whitelist."""

    scene_definition: SceneDefinition
    retriever: AgenticRetriever

    def run_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        query = str(sample.get("query") or "").strip()
        if not query:
            raise ValueError("retrieval probe sample query must not be empty.")

        policy = self.scene_definition.retrieval_policy
        candidate_tools = self.scene_definition.resolve_candidate_retrieval_tools(("documents",))
        outcome = self.retriever.retrieve_with_trace(
            query=query,
            candidate_tools=candidate_tools,
            top_k=policy.top_k,
            min_relevance_score=policy.min_relevance_score,
            recall_strategy=policy.recall_strategy,
            rerank_enabled=policy.rerank_enabled,
            rerank_top_n=policy.rerank_top_n,
        )
        ranked_list = [
            _safe_ranked_item(rank=rank, document=document)
            for rank, document in enumerate(outcome.documents, start=1)
        ]
        return {
            "sample_id": str(sample.get("sample_id")),
            "status": "ok",
            "ranked_list": ranked_list,
            "failure_reasons": [],
        }


def build_retrieval_probe(
    *,
    app_settings: AppSettings | None = None,
    document_retrieval_service: DocumentRetrievalService | None = None,
    scene_definition: SceneDefinition | None = None,
) -> RetrievalProbe:
    """Build a generic_assistant + documents retrieval probe using runtime components."""
    resolved_settings = app_settings or settings
    resolved_service = document_retrieval_service or DocumentRetrievalService(
        app_settings=resolved_settings,
        vector_repository=VectorStoreFactory.create_document_chunk_vector_repository(resolved_settings),
        chunk_source=VectorStoreFactory.create_active_document_chunk_source(resolved_settings),
    )
    resolved_scene = scene_definition or build_generic_assistant_scene_definition(
        app_settings=resolved_settings,
        document_retrieval_service=resolved_service,
    )
    retriever = resolved_scene.build_retriever()
    if not isinstance(retriever, AgenticRetriever):
        raise TypeError("generic_assistant retrieval probe requires an AgenticRetriever.")
    retriever.attach_trace = False
    return RetrievalProbe(scene_definition=resolved_scene, retriever=retriever)


def run_retrieval_probe(
    *,
    samples: Sequence[Mapping[str, Any]],
    allowed_source_docs: Sequence[str] | None = None,
    app_settings: AppSettings | None = None,
    document_retrieval_service: DocumentRetrievalService | None = None,
    scene_definition: SceneDefinition | None = None,
) -> dict[str, Any]:
    """Run the safe retrieval probe for all samples and keep per-sample failures local."""
    probe = build_retrieval_probe(
        app_settings=app_settings,
        document_retrieval_service=document_retrieval_service,
        scene_definition=scene_definition,
    )
    results: list[dict[str, Any]] = []
    allowed_source_doc_set = (
        {Path(source_doc).name for source_doc in allowed_source_docs}
        if allowed_source_docs is not None
        else None
    )
    for sample in samples:
        try:
            result = probe.run_sample(sample)
            if allowed_source_doc_set is not None:
                result["ranked_list"] = [
                    item
                    for item in result["ranked_list"]
                    if item.get("source_doc") in allowed_source_doc_set
                ]
                for rank, item in enumerate(result["ranked_list"], start=1):
                    item["rank"] = rank
            results.append(result)
        except Exception as exc:  # pragma: no cover - defensive path exercised by runner
            results.append(
                {
                    "sample_id": str(sample.get("sample_id")),
                    "status": "error",
                    "ranked_list": [],
                    "failure_reasons": [str(exc)],
                }
            )
    return {
        "scene": probe.scene_definition.scene,
        "namespace": "documents",
        "allowed_source_docs": sorted(allowed_source_doc_set) if allowed_source_doc_set is not None else None,
        "ranked_item_fields": list(SAFE_RANKED_ITEM_FIELDS),
        "samples": results,
    }


def _safe_ranked_item(*, rank: int, document: Document) -> dict[str, Any]:
    metadata = dict(document.metadata)
    safe = {
        "rank": rank,
        "source_doc": _source_doc_name(metadata),
        "document_id": _optional_str(metadata.get("document_id")),
        "chunk_id": _optional_str(metadata.get("chunk_id") or metadata.get("citation_id")),
        "chunk_index": _optional_int(metadata.get("chunk_index")),
        "score": _optional_float(metadata.get("score")),
        "matched_by": [
            str(item)
            for item in metadata.get("matched_by", [])
            if isinstance(item, str)
        ],
    }
    return {field: safe.get(field) for field in SAFE_RANKED_ITEM_FIELDS}


def _source_doc_name(metadata: Mapping[str, Any]) -> str | None:
    source = metadata.get("source_path") or metadata.get("source") or metadata.get("title")
    if source is None:
        return None
    return Path(str(source)).name


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

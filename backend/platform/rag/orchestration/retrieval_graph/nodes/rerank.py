from __future__ import annotations

from typing import Any

from backend.platform.rag.orchestration.retrieval_graph.config import AgenticRagGraphDependencies
from backend.platform.rag.orchestration.retrieval_graph.state import (
    AgenticRagGraphState,
    _documents_to_payload,
)


def build_rerank_node(dependencies: AgenticRagGraphDependencies):
    """保留现有 rerank 边界和降级行为。"""

    retriever = dependencies.retriever

    def rerank(state: AgenticRagGraphState) -> dict[str, Any]:
        plan = state["plan"]
        result = state["current_result"]
        if result is None:
            raise ValueError("Agentic RAG rerank node requires current retrieval result.")

        reranked = retriever._apply_rerank(plan=plan, result=result)
        results = list(state["results"])
        results[-1] = reranked
        documents = retriever._merge_documents(list(state["documents"]), reranked.documents)
        return {
            "current_result": reranked,
            "results": results,
            "documents": documents,
            "candidate_docs": _documents_to_payload(reranked.documents),
        }

    return rerank


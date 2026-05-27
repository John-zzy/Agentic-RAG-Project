from backend.platform.rag.contracts import RecallStrategy
from backend.platform.rag.retrieval.documents.embedding import DocumentEmbeddingStrategy
from backend.platform.rag.retrieval.documents.filters import (
    DOCUMENT_MINIMUM_RELEVANCE,
    filter_low_relevance_document_results,
    filter_managed_document_results,
)
from backend.platform.rag.retrieval.documents.fusion import HybridFusionRanker
from backend.platform.rag.retrieval.documents.keyword import DocumentKeywordRetriever
from backend.platform.rag.retrieval.documents.keyword_scoring import DocumentKeywordScoreCalculator
from backend.platform.rag.retrieval.documents.semantic import DocumentSemanticRetriever
from backend.platform.rag.retrieval.documents.service import (
    DocumentHybridRetriever,
    DocumentRetrievalService,
)
from backend.platform.rag.retrieval.documents.types import (
    DocumentChunkRetrievalResult,
    merge_matched_by,
)

__all__ = [
    "DOCUMENT_MINIMUM_RELEVANCE",
    "DocumentChunkRetrievalResult",
    "DocumentEmbeddingStrategy",
    "DocumentHybridRetriever",
    "DocumentKeywordRetriever",
    "DocumentKeywordScoreCalculator",
    "DocumentRetrievalService",
    "DocumentSemanticRetriever",
    "HybridFusionRanker",
    "RecallStrategy",
    "filter_low_relevance_document_results",
    "filter_managed_document_results",
    "merge_matched_by",
]

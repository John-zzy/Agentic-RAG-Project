"""Knowledge base package."""

from backend.knowledge.loader import KnowledgeLoadSummary, preload_knowledge_base
from backend.knowledge.retriever import KnowledgeBaseRetriever
from backend.knowledge.service import KnowledgeService, KnowledgeUpsertSummary, create_knowledge_service
from backend.knowledge.store import (
    ChromaVectorStore,
    ElasticsearchVectorStore,
    VectorSearchResult,
    VectorStore,
    VectorStoreDocument,
    VectorStoreFactory,
    VectorStoreHealth,
)

__all__ = [
    "ChromaVectorStore",
    "ElasticsearchVectorStore",
    "KnowledgeLoadSummary",
    "KnowledgeBaseRetriever",
    "KnowledgeService",
    "KnowledgeUpsertSummary",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreDocument",
    "VectorStoreFactory",
    "VectorStoreHealth",
    "preload_knowledge_base",
    "create_knowledge_service",
]

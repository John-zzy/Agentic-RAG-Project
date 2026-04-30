"""知识库包。"""

from backend.knowledge.base import (
    MAX_SNIPPET_LENGTH,
    SUPPORTED_NAMESPACES,
    ChromaVectorStore,
    ElasticsearchVectorStore,
    VectorSearchResult,
    VectorStore,
    VectorStoreDocument,
    VectorStoreFactory,
    VectorStoreHealth,
    truncate_snippet,
)
from backend.knowledge.ecommerce import (
    KnowledgeBaseRetriever,
    KnowledgeLoadSummary,
    KnowledgeService,
    KnowledgeUpsertSummary,
    build_product_document,
    build_review_document,
    create_knowledge_service,
    preload_knowledge_base,
)

__all__ = [
    "ChromaVectorStore",
    "ElasticsearchVectorStore",
    "KnowledgeBaseRetriever",
    "KnowledgeLoadSummary",
    "KnowledgeService",
    "KnowledgeUpsertSummary",
    "MAX_SNIPPET_LENGTH",
    "SUPPORTED_NAMESPACES",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreDocument",
    "VectorStoreFactory",
    "VectorStoreHealth",
    "build_product_document",
    "build_review_document",
    "create_knowledge_service",
    "preload_knowledge_base",
    "truncate_snippet",
]

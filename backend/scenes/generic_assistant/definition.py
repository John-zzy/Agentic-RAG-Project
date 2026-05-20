from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from backend.platform.config.settings import AppSettings, settings
from backend.scenes.base import (
    SceneBootstrapResult,
    SceneDefinition,
    SceneFallbackPolicy,
)
from backend.platform.knowledge.base.text import truncate_snippet
from backend.platform.rag.agentic import AgenticRetriever
from backend.platform.rag.core import RetrievalCitation, RetrievalResult, RetrievalTool
from backend.platform.rag.document_retrieval import DocumentRetrievalService
from backend.platform.tools import ToolResult, build_structured_tool
from backend.scenes.ecommerce.definition import EcommerceQueryRewriter, EcommerceSufficiencyJudge
from backend.scenes.ecommerce.retrieval_tools import (
    ProductCatalogStore,
    build_agentic_retrieval_tools,
)
from backend.scenes.ecommerce.knowledge_service import create_knowledge_service


GENERIC_ASSISTANT_SYSTEM_PROMPT = (
    "你是一名通用知识助手。"
    "请优先依据检索到的文档上下文回答问题，回答要清晰、克制。"
    "当证据不足时，明确说明不确定，并提示用户补充更具体的文档主题、术语或背景。"
)


class GenericKnowledgeDocumentSearchInput(BaseModel):
    """通用知识文档检索工具输入。"""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class GenericKnowledgeDocumentRetriever(BaseRetriever):
    """通用助手默认 retriever，只依赖文档知识库。"""

    document_retrieval_service: DocumentRetrievalService = Field(exclude=True)
    default_top_k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        """适配 LangChain retriever 协议。"""
        return self.search(query=query, top_k=self.default_top_k)

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """仅在已上传文档知识中检索证据。"""
        return self.document_retrieval_service.search(
            query=query,
            top_k=top_k or self.default_top_k,
        )


class GenericKnowledgeDocumentSearchTool(RetrievalTool):
    """通用知识文档检索工具，供 scene runtime 直接挂载。"""

    name: str = "knowledge_document_search"
    description: str = "Search semantically relevant uploaded knowledge documents."
    document_retrieval_service: DocumentRetrievalService = Field(exclude=True)
    default_top_k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def retrieve(self, query: str, *, run_manager: Any | None = None) -> RetrievalResult:
        """在上传文档分块中检索并返回标准化结果。"""
        del run_manager
        retrieval_results = self.document_retrieval_service.retrieve(query=query, top_k=self.default_top_k)
        records = [_build_document_record(result) for result in retrieval_results]
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
                page_content=record["snippet"],
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
        confidence = _average_score(records)
        return RetrievalResult.ok(
            tool_name=self.name,
            query=query,
            records=records,
            documents=documents,
            citations=citations,
            confidence=confidence,
            metadata={"namespace": "documents", "result_count": len(records), "scene": "generic_assistant"},
        )


def build_generic_assistant_scene_definition(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: object | None = None,
    document_retrieval_service: DocumentRetrievalService | None = None,
    product_store: ProductCatalogStore | None = None,
    max_rounds: int = 3,
) -> SceneDefinition:
    """构建通用知识助手场景定义。"""
    current_settings = app_settings or settings
    resolved_document_retrieval_service = document_retrieval_service or DocumentRetrievalService(
        app_settings=current_settings
    )
    resolved_product_store = product_store or ProductCatalogStore(data_dir=current_settings.data_dir)
    return SceneDefinition(
        scene="generic_assistant",
        name="Generic Knowledge Assistant",
        description="以用户上传文档为主，并可按会话挂载扩展到其他知识源的通用 RAG 助手。",
        build_retriever=lambda: _build_generic_agentic_retriever(
            document_retrieval_service=resolved_document_retrieval_service,
            knowledge_service=_resolve_knowledge_service(current_settings, knowledge_service),
            product_store=resolved_product_store,
            max_rounds=max_rounds,
        ),
        build_tools=lambda: (
            build_generic_knowledge_document_tool(
                resolved_document_retrieval_service,
            ),
        ),
        system_prompt=GENERIC_ASSISTANT_SYSTEM_PROMPT,
        fallback_policy=SceneFallbackPolicy(
            no_hit_message="暂时没有检索到足够相关的文档知识。请补充更具体的主题、术语或文档范围，我再继续帮你查。"
        ),
        infer_complexity=infer_generic_assistant_complexity,
        bootstrap=lambda: SceneBootstrapResult(),
        metadata={
            "supports_agentic_retrieval": True,
            "knowledge_sources": ("documents", "ecommerce"),
            "default_agent": None,
            "prompt_style": "generic_knowledge_assistant",
        },
    )


def build_generic_knowledge_document_tool(
    document_retrieval_service: DocumentRetrievalService,
) -> BaseTool:
    """构建面向通用知识助手的文档检索工具。"""

    def knowledge_document_search(query: str, top_k: int = 5) -> ToolResult:
        retrieval_results = document_retrieval_service.retrieve(query=query, top_k=top_k)
        records = [_build_document_record(result) for result in retrieval_results]
        return ToolResult.ok(
            tool_name="knowledge_document_search",
            records=records,
            citations=[
                {
                    "citation_id": record["citation_id"],
                    "namespace": record["namespace"],
                    "snippet": record["snippet"],
                    "metadata": {
                        "score": record.get("score"),
                        "vector_score": record.get("vector_score"),
                        "keyword_score": record.get("keyword_score"),
                        "vector_rank": record.get("vector_rank"),
                        "keyword_rank": record.get("keyword_rank"),
                        "matched_by": record.get("matched_by", []),
                    },
                }
                for record in records
            ],
            confidence=_average_score(records),
            metadata={"namespace": "documents", "result_count": len(records), "scene": "generic_assistant"},
        )

    return build_structured_tool(
        name="knowledge_document_search",
        description="Search semantically relevant uploaded knowledge documents.",
        capability_type="retrieval",
        args_schema=GenericKnowledgeDocumentSearchInput,
        func=knowledge_document_search,
    )


def infer_generic_assistant_complexity(message: str) -> str:
    """按通用问答场景估算模型复杂度，避免平台层耦合业务关键词。"""
    normalized = message.strip().lower()
    complex_keywords = ("总结", "对比", "方案", "流程", "原因", "风险", "设计")
    moderate_keywords = ("解释", "说明", "如何", "为什么", "步骤", "文档", "知识库")

    if any(keyword in normalized for keyword in complex_keywords) or len(normalized) > 120:
        return "complex"
    if any(keyword in normalized for keyword in moderate_keywords) or len(normalized) > 40:
        return "moderate"
    return "simple"


def _build_generic_agentic_retriever(
    *,
    document_retrieval_service: DocumentRetrievalService,
    knowledge_service: Any,
    product_store: ProductCatalogStore,
    max_rounds: int,
) -> AgenticRetriever:
    """为通用场景构建文档优先的 AgenticRetriever。"""
    tools = build_agentic_retrieval_tools(
        document_retrieval_service=document_retrieval_service,
        knowledge_service=knowledge_service,
        product_store=product_store,
    )
    return AgenticRetriever(
        tools={tool.name: tool for tool in tools},
        default_tool="knowledge_document_search",
        sufficiency_judge=EcommerceSufficiencyJudge(),
        query_rewriter=EcommerceQueryRewriter(),
        max_rounds=max_rounds,
    )


def _resolve_knowledge_service(
    current_settings: AppSettings,
    knowledge_service: object | None,
) -> Any:
    if knowledge_service is not None:
        return knowledge_service
    return create_knowledge_service(current_settings)


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
        "score": float(result.score) if result.score is not None else None,
        "vector_score": float(result.vector_score) if result.vector_score is not None else None,
        "keyword_score": float(result.keyword_score) if result.keyword_score is not None else None,
        "vector_rank": result.vector_rank,
        "keyword_rank": result.keyword_rank,
        "matched_by": list(result.matched_by),
        "metadata": result.document.metadata,
    }


def _resolve_document_namespace(document: Any) -> str:
    """优先保留文档知识源自己的 namespace。"""
    namespace = document.metadata.get("namespace")
    if isinstance(namespace, str) and namespace:
        return namespace
    return "documents"


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

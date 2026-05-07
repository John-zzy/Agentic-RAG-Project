from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict, Field

from backend.config.settings import AppSettings, settings
from backend.knowledge.base.store import VectorSearchResult
from backend.knowledge.base.text import truncate_snippet
from backend.knowledge.ecommerce.service import KnowledgeService, create_knowledge_service
from backend.knowledge.rag.agentic import AgenticRetriever
from backend.knowledge.rag.core import (
    QueryRewrite,
    QueryRewriter,
    RetrievalContext,
    SufficiencyDecision,
    SufficiencyJudge,
)
from backend.tools.ecommerce.retrieval import ProductCatalogStore, build_agentic_retrieval_tools


class KnowledgeBaseRetriever(BaseRetriever):
    """将电商知识检索结果适配为 LangChain Retriever。"""

    knowledge_service: Any = Field(exclude=True)
    default_top_k: int = 5
    minimum_relevance: float = 0.18

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        """适配 LangChain 检索器协议，返回相关文档。"""
        return self.search(query=query, top_k=self.default_top_k)

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """聚合商品、评价与订单检索结果，排序去重后返回。"""
        requested_top_k = top_k or self.default_top_k
        product_results = self.knowledge_service.search_products(query=query, top_k=requested_top_k)
        review_results = self.knowledge_service.search_reviews(query=query, top_k=requested_top_k)
        order_results = self.knowledge_service.search_orders(query=query, top_k=requested_top_k)

        combined = (
            self._to_documents("products", product_results)
            + self._to_documents("reviews", review_results)
            + self._to_documents("orders", order_results)
        )
        combined.sort(key=self._doc_score, reverse=True)

        deduped: list[Document] = []
        seen: set[tuple[str, str]] = set()
        for doc in combined:
            namespace = str(doc.metadata.get("namespace", "knowledge"))
            citation_id = str(doc.metadata.get("citation_id", "unknown"))
            key = (namespace, citation_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
            if len(deduped) >= requested_top_k:
                break
        return deduped

    def _to_documents(self, namespace: str, results: list[VectorSearchResult]) -> list[Document]:
        """将向量检索结果转换为统一的 LangChain Document。"""
        documents: list[Document] = []
        for result in results:
            score = float(result.score) if result.score is not None else None
            if score is not None and score < self.minimum_relevance:
                continue
            metadata = result.document.metadata
            citation_id = str(
                metadata.get("review_id")
                or metadata.get("product_id")
                or metadata.get("order_id")
                or metadata.get("id")
                or result.document.id
            )
            snippet = truncate_snippet(result.document.content)
            if not snippet:
                continue
            documents.append(
                Document(
                    page_content=snippet,
                    metadata={"namespace": namespace, "citation_id": citation_id, "score": score},
                )
            )
        return documents

    def _doc_score(self, doc: Document) -> float:
        """提取文档分数用于排序。"""
        score = doc.metadata.get("score")
        if isinstance(score, int | float):
            return float(score)
        return -1.0


class EcommerceSufficiencyJudge(SufficiencyJudge):
    """为电商检索场景提供可解释的充分性判断策略。"""

    inventory_keywords: tuple[str, ...] = ("库存", "有货", "现货", "缺货", "到货", "补货")
    detail_keywords: tuple[str, ...] = ("参数", "规格", "配置", "价格", "多少钱", "品牌", "摄像")
    review_keywords: tuple[str, ...] = ("评价", "口碑", "体验", "值得买", "好用", "优缺点")
    order_keywords: tuple[str, ...] = ("订单", "发货", "物流", "快递", "单号", "签收", "运输")

    def invoke(
        self,
        input: RetrievalContext,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> SufficiencyDecision:
        """根据当前轮结果决定结束、改写、切工具还是追问用户。"""
        plan = input.plan
        result = input.results[-1]
        normalized_query = plan.user_query.lower()
        current_tool = plan.selected_tool
        result_count = len(result.records)
        top_product_id = self._resolve_top_product_id(result.records)

        if result_count == 0:
            if plan.round_index >= plan.max_rounds:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="ask_user",
                    reason="No relevant evidence was found after the allowed retrieval attempts.",
                    follow_up_question="请补充更具体的商品名称、预算范围，或明确想了解的参数/库存信息。",
                )
            return SufficiencyDecision(
                is_sufficient=False,
                next_action="rewrite",
                reason="The current retrieval result is empty and needs a narrower or clearer query.",
            )

        if current_tool == "product_semantic_search":
            if self._contains_any(normalized_query, self.inventory_keywords) and top_product_id:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="Product candidates are available; inventory status should be confirmed next.",
                    suggested_tool="inventory_lookup",
                    metadata={"resolved_query": top_product_id},
                )
            if self._contains_any(normalized_query, self.detail_keywords) and top_product_id:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="Product candidates are available; exact details should be filled in next.",
                    suggested_tool="product_detail_lookup",
                    metadata={"resolved_query": top_product_id},
                )
            if self._contains_any(normalized_query, self.review_keywords) and "review_semantic_search" not in plan.attempted_tools:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="Need review evidence to support recommendation quality or user sentiment.",
                    suggested_tool="review_semantic_search",
                )
            if self._contains_any(normalized_query, self.order_keywords) and "order_semantic_search" not in plan.attempted_tools:
                return SufficiencyDecision(
                    is_sufficient=False,
                    next_action="switch_tool",
                    reason="Need order evidence to support order tracking or logistics inquiries.",
                    suggested_tool="order_semantic_search",
                )

        return SufficiencyDecision(
            is_sufficient=True,
            next_action="finish",
            reason="Current evidence is sufficient to support a grounded answer.",
            confidence=result.confidence,
        )

    def _contains_any(self, query: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in query for keyword in keywords)

    def _resolve_top_product_id(self, records: list[dict[str, Any]]) -> str | None:
        for record in records:
            product_id = record.get("product_id") or record.get("citation_id")
            if isinstance(product_id, str) and product_id:
                return product_id
        return None


class EcommerceQueryRewriter(QueryRewriter):
    """在证据不足时，对电商检索 query 做轻量改写。"""

    def invoke(
        self,
        input: RetrievalContext,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> QueryRewrite:
        """基于当前上下文生成下一轮检索 query。"""
        query = input.plan.active_query.strip()
        rewritten = (
            query.replace("有没有货", "库存")
            .replace("有货吗", "库存")
            .replace("值得买吗", "评价")
            .replace("怎么样", "参数 评价")
        ).strip()
        if rewritten == query:
            rewritten = f"{query} 商品 参数 评价"
        return QueryRewrite(
            query=rewritten,
            reason="Broadened the query with product, parameter, or review terms for the next round.",
            metadata={"original_query": query},
        )


def create_agentic_knowledge_retriever(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: KnowledgeService | None = None,
    product_store: ProductCatalogStore | None = None,
    max_rounds: int = 3,
) -> AgenticRetriever:
    """构建面向电商知识库的 AgenticRetriever。

    该函数组装以下组件：
    1. 检索工具集：包含语义检索（商品/评价/订单）和精确查询（库存/详情）
    2. 充分性判断器：决定检索结果是否足够回答用户问题
    3. 查询改写器：在证据不足时改写查询以获取更多结果

    数据源说明：
    - resolved_knowledge_service: 提供向量库的语义检索能力
      （商品、评价、订单的语义搜索，数据来自向量数据库）
    - resolved_product_store: 提供 JSON 文件的精确查询能力
      （库存、价格等结构化数据，数据来自 products.json）

    使用场景：
    - 用户问"续航好的手机" → 语义检索找到相关商品
    - 用户问"iPhone 16 有货吗" → 语义检索找到商品 → 精确查询库存
    """
    current_settings = app_settings or settings
    resolved_knowledge_service = knowledge_service or create_knowledge_service(current_settings)
    resolved_product_store = product_store or ProductCatalogStore(data_dir=current_settings.data_dir)
    tools = build_agentic_retrieval_tools(
        app_settings=current_settings,
        knowledge_service=resolved_knowledge_service,
        product_store=resolved_product_store,
    )
    return AgenticRetriever(
        tools={tool.name: tool for tool in tools},
        default_tool="product_semantic_search",
        sufficiency_judge=EcommerceSufficiencyJudge(),
        query_rewriter=EcommerceQueryRewriter(),
        max_rounds=max_rounds,
    )

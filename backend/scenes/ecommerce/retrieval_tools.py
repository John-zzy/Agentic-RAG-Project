from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from backend.platform.config.settings import AppSettings, settings
from backend.platform.knowledge.base.relevance import (
    filter_managed_document_results,
    filter_low_relevance_document_results,
)
from backend.platform.knowledge.base.store import VectorSearchResult, VectorStoreDocument
from backend.platform.knowledge.base.text import truncate_snippet
from backend.platform.rag.core import RetrievalCitation, RetrievalResult, RetrievalTool
from backend.platform.tools import BaseJsonStore, ToolResult, build_structured_tool
from backend.scenes.ecommerce.knowledge_service import KnowledgeService, create_knowledge_service
from backend.scenes.ecommerce.loader import preload_knowledge_base


PRODUCTS_FILE_NAME = "products.json"
ECOMMERCE_SCENE_NAME = "ecommerce"
GENERIC_ASSISTANT_SCENE_NAME = "generic_assistant"
ECOMMERCE_FALLBACK_MESSAGE = "暂时没有检索到足够相关的电商知识。请补充更具体的商品名、订单信息或文档关键词。"
GENERIC_ASSISTANT_FALLBACK_MESSAGE = "暂时没有检索到足够相关的文档知识。请补充更具体的主题、术语或文档范围，我再继续帮你查。"


class SemanticSearchInput(BaseModel):
    """语义检索工具的输入参数。"""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class ReviewSemanticSearchInput(SemanticSearchInput):
    """评价语义检索工具的输入参数。"""

    product_id: str | None = None


class ProductLookupInput(BaseModel):
    """商品精确查询类工具的输入参数。"""

    product_id: str = Field(min_length=1)


@dataclass
class ProductCatalogStore(BaseJsonStore):
    """封装本地商品目录读取，供库存与详情工具复用。

    继承 BaseJsonStore 复用 JSON 文件读写能力，专注于商品数据的业务操作。
    数据存储在本地 JSON 文件（products.json）中，提供精确查询能力。

    与 KnowledgeService 的区别：
    - ProductCatalogStore：基于 JSON 文件的精确查询，适合库存、价格等结构化数据
    - KnowledgeService：基于向量库的语义检索，适合模糊匹配、相似搜索
    """

    def load_products(self) -> list[dict[str, Any]]:
        """读取商品列表；若文件不存在则返回空列表。"""
        return self._load_json_list(PRODUCTS_FILE_NAME)

    def find_product(self, product_id: str) -> dict[str, Any] | None:
        """按商品 ID 精确查询结构化商品数据，不存在时返回 None。

        遍历商品列表进行线性查找，适用于数据量较小的场景。
        """
        for product in self.load_products():
            if str(product.get("product_id")) == product_id:
                return product
        return None

    def find_product_by_query(self, query: str) -> dict[str, Any] | None:
        """在 query 中出现明确商品名时，直接解析对应商品。

        用于将用户自然语言中的商品名映射到具体商品，
        例如 "iPhone 16 有货吗" → 匹配到 iPhone 16 商品记录。
        """
        normalized_query = query.lower()
        for product in self.load_products():
            name = str(product.get("name", "")).lower()
            if name and name in normalized_query:
                return product
        return None


def build_retrieval_tools(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: KnowledgeService | None = None,
    product_store: ProductCatalogStore | None = None,
) -> tuple[BaseTool, ...]:
    """按统一顺序构建电商检索工具集合。"""
    current_settings = app_settings or settings
    resolved_knowledge_service = knowledge_service or create_knowledge_service(current_settings)
    resolved_product_store = product_store or ProductCatalogStore(data_dir=current_settings.data_dir)
    if hasattr(resolved_knowledge_service, "store"):
        preload_knowledge_base(current_settings, store=resolved_knowledge_service.store)
    return (
        _build_semantic_search_tool(
            resolved_knowledge_service,
            namespace="products",
            tool_name="product_semantic_search",
            description="Search semantically similar products for the user's shopping question.",
            product_store=resolved_product_store,
        ),
        _build_semantic_search_tool(
            resolved_knowledge_service,
            namespace="reviews",
            tool_name="review_semantic_search",
            description="Search review evidence and recommendation reasons related to the query.",
            args_schema=ReviewSemanticSearchInput,
            supports_filter=True,
        ),
        _build_semantic_search_tool(
            resolved_knowledge_service,
            namespace="orders",
            tool_name="order_semantic_search",
            description="Search order information semantically for order tracking or status inquiries.",
        ),
        _build_inventory_lookup_tool(resolved_product_store),
        _build_product_detail_lookup_tool(resolved_product_store),
    )


def build_generic_document_retrieval_tools(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: KnowledgeService | None = None,
) -> tuple[BaseTool, ...]:
    """构建通用知识助手场景默认使用的最小文档检索工具集。"""
    current_settings = app_settings or settings
    resolved_knowledge_service = knowledge_service or create_knowledge_service(current_settings)
    return (
        _build_knowledge_document_tool(
            resolved_knowledge_service,
            files_root=str(current_settings.data_dir / "files"),
        ),
    )


class SemanticRetrievalTool(RetrievalTool):
    """通用语义检索工具基类，支持商品、评价、订单的统一召回。"""

    name: str
    description: str
    namespace: str
    knowledge_service: Any = Field(exclude=True)
    product_store: Any | None = Field(default=None, exclude=True)
    search_method: str = Field(exclude=True)  # "search_products", "search_reviews", "search_orders"
    default_top_k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def retrieve(self, query: str, *, run_manager: Any | None = None) -> RetrievalResult:
        """从指定知识库执行语义检索并返回标准化结果。"""
        search_func = getattr(self.knowledge_service, self.search_method)
        vector_results = search_func(query=query, top_k=self.default_top_k)
        
        if self.namespace == "products" and self.product_store:
            vector_results = _inject_named_product_match(query, vector_results, self.product_store)
        if self.namespace == "orders":
            vector_results = _rank_order_results(query, vector_results)
        
        return build_retrieval_result(
            tool_name=self.name,
            namespace=self.namespace,
            query=query,
            vector_results=vector_results,
        )


class KnowledgeDocumentSemanticRetrievalTool(RetrievalTool):
    """用户上传知识文档的语义检索工具。"""

    name: str = "knowledge_document_search"
    description: str = "Search semantically relevant user-uploaded knowledge documents."
    knowledge_service: Any = Field(exclude=True)
    files_root: str = Field(exclude=True)
    default_top_k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def retrieve(self, query: str, *, run_manager: Any | None = None) -> RetrievalResult:
        """从文档分块索引中检索用户上传的知识。"""
        vector_results = self.knowledge_service.search_document_chunks(query=query, top_k=self.default_top_k)
        vector_results = filter_managed_document_results(vector_results, files_root=self.files_root)
        vector_results = filter_low_relevance_document_results(vector_results)
        return build_retrieval_result(
            tool_name=self.name,
            namespace="documents",
            query=query,
            vector_results=vector_results,
        )


def build_knowledge_document_retrieval_tool(
    knowledge_service: KnowledgeService,
    *,
    files_root: str,
) -> KnowledgeDocumentSemanticRetrievalTool:
    """构建可挂载到 scene runtime 的文档检索 RetrievalTool。"""
    return KnowledgeDocumentSemanticRetrievalTool(
        knowledge_service=knowledge_service,
        files_root=files_root,
    )


def build_semantic_retrieval_tool(
    knowledge_service: KnowledgeService,
    *,
    namespace: str,
    tool_name: str,
    description: str,
    product_store: ProductCatalogStore | None = None,
) -> SemanticRetrievalTool:
    """构建通用语义检索工具的工厂函数。"""
    return SemanticRetrievalTool(
        name=tool_name,
        description=description,
        namespace=namespace,
        knowledge_service=knowledge_service,
        product_store=product_store,
        search_method=f"search_{namespace}",
    )


class InventoryLookupRetrievalTool(RetrievalTool):
    """库存查询工具，返回标准化库存状态。"""

    name: str = "inventory_lookup"
    description: str = "Look up structured inventory status for a known product ID."
    product_store: Any = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def retrieve(self, query: str, *, run_manager: Any | None = None) -> RetrievalResult:
        """按 product_id 查询库存信息。"""
        product = self.product_store.find_product(query.strip())
        if product is None:
            return RetrievalResult.fail(
                tool_name=self.name,
                query=query,
                error=f"Product '{query}' was not found.",
                metadata={"namespace": "inventory"},
            )

        record = _build_inventory_record(product)
        snippet = record["inventory_summary"]
        return RetrievalResult.ok(
            tool_name=self.name,
            query=query,
            records=[record],
            documents=[
                _build_document(
                    snippet=snippet,
                    namespace="inventory",
                    citation_id=record["product_id"],
                    score=1.0,
                    extra_metadata={"product_id": record["product_id"]},
                )
            ],
            citations=[
                _build_citation(
                    citation_id=record["product_id"],
                    namespace="inventory",
                    snippet=snippet,
                    metadata={"product_id": record["product_id"]},
                )
            ],
            confidence=1.0,
            metadata={"namespace": "inventory", "result_count": 1},
        )


class ProductDetailLookupRetrievalTool(RetrievalTool):
    """商品详情精确查询工具，语义检索后可补充精确信息。"""

    name: str = "product_detail_lookup"
    description: str = "Look up structured product details by product ID."
    product_store: Any = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def retrieve(self, query: str, *, run_manager: Any | None = None) -> RetrievalResult:
        """按 product_id 返回结构化详情。"""
        product = self.product_store.find_product(query.strip())
        if product is None:
            return RetrievalResult.fail(
                tool_name=self.name,
                query=query,
                error=f"Product '{query}' was not found.",
                metadata={"namespace": "product_detail"},
            )

        record = _build_product_detail_record(product)
        snippet = (
            f"{record['product_name']}，价格 {record['price']} {record['currency']}，"
            f"核心参数：{record['spec_summary']}"
        )
        return RetrievalResult.ok(
            tool_name=self.name,
            query=query,
            records=[record],
            documents=[
                _build_document(
                    snippet=snippet,
                    namespace="product_detail",
                    citation_id=record["product_id"],
                    score=1.0,
                    extra_metadata={"product_id": record["product_id"]},
                )
            ],
            citations=[
                _build_citation(
                    citation_id=record["product_id"],
                    namespace="product_detail",
                    snippet=snippet,
                    metadata={"product_id": record["product_id"]},
                )
            ],
            confidence=1.0,
            metadata={"namespace": "product_detail", "result_count": 1},
        )


def build_agentic_retrieval_tools(
    app_settings: AppSettings | None = None,
    *,
    knowledge_service: KnowledgeService | None = None,
    product_store: ProductCatalogStore | None = None,
    files_root: str | None = None,
) -> tuple[RetrievalTool, ...]:
    """构建供 AgenticRetriever 使用的 RetrievalTool 集合。"""
    current_settings = app_settings or settings
    resolved_knowledge_service = knowledge_service or create_knowledge_service(current_settings)
    resolved_product_store = product_store or ProductCatalogStore(data_dir=current_settings.data_dir)
    resolved_files_root = files_root or str(current_settings.data_dir / "files")
    if hasattr(resolved_knowledge_service, "store"):
        preload_knowledge_base(current_settings, store=resolved_knowledge_service.store)
    return (
        build_semantic_retrieval_tool(
            resolved_knowledge_service,
            namespace="products",
            tool_name="product_semantic_search",
            description="Search semantically similar products for the user's shopping intent.",
            product_store=resolved_product_store,
        ),
        build_semantic_retrieval_tool(
            resolved_knowledge_service,
            namespace="reviews",
            tool_name="review_semantic_search",
            description="Search review evidence related to the user's shopping question.",
        ),
        build_semantic_retrieval_tool(
            resolved_knowledge_service,
            namespace="orders",
            tool_name="order_semantic_search",
            description="Search order information semantically for order tracking or status inquiries.",
        ),
        KnowledgeDocumentSemanticRetrievalTool(
            knowledge_service=resolved_knowledge_service,
            files_root=resolved_files_root,
        ),
        InventoryLookupRetrievalTool(product_store=resolved_product_store),
        ProductDetailLookupRetrievalTool(product_store=resolved_product_store),
    )


def build_retrieval_result(
    *,
    tool_name: str,
    namespace: str,
    query: str,
    vector_results: list[VectorSearchResult],
) -> RetrievalResult:
    """将向量检索结果映射为统一的 Agentic Retrieval 结果结构。"""
    records = [_build_semantic_record(namespace=namespace, result=result) for result in vector_results]
    citations = [
        _build_citation(
            citation_id=record["citation_id"],
            namespace=namespace,
            snippet=record["snippet"],
            metadata={
                **record.get("metadata", {}),
                "product_id": record.get("product_id"),
                "score": record.get("score"),
            },
        )
        for record in records
    ]
    documents = [
        _build_document(
            snippet=record["snippet"],
            namespace=namespace,
            citation_id=record["citation_id"],
            score=record.get("score"),
            extra_metadata={
                **record.get("metadata", {}),
                "product_id": record.get("product_id"),
            },
        )
        for record in records
    ]
    confidence = _average_score(records)
    return RetrievalResult.ok(
        tool_name=tool_name,
        query=query,
        records=records,
        documents=documents,
        citations=citations,
        confidence=confidence,
        metadata={"namespace": namespace, "result_count": len(records)},
    )


def _build_semantic_search_tool(
    knowledge_service: KnowledgeService,
    *,
    namespace: str,
    tool_name: str,
    description: str,
    args_schema: type[BaseModel] = SemanticSearchInput,
    product_store: ProductCatalogStore | None = None,
    supports_filter: bool = False,
) -> BaseTool:
    """构建通用语义检索 StructuredTool 的工厂函数。"""
    search_method = getattr(knowledge_service, f"search_{namespace}")

    def semantic_search(query: str, top_k: int = 5, product_id: str | None = None) -> ToolResult:
        filters = {"product_id": product_id} if supports_filter and product_id else None
        vector_results = search_method(query=query, top_k=top_k, filters=filters)
        
        if namespace == "products" and product_store:
            vector_results = _inject_named_product_match(query, vector_results, product_store)
        
        retrieval_result = build_retrieval_result(
            tool_name=tool_name,
            namespace=namespace,
            query=query,
            vector_results=vector_results,
        )
        if filters:
            retrieval_result.metadata["filters"] = filters
        return _to_tool_result(retrieval_result)

    return build_structured_tool(
        name=tool_name,
        description=description,
        capability_type="retrieval",
        args_schema=args_schema,
        func=semantic_search,
    )


def _build_inventory_lookup_tool(product_store: ProductCatalogStore) -> BaseTool:
    """构建库存查询 StructuredTool。"""

    def inventory_lookup(product_id: str) -> ToolResult:
        product = product_store.find_product(product_id)
        if product is None:
            return ToolResult.fail(
                tool_name="inventory_lookup",
                error=f"Product '{product_id}' was not found.",
                metadata={"namespace": "inventory"},
            )
        record = _build_inventory_record(product)
        return ToolResult.ok(
            tool_name="inventory_lookup",
            records=[record],
            citations=[
                {
                    "citation_id": record["product_id"],
                    "namespace": "inventory",
                    "snippet": record["inventory_summary"],
                }
            ],
            confidence=1.0,
            metadata={"namespace": "inventory", "result_count": 1},
        )

    return build_structured_tool(
        name="inventory_lookup",
        description="Look up inventory availability and stock status for a known product ID.",
        capability_type="retrieval",
        args_schema=ProductLookupInput,
        func=inventory_lookup,
    )


def _build_knowledge_document_tool(
    knowledge_service: KnowledgeService,
    *,
    files_root: str,
) -> BaseTool:
    """构建上传文档检索 StructuredTool，供通用场景与兼容链路复用。"""

    def knowledge_document_search(query: str, top_k: int = 5) -> ToolResult:
        vector_results = knowledge_service.search_document_chunks(query=query, top_k=top_k)
        vector_results = filter_managed_document_results(vector_results, files_root=files_root)
        vector_results = filter_low_relevance_document_results(vector_results)
        retrieval_result = build_retrieval_result(
            tool_name="knowledge_document_search",
            namespace="documents",
            query=query,
            vector_results=vector_results,
        )
        retrieval_result.metadata["scene"] = GENERIC_ASSISTANT_SCENE_NAME
        return _to_tool_result(retrieval_result)

    return build_structured_tool(
        name="knowledge_document_search",
        description="Search semantically relevant uploaded knowledge documents.",
        capability_type="retrieval",
        args_schema=SemanticSearchInput,
        func=knowledge_document_search,
    )


def _build_product_detail_lookup_tool(product_store: ProductCatalogStore) -> BaseTool:
    """构建商品详情精确查询 StructuredTool。"""

    def product_detail_lookup(product_id: str) -> ToolResult:
        product = product_store.find_product(product_id)
        if product is None:
            return ToolResult.fail(
                tool_name="product_detail_lookup",
                error=f"Product '{product_id}' was not found.",
                metadata={"namespace": "product_detail"},
            )
        record = _build_product_detail_record(product)
        snippet = (
            f"{record['product_name']}，价格 {record['price']} {record['currency']}，"
            f"核心参数：{record['spec_summary']}"
        )
        return ToolResult.ok(
            tool_name="product_detail_lookup",
            records=[record],
            citations=[
                {
                    "citation_id": record["product_id"],
                    "namespace": "product_detail",
                    "snippet": snippet,
                }
            ],
            confidence=1.0,
            metadata={"namespace": "product_detail", "result_count": 1},
        )

    return build_structured_tool(
        name="product_detail_lookup",
        description="Look up structured product details by product ID after semantic retrieval.",
        capability_type="retrieval",
        args_schema=ProductLookupInput,
        func=product_detail_lookup,
    )


def _build_semantic_record(namespace: str, result: VectorSearchResult) -> dict[str, Any]:
    """将商品/评价/订单向量结果映射为统一 record。"""
    metadata = result.document.metadata
    citation_id = str(
        metadata.get("chunk_id")
        or metadata.get("review_id")
        or metadata.get("product_id")
        or metadata.get("order_id")
        or metadata.get("document_id")
        or metadata.get("source_path")
        or metadata.get("id")
        or result.document.id
    )
    score = float(result.score) if result.score is not None else None
    return {
        "record_type": namespace.removesuffix("s"),
        "namespace": namespace,
        "citation_id": citation_id,
        "product_id": metadata.get("product_id") or citation_id,
        "title": metadata.get("title") or metadata.get("name") or metadata.get("order_id") or citation_id,
        "snippet": truncate_snippet(result.document.content),
        "score": score,
        "metadata": metadata,
    }


def _inject_named_product_match(
    query: str,
    vector_results: list[VectorSearchResult],
    product_store: ProductCatalogStore,
) -> list[VectorSearchResult]:
    """当 query 中出现明确商品名时，将该商品稳定放到首位。"""
    named_product = product_store.find_product_by_query(query)
    ranked_results = _rank_product_results(query, vector_results)
    if named_product is None:
        return ranked_results

    product_id = str(named_product["product_id"])
    match_index = next(
        (index for index, result in enumerate(ranked_results) if result.document.metadata.get("product_id") == product_id),
        None,
    )
    if match_index is not None:
        matched_result = ranked_results.pop(match_index)
        return [matched_result, *ranked_results]

    synthetic_result = VectorSearchResult(
        document=VectorStoreDocument(
            id=product_id,
            content=_build_product_semantic_content(named_product),
            metadata={
                "product_id": product_id,
                "name": str(named_product["name"]),
                "category": str(named_product.get("category", "")),
            },
        ),
        score=1.0,
    )
    deduped = [
        result
        for result in ranked_results
        if str(result.document.metadata.get("product_id") or result.document.id) != product_id
    ]
    return [synthetic_result, *deduped]


def _rank_product_results(query: str, vector_results: list[VectorSearchResult]) -> list[VectorSearchResult]:
    """对显式命中商品名的结果做轻量排序增益。"""
    normalized_query = query.lower()

    def sort_key(result: VectorSearchResult) -> tuple[int, float]:
        name = str(result.document.metadata.get("name", "")).lower()
        exact_name_hit = 1 if name and name in normalized_query else 0
        score = float(result.score) if result.score is not None else -1.0
        return exact_name_hit, score

    return sorted(vector_results, key=sort_key, reverse=True)


def _rank_order_results(query: str, vector_results: list[VectorSearchResult]) -> list[VectorSearchResult]:
    """对显式命中订单号、运单号、承运商的结果做轻量排序增益。"""
    normalized_query = query.lower()

    def sort_key(result: VectorSearchResult) -> tuple[int, int, int, float]:
        metadata = result.document.metadata
        order_id = str(metadata.get("order_id", "")).lower()
        tracking_no = str(metadata.get("tracking_no", "")).lower()
        carrier = str(metadata.get("carrier", "")).lower()
        exact_order_hit = 1 if order_id and order_id in normalized_query else 0
        exact_tracking_hit = 1 if tracking_no and tracking_no in normalized_query else 0
        carrier_hit = 1 if carrier and carrier in normalized_query else 0
        score = float(result.score) if result.score is not None else -1.0
        return exact_order_hit, exact_tracking_hit, carrier_hit, score

    return sorted(vector_results, key=sort_key, reverse=True)


def _build_product_semantic_content(product: dict[str, Any]) -> str:
    """为显式命中商品名的合成结果构建内容摘要。"""
    specs = product.get("specs", {})
    spec_summary = "；".join(f"{key}: {value}" for key, value in specs.items())
    return (
        f"{product['name']}。"
        f"{product.get('description', '')}。"
        f"分类：{product.get('category', '')}。"
        f"规格：{spec_summary}"
    )


def _build_inventory_record(product: dict[str, Any]) -> dict[str, Any]:
    """将库存数据标准化为统一 record。"""
    inventory = product.get("inventory", {})
    summary = (
        f"{product['name']} 当前库存状态为 {inventory.get('status', 'unknown')}，"
        f"库存数量 {inventory.get('quantity', 0)}，仓库 {inventory.get('warehouse', 'unknown')}。"
    )
    return {
        "record_type": "inventory",
        "namespace": "inventory",
        "product_id": str(product["product_id"]),
        "product_name": str(product["name"]),
        "inventory_status": str(inventory.get("status", "unknown")),
        "inventory_quantity": int(inventory.get("quantity", 0)),
        "warehouse": str(inventory.get("warehouse", "")),
        "inventory_summary": summary,
    }


def _build_product_detail_record(product: dict[str, Any]) -> dict[str, Any]:
    """将商品详情标准化为统一 record。"""
    specs = product.get("specs", {})
    spec_summary = "；".join(f"{key}: {value}" for key, value in specs.items())
    return {
        "record_type": "product_detail",
        "namespace": "product_detail",
        "product_id": str(product["product_id"]),
        "product_name": str(product["name"]),
        "category": str(product.get("category", "")),
        "description": str(product.get("description", "")),
        "price": float(product["price"]),
        "currency": str(product.get("currency", "CNY")),
        "specs": specs,
        "spec_summary": spec_summary,
    }


def _build_citation(
    *,
    citation_id: str,
    namespace: str,
    snippet: str,
    metadata: dict[str, Any] | None = None,
) -> RetrievalCitation:
    """构建统一 citation。"""
    return RetrievalCitation(
        citation_id=citation_id,
        snippet=snippet,
        source_type=namespace,
        metadata=metadata or {},
    )


def _build_document(
    *,
    snippet: str,
    namespace: str,
    citation_id: str,
    score: float | None,
    extra_metadata: dict[str, Any] | None = None,
) -> Document:
    """构建统一 Document。"""
    metadata = {
        "namespace": namespace,
        "citation_id": citation_id,
        "score": score,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return Document(page_content=snippet, metadata=metadata)


def _average_score(records: list[dict[str, Any]]) -> float | None:
    """计算结果集合的平均分，作为工具级 confidence。"""
    scores = [float(score) for score in (record.get("score") for record in records) if isinstance(score, int | float)]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _to_tool_result(retrieval_result: RetrievalResult) -> ToolResult:
    """将 Agentic Retrieval 结果映射为通用 ToolResult。"""
    if not retrieval_result.success:
        return ToolResult.fail(
            tool_name=retrieval_result.tool_name,
            error=retrieval_result.error or "Unknown retrieval error.",
            metadata=retrieval_result.metadata,
        )
    return ToolResult.ok(
        tool_name=retrieval_result.tool_name,
        records=retrieval_result.records,
        citations=[
            {
                "citation_id": citation.citation_id,
                "namespace": citation.source_type,
                "snippet": citation.snippet,
                "metadata": citation.metadata,
            }
            for citation in retrieval_result.citations
        ],
        confidence=retrieval_result.confidence,
        metadata=retrieval_result.metadata,
    )

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from backend.platform.knowledge.base.text import truncate_snippet
from backend.platform.rag.contracts import RetrievalCitation, RetrievalResult
from backend.platform.rag.retrieval.documents import DocumentChunkRetrievalResult
from backend.platform.search_foundation import VectorSearchResult, VectorStoreDocument
from backend.platform.tools import ToolResult
from backend.scenes.ecommerce.tools.stores import ProductCatalogStore


def build_retrieval_result(
    *,
    tool_name: str,
    namespace: str,
    query: str,
    vector_results: list[VectorSearchResult] | list[DocumentChunkRetrievalResult],
) -> RetrievalResult:
    """将检索结果映射为统一的 Agentic Retrieval 结果结构。"""
    records = [build_semantic_record(namespace=namespace, result=result) for result in vector_results]
    citations = [
        build_citation(
            citation_id=record["citation_id"],
            namespace=namespace,
            snippet=record["snippet"],
            metadata={
                **record.get("metadata", {}),
                "product_id": record.get("product_id"),
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
        build_document(
            snippet=record["snippet"],
            namespace=namespace,
            citation_id=record["citation_id"],
            score=record.get("score"),
            extra_metadata={
                **record.get("metadata", {}),
                "product_id": record.get("product_id"),
                "vector_score": record.get("vector_score"),
                "keyword_score": record.get("keyword_score"),
                "vector_rank": record.get("vector_rank"),
                "keyword_rank": record.get("keyword_rank"),
                "matched_by": record.get("matched_by", []),
            },
        )
        for record in records
    ]
    return RetrievalResult.ok(
        tool_name=tool_name,
        query=query,
        records=records,
        documents=documents,
        citations=citations,
        confidence=average_score(records),
        metadata={"namespace": namespace, "result_count": len(records)},
    )


def to_tool_result(retrieval_result: RetrievalResult) -> ToolResult:
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


def build_semantic_record(
    namespace: str,
    result: VectorSearchResult | DocumentChunkRetrievalResult,
) -> dict[str, Any]:
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
    record = {
        "record_type": namespace.removesuffix("s"),
        "namespace": namespace,
        "citation_id": citation_id,
        "product_id": metadata.get("product_id") or citation_id,
        "title": metadata.get("title") or metadata.get("name") or metadata.get("order_id") or citation_id,
        "snippet": truncate_snippet(result.document.content),
        "score": float(result.score) if result.score is not None else None,
        "metadata": metadata,
    }
    if isinstance(result, DocumentChunkRetrievalResult):
        record["vector_score"] = float(result.vector_score) if result.vector_score is not None else None
        record["keyword_score"] = float(result.keyword_score) if result.keyword_score is not None else None
        record["vector_rank"] = result.vector_rank
        record["keyword_rank"] = result.keyword_rank
        record["matched_by"] = list(result.matched_by)
    return record


def inject_named_product_match(
    query: str,
    vector_results: list[VectorSearchResult],
    product_store: ProductCatalogStore,
) -> list[VectorSearchResult]:
    """当 query 中出现明确商品名时，将该商品稳定放到首位。"""
    named_product = product_store.find_product_by_query(query)
    ranked_results = rank_product_results(query, vector_results)
    if named_product is None:
        return ranked_results

    product_id = str(named_product["product_id"])
    match_index = next(
        (
            index
            for index, result in enumerate(ranked_results)
            if result.document.metadata.get("product_id") == product_id
        ),
        None,
    )
    if match_index is not None:
        matched_result = ranked_results.pop(match_index)
        return [matched_result, *ranked_results]

    synthetic_result = VectorSearchResult(
        document=VectorStoreDocument(
            id=product_id,
            content=build_product_semantic_content(named_product),
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


def rank_product_results(query: str, vector_results: list[VectorSearchResult]) -> list[VectorSearchResult]:
    """对显式命中商品名的结果做轻量排序增益。"""
    normalized_query = query.lower()

    def sort_key(result: VectorSearchResult) -> tuple[int, float]:
        name = str(result.document.metadata.get("name", "")).lower()
        exact_name_hit = 1 if name and name in normalized_query else 0
        score = float(result.score) if result.score is not None else -1.0
        return exact_name_hit, score

    return sorted(vector_results, key=sort_key, reverse=True)


def rank_order_results(query: str, vector_results: list[VectorSearchResult]) -> list[VectorSearchResult]:
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


def build_product_semantic_content(product: dict[str, Any]) -> str:
    """为显式命中商品名的合成结果构建内容摘要。"""
    specs = product.get("specs", {})
    spec_summary = "；".join(f"{key}: {value}" for key, value in specs.items())
    return (
        f"{product['name']}。"
        f"{product.get('description', '')}。"
        f"分类：{product.get('category', '')}。"
        f"规格：{spec_summary}"
    )


def build_inventory_record(product: dict[str, Any]) -> dict[str, Any]:
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


def build_product_detail_record(product: dict[str, Any]) -> dict[str, Any]:
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


def build_citation(
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


def build_document(
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


def average_score(records: list[dict[str, Any]]) -> float | None:
    """计算结果集合的平均分，作为工具级 confidence。"""
    scores = [float(score) for score in (record.get("score") for record in records) if isinstance(score, int | float)]
    if not scores:
        return None
    return sum(scores) / len(scores)


def filter_vector_results_by_min_relevance(
    results: list[VectorSearchResult],
    minimum_relevance: float | None,
) -> list[VectorSearchResult]:
    """按 scene 阈值过滤低相关语义结果。"""
    if minimum_relevance is None:
        return results
    return [
        result
        for result in results
        if result.score is None or float(result.score) >= minimum_relevance
    ]

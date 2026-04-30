from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from backend.config.settings import AppSettings, settings
from backend.knowledge.base.store import VectorStore, VectorStoreFactory
from backend.knowledge.ecommerce.extractor import build_product_document, build_review_document


class KnowledgeLoadSummary(BaseModel):
    """描述商品与评论预加载的数量结果。"""

    products_loaded: int
    reviews_loaded: int


def load_json_records(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 文件并返回记录列表。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def preload_knowledge_base(
    app_settings: AppSettings | None = None,
    store: VectorStore | None = None,
) -> KnowledgeLoadSummary:
    """将商品与评论数据预加载到向量库。"""
    resolved_settings = app_settings or settings
    resolved_store = store or VectorStoreFactory.create(resolved_settings)

    products = load_json_records(resolved_settings.data_dir / "products.json")
    reviews = load_json_records(resolved_settings.data_dir / "reviews.json")
    product_documents = [build_product_document(product) for product in products]
    review_documents = [build_review_document(review) for review in reviews]

    resolved_store.ensure_collections()
    resolved_store.upsert_documents("products", product_documents)
    resolved_store.upsert_documents("reviews", review_documents)

    return KnowledgeLoadSummary(
        products_loaded=len(product_documents),
        reviews_loaded=len(review_documents),
    )

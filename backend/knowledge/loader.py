from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from backend.config.settings import AppSettings, settings
from backend.knowledge.extractor import build_product_document, build_review_document
from backend.knowledge.store import VectorStore, VectorStoreFactory


class KnowledgeLoadSummary(BaseModel):
    products_loaded: int
    reviews_loaded: int


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def preload_knowledge_base(
    app_settings: AppSettings | None = None,
    store: VectorStore | None = None,
) -> KnowledgeLoadSummary:
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

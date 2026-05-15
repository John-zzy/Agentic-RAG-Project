from __future__ import annotations

from typing import Any

from backend.platform.config.settings import AppSettings
from backend.platform.knowledge.base.store import KnowledgeDocumentRepository, VectorStoreDocument
from backend.platform.knowledge.documents.models import KnowledgeDocumentStoreError


class KnowledgeDocumentRepositoryGateway:
    """把文档仓储调用统一包一层，方便在一个地方转换异常。"""

    def __init__(self, app_settings: AppSettings, repository: KnowledgeDocumentRepository) -> None:
        self.app_settings = app_settings
        self.repository = repository

    def ensure_document_indexes(self) -> None:
        self._call("ensure document indexes", self.repository.ensure_document_indexes)

    def upsert_document_record(self, record: dict[str, Any]) -> None:
        self._call("upsert document record", self.repository.upsert_document_record, record)

    def get_document_record(self, document_id: str) -> dict[str, Any] | None:
        return self._call("get document record", self.repository.get_document_record, document_id)

    def list_document_records(self, namespace: str | None = None) -> list[dict[str, Any]]:
        return self._call("list document records", self.repository.list_document_records, namespace)

    def search_document_chunks(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
    ) -> object:
        return self._call(
            "search document chunks",
            self.repository.search_document_chunks,
            query,
            top_k,
            namespace,
        )

    def delete_document_record(self, document_id: str) -> None:
        self._call("delete document record", self.repository.delete_document_record, document_id)

    def upsert_document_chunks(self, chunks: list[VectorStoreDocument]) -> None:
        self._call("upsert document chunks", self.repository.upsert_document_chunks, chunks)

    def deactivate_document_chunks(self, document_id: str, document_version: int | None = None) -> None:
        self._call(
            "deactivate document chunks",
            self.repository.deactivate_document_chunks,
            document_id,
            document_version,
        )

    def activate_document_chunks(self, document_id: str, document_version: int) -> None:
        self._call(
            "activate document chunks",
            self.repository.activate_document_chunks,
            document_id,
            document_version,
        )

    def delete_document_chunks(self, chunk_ids: list[str]) -> None:
        self._call("delete document chunks", self.repository.delete_document_chunks, chunk_ids)

    def _call(self, operation: str, method: object, *args: object) -> Any:
        """这里专门做两件事：先调用仓储方法，再把不支持能力的报错改成统一异常。"""
        try:
            return method(*args)  # type: ignore[operator]
        except NotImplementedError as exc:
            raise KnowledgeDocumentStoreError(
                f"Vector store '{self.app_settings.vector_store.provider}' does not support document management: "
                f"{operation}."
            ) from exc

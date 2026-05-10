from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter, HTTPException, Query, Request

from backend.application.runtime.api.knowledge.schemas import (
    KnowledgeDocumentDeleteResponse,
    KnowledgeDocumentDetailResponse,
    KnowledgeDocumentListResponse,
    KnowledgeDocumentOperationResponse,
    KnowledgeDocumentRechunkRequest,
    KnowledgeDocumentRegisterRequest,
    KnowledgeDocumentSummaryResponse,
    KnowledgeFileIndexListResponse,
    KnowledgeFileIndexSummaryResponse,
)
from backend.platform.knowledge.documents.service import (
    KnowledgeDocumentError,
    KnowledgeDocumentNotFoundError,
    KnowledgeDocumentService,
    KnowledgeDocumentStoreError,
)


class KnowledgeDocumentServiceProtocol(Protocol):
    """知识文档路由依赖协议。"""

    def register_document(
        self,
        namespace: str,
        source_path: str,
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool = False,
    ) -> object:
        ...

    def list_documents(self, namespace: str | None = None) -> list[object]:
        ...

    def list_file_indexes(self, namespace: str | None = None) -> list[object]:
        ...

    def get_document(self, document_id: str) -> object:
        ...

    def delete_document(self, document_id: str) -> object:
        ...

    def rechunk_document(
        self,
        document_id: str,
        chunk_size: int,
        chunk_overlap: int,
        keep_version: bool = False,
    ) -> object:
        ...


router = APIRouter(prefix="/knowledge/documents", tags=["knowledge-documents"])


@router.post("", response_model=KnowledgeDocumentOperationResponse)
def register_knowledge_document(
    payload: KnowledgeDocumentRegisterRequest,
    request: Request,
) -> object:
    """注册知识文档并返回新版本。"""
    service = _get_document_service(request)
    try:
        return service.register_document(
            namespace=payload.namespace,
            source_path=payload.source_path,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            keep_version=payload.keep_version,
        )
    except Exception as exc:
        _raise_document_http_error(exc)


@router.get("", response_model=KnowledgeDocumentListResponse)
def list_knowledge_documents(
    request: Request,
    namespace: str | None = Query(default=None),
) -> KnowledgeDocumentListResponse:
    """列出知识文档。"""
    service = _get_document_service(request)
    try:
        documents = service.list_documents(namespace=namespace)
    except Exception as exc:
        _raise_document_http_error(exc)
    return KnowledgeDocumentListResponse(
        documents=[KnowledgeDocumentSummaryResponse.model_validate(_to_response_data(document)) for document in documents]
    )


@router.get("/files", response_model=KnowledgeFileIndexListResponse)
def list_knowledge_files(
    request: Request,
    namespace: str | None = Query(default=None),
) -> KnowledgeFileIndexListResponse:
    """按上传文件聚合索引状态。"""
    service = _get_document_service(request)
    try:
        items = service.list_file_indexes(namespace=namespace)
    except Exception as exc:
        _raise_document_http_error(exc)
    return KnowledgeFileIndexListResponse(
        items=[KnowledgeFileIndexSummaryResponse.model_validate(_to_response_data(item)) for item in items]
    )


@router.get("/{document_id}", response_model=KnowledgeDocumentDetailResponse)
def get_knowledge_document(document_id: str, request: Request) -> object:
    """读取单个知识文档详情。"""
    service = _get_document_service(request)
    try:
        return service.get_document(document_id)
    except Exception as exc:
        _raise_document_http_error(exc)


@router.delete("/{document_id}", response_model=KnowledgeDocumentDeleteResponse)
def delete_knowledge_document(document_id: str, request: Request) -> object:
    """软删除知识文档。"""
    service = _get_document_service(request)
    try:
        return service.delete_document(document_id)
    except Exception as exc:
        _raise_document_http_error(exc)


@router.post("/{document_id}/rechunk", response_model=KnowledgeDocumentOperationResponse)
def rechunk_knowledge_document(
    document_id: str,
    payload: KnowledgeDocumentRechunkRequest,
    request: Request,
) -> object:
    """重建知识文档分块。"""
    service = _get_document_service(request)
    try:
        return service.rechunk_document(
            document_id=document_id,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            keep_version=payload.keep_version,
        )
    except Exception as exc:
        _raise_document_http_error(exc)


def _get_document_service(request: Request) -> KnowledgeDocumentServiceProtocol:
    """从应用状态读取服务，缺失时懒加载。"""
    service = getattr(request.app.state, "knowledge_document_service", None)
    if service is not None:
        return service
    try:
        service = KnowledgeDocumentService()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "KNOWLEDGE_DOCUMENT_STORE_ERROR",
                "message": "Knowledge document backend is unavailable.",
            },
        ) from exc
    request.app.state.knowledge_document_service = service
    return service


def _raise_document_http_error(exc: Exception) -> None:
    """将服务层异常映射为结构化 HTTP 错误。"""
    if isinstance(exc, KnowledgeDocumentNotFoundError):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "KNOWLEDGE_DOCUMENT_NOT_FOUND",
                "message": str(exc),
            },
        ) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "KNOWLEDGE_DOCUMENT_VALIDATION_ERROR",
                "message": str(exc),
            },
        ) from exc
    if isinstance(exc, KnowledgeDocumentStoreError):
        raise HTTPException(
            status_code=500,
            detail={
                "code": "KNOWLEDGE_DOCUMENT_STORE_ERROR",
                "message": "Knowledge document backend is unavailable.",
            },
        ) from exc
    if isinstance(exc, KnowledgeDocumentError):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "KNOWLEDGE_DOCUMENT_ERROR",
                "message": str(exc),
            },
        ) from exc
    raise HTTPException(
        status_code=500,
        detail={
            "code": "KNOWLEDGE_DOCUMENT_INTERNAL_ERROR",
            "message": "Knowledge document operation failed.",
        },
    ) from exc


def _to_response_data(payload: object) -> object:
    """兼容 Pydantic 模型与普通字典响应。"""
    if hasattr(payload, "model_dump"):
        return payload.model_dump()  # type: ignore[attr-defined]
    return payload

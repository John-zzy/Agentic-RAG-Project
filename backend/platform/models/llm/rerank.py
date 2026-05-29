from __future__ import annotations

from collections.abc import Callable
from typing import Any

from backend.platform.models.base.router import RoutedRerankModel, get_rerank_model


class RerankWrapperFactory:
    """集中创建 LangChain DashScopeRerank wrapper。"""

    def __init__(
        self,
        *,
        client_factory: Callable[[], Any] | None = None,
        wrapper_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._wrapper_factory = wrapper_factory

    def create(self, routed_model: RoutedRerankModel | None = None) -> Any:
        model = routed_model or get_rerank_model()
        if model.provider.strip().lower() != "dashscope":
            raise ValueError(f"Unsupported rerank provider: {model.provider}")
        if not model.api_key:
            raise ValueError(f"Missing API key for rerank model: {model.model_name}")

        wrapper_cls = self._resolve_wrapper_factory()
        # LangChain wrapper 负责真实 query/documents 调用；此处只注入模型路由结果。
        return wrapper_cls(
            client=self._resolve_client(),
            model=model.model_name,
            top_n=model.top_n,
            api_key=model.api_key,
        )

    def _resolve_wrapper_factory(self) -> Callable[..., Any]:
        if self._wrapper_factory is not None:
            return self._wrapper_factory

        try:
            from langchain_community.document_compressors.dashscope_rerank import (
                DashScopeRerank,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "langchain-community is required for rerank model execution. "
                "Install backend/requirements.txt and retry."
            ) from exc

        self._wrapper_factory = DashScopeRerank
        return DashScopeRerank

    def _resolve_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()

        try:
            import dashscope
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "dashscope is required for rerank model execution. "
                "Install backend/requirements.txt and retry."
            ) from exc

        return dashscope.TextReRank


rerank_wrapper_factory = RerankWrapperFactory()


def get_rerank_wrapper() -> Any:
    """模块级快捷入口，返回模型路由管理的 LangChain rerank wrapper。"""
    return rerank_wrapper_factory.create()

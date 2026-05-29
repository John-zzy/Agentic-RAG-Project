from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from backend.platform.models.base.router import RoutedEmbeddingModel, get_embedding_model


class TextEmbeddingClient(Protocol):
    """DashScope TextEmbedding 客户端最小调用契约。"""

    def call(self, **kwargs: Any) -> Any:
        ...


class DashScopeEmbeddingStrategy:
    """基于模型路由配置执行 DashScope embedding 调用。"""

    def __init__(
        self,
        *,
        routed_model: RoutedEmbeddingModel,
        client: TextEmbeddingClient,
    ) -> None:
        self.provider = routed_model.provider
        self.model_name = routed_model.model_name
        self.api_base = routed_model.api_base
        self.api_key = routed_model.api_key
        self.api_key_env = routed_model.api_key_env
        self.dimensions = routed_model.dimensions
        self.timeout_seconds = routed_model.timeout_seconds
        self.max_retries = routed_model.max_retries
        self._client = client

    def embed(self, text: str) -> list[float]:
        """把文本转换为向量；缺失凭证时显式失败。"""
        if not self.api_key:
            raise ValueError(f"Missing API key for embedding model: {self.model_name}")

        response = self._client.call(
            model=self.model_name,
            input=text,
            text_type="query",
            dimension=self.dimensions,
            api_key=self.api_key,
        )
        self._ensure_successful_response(response)
        embedding = self._extract_embedding(response)
        if len(embedding) != self.dimensions:
            raise ValueError(
                f"Embedding model returned {len(embedding)} dimensions, expected {self.dimensions}"
            )
        return embedding

    def _ensure_successful_response(self, response: Any) -> None:
        """识别 DashScope API 失败响应，保留服务端返回的错误信息。"""
        status_code = getattr(response, "status_code", None)
        if status_code is None or int(status_code) == 200:
            return

        code = getattr(response, "code", None)
        message = getattr(response, "message", None)
        raise ValueError(
            "Embedding model call failed: "
            f"status_code={status_code}, code={code or 'unknown'}, message={message or 'unknown'}"
        )

    def _extract_embedding(self, response: Any) -> list[float]:
        """按 DashScope TextEmbedding 响应结构提取首个向量。"""
        output = getattr(response, "output", None)
        if not isinstance(output, dict):
            raise ValueError("Embedding model returned invalid output")

        embeddings = output.get("embeddings")
        if not embeddings:
            raise ValueError("Embedding model returned empty embeddings")

        first_embedding = embeddings[0]
        if not isinstance(first_embedding, dict):
            raise ValueError("Embedding model returned invalid embedding item")

        vector = first_embedding.get("embedding")
        if not vector:
            raise ValueError("Embedding model returned empty vector")
        return [float(value) for value in vector]


class EmbeddingStrategyFactory:
    """集中创建 embedding strategy"""

    def __init__(self, client_factory: Callable[[], Any] | None = None) -> None:
        self._client_factory = client_factory

    def create(self, routed_model: RoutedEmbeddingModel | None = None) -> DashScopeEmbeddingStrategy:
        model = routed_model or get_embedding_model()
        if model.provider.strip().lower() != "dashscope":
            raise ValueError(f"Unsupported embedding provider: {model.provider}")
        if not model.api_key:
            raise ValueError(f"Missing API key for embedding model: {model.model_name}")

        return DashScopeEmbeddingStrategy(
            routed_model=model,
            client=self._resolve_client(),
        )

    def _resolve_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()

        try:
            import dashscope
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "dashscope is required for embedding model execution. "
                "Install backend/requirements.txt and retry."
            ) from exc

        return dashscope.TextEmbedding


embedding_strategy_factory = EmbeddingStrategyFactory()


def get_embedding_strategy() -> DashScopeEmbeddingStrategy:
    """模块级快捷入口，返回模型路由管理的 embedding strategy。"""
    return embedding_strategy_factory.create()

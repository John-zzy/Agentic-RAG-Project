from dataclasses import dataclass
from typing import Literal

from backend.platform.config.settings import (
    EmbeddingModelConfig,
    ModelEndpointConfig,
    ModelRoutingConfig,
    RerankModelConfig,
    settings,
)

TaskComplexity = Literal["simple", "moderate", "complex"]


@dataclass(frozen=True)
class RoutedModel:
    """描述一次路由后的模型配置，供调用层直接消费。"""

    complexity: TaskComplexity
    provider: str
    model_name: str
    api_base: str | None
    api_key: str | None
    supports_streaming: bool
    timeout_seconds: int
    max_tokens: int
    temperature: float

    @classmethod
    def from_config(
        cls,
        complexity: TaskComplexity,
        config: ModelEndpointConfig,
    ) -> "RoutedModel":
        """将配置对象转换为可直接路由的模型描述。"""
        return cls(
            complexity=complexity,
            provider=config.provider,
            model_name=config.model_name,
            api_base=config.api_base,
            api_key=config.api_key,
            supports_streaming=config.supports_streaming,
            timeout_seconds=config.timeout_seconds,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )


@dataclass(frozen=True)
class RoutedEmbeddingModel:
    """描述 embedding 模型路由结果，供向量化工厂消费。"""

    provider: str
    model_name: str
    api_base: str | None
    api_key_env: str | None
    api_key: str | None
    dimensions: int
    timeout_seconds: int
    max_retries: int

    @classmethod
    def from_config(cls, config: EmbeddingModelConfig) -> "RoutedEmbeddingModel":
        return cls(
            provider=config.provider,
            model_name=config.model_name,
            api_base=config.api_base,
            api_key_env=config.api_key_env,
            api_key=config.api_key,
            dimensions=config.dimensions,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )


@dataclass(frozen=True)
class RoutedRerankModel:
    """描述 rerank 模型路由结果，供重排工厂消费。"""

    provider: str
    model_name: str
    api_base: str | None
    api_key_env: str | None
    api_key: str | None
    top_n: int
    timeout_seconds: int

    @classmethod
    def from_config(cls, config: RerankModelConfig) -> "RoutedRerankModel":
        return cls(
            provider=config.provider,
            model_name=config.model_name,
            api_base=config.api_base,
            api_key_env=config.api_key_env,
            api_key=config.api_key,
            top_n=config.top_n,
            timeout_seconds=config.timeout_seconds,
        )


class ModelRouter:
    """维护模型用途到具体模型配置的选择逻辑。"""

    def __init__(self, config: ModelRoutingConfig | None = None) -> None:
        """加载全局模型路由配置。"""
        self._config = config or settings.models

    def select(self, complexity: TaskComplexity) -> RoutedModel:
        """按复杂度选择模型配置。"""
        config = getattr(self._config, complexity)
        return RoutedModel.from_config(complexity, config)

    def select_embedding(self) -> RoutedEmbeddingModel:
        """选择文档向量化模型配置。"""
        return RoutedEmbeddingModel.from_config(self._config.embedding)

    def select_rerank(self) -> RoutedRerankModel:
        """选择检索重排模型配置。"""
        return RoutedRerankModel.from_config(self._config.rerank)

router = ModelRouter()


def get_model_for_task(complexity: TaskComplexity) -> RoutedModel:
    """对外暴露的模型路由入口。"""
    return router.select(complexity)


def get_embedding_model() -> RoutedEmbeddingModel:
    """对外暴露的 embedding 模型路由入口。"""
    return router.select_embedding()


def get_rerank_model() -> RoutedRerankModel:
    """对外暴露的 rerank 模型路由入口。"""
    return router.select_rerank()
